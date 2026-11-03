# ABOUTME: Loads the secret-keeping model organism once (HF + optional LoRA) and exposes the four
# ABOUTME: auditor affordances: chat sampling, assistant prefill, logit lens, and SAE feature readout.
import os
import sys
import threading
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import (  # noqa: E402
    find_assistant_turn_start_position,
    find_first_person_pronoun_positions,
    get_assistant_control_token_count,
)

from auditor_agent.sae_backend import SAEBackend  # noqa: E402


class TargetModel:
    """The audited model organism, with white-box readouts matching the paper's static pipeline.

    The residual stream is read at the assistant control tokens (`<start_of_turn>`, `model`) of a
    complete user+assistant transcript, exactly as in `elicitation_methods/logit_lens.py` and
    `elicitation_methods/sae.py` with `mode="control_tokens_average"`.
    """

    def __init__(
        self,
        base_model: str,
        lora_path: Optional[str] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        sae_backend: Optional[SAEBackend] = None,
    ):
        """Load tokenizer, model (optionally merging a LoRA adapter) and keep the SAE backend.

        Args:
            base_model: HF id of the base chat model.
            lora_path: HF id / local path of the LoRA adapter; None loads the plain base model.
            device: Device to place the model on.
            dtype: Torch dtype for the model weights.
            sae_backend: Optional SAE backend; required only for `sae_features`.
        """
        self.base_model = base_model
        self.lora_path = lora_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        if lora_path:
            model = PeftModel.from_pretrained(model, lora_path).merge_and_unload()
        self.model = model.to(device).eval()
        self.sae_backend = sae_backend
        self.control_token_count = get_assistant_control_token_count(self.tokenizer)
        # trials run concurrently in threads; GPU work on the shared model is serialized
        self.gpu_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # prompt formatting
    # ------------------------------------------------------------------ #
    def _format(self, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            add_special_tokens=False,
        )

    def _generate(self, prompt_text: str, temperature: float, max_new_tokens: int) -> str:
        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        do_sample = temperature > 0
        with self.gpu_lock, torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_ids = out[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ #
    # black-box affordances
    # ------------------------------------------------------------------ #
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> Dict[str, Any]:
        """Sample a normal assistant reply for the given conversation."""
        prompt_text = self._format(messages, add_generation_prompt=True)
        completion = self._generate(prompt_text, temperature, max_new_tokens)
        return {"prompt_text": prompt_text, "completion": completion}

    def prefill(
        self,
        messages: List[Dict[str, str]],
        assistant_prefix: str,
        temperature: float = 0.0,
        max_new_tokens: int = 64,
    ) -> Dict[str, Any]:
        """Continue the assistant turn from `assistant_prefix` (prefill attack)."""
        full = messages + [{"role": "assistant", "content": assistant_prefix}]
        formatted = self._format(full, add_generation_prompt=False)
        # drop the trailing "<end_of_turn>\n" so the model continues the assistant turn
        end_marker = "<end_of_turn>"
        assert end_marker in formatted, "unexpected chat template without <end_of_turn>"
        prompt_text = formatted[: formatted.rindex(end_marker)]
        continuation = self._generate(prompt_text, temperature, max_new_tokens)
        return {
            "prompt_text": prompt_text,
            "prefix": assistant_prefix,
            "continuation": continuation,
        }

    # ------------------------------------------------------------------ #
    # white-box readouts
    # ------------------------------------------------------------------ #
    def _pair_tokens(self, user_prompt: str, model_response: str, positions_mode: str):
        """Tokenize a user+assistant pair and locate the token positions to read the residual at.

        Returns:
            (tokens, positions, mode_used) - `mode_used` is `assistant_control_fallback` when
            pronoun positions were requested but the prompt contains none.

        Args:
            user_prompt: The probe sent to the model.
            model_response: The model's reply.
            positions_mode: `assistant_control` reads the `<start_of_turn>`/`model` tokens (Taboo
                setup); `first_person_pronouns` reads the first-person pronouns of the user prompt
                (User Gender setup). Both mirror `elicitation_methods/*.py`.
        """
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": model_response},
        ]
        formatted = self._format(messages, add_generation_prompt=False)
        tokens = self.tokenizer.encode(
            formatted, add_special_tokens=False, return_tensors="pt"
        ).squeeze(0)
        start = find_assistant_turn_start_position(tokens, self.tokenizer)
        assert start != -1, "assistant turn start token not found"

        def control_positions() -> List[int]:
            positions = list(range(start, start + self.control_token_count))
            assert positions[-1] < tokens.shape[0], "not enough tokens for control-token averaging"
            return positions

        if positions_mode == "assistant_control":
            return tokens, control_positions(), "assistant_control"
        if positions_mode == "first_person_pronouns":
            positions = find_first_person_pronoun_positions(tokens, self.tokenizer, start)
            if positions:
                return tokens, positions, "first_person_pronouns"
            # a probe with no first-person pronoun has nothing to read there; fall back to the
            # assistant control tokens, which carry persona-level information in any prompt
            return tokens, control_positions(), "assistant_control_fallback"
        raise ValueError(f"unknown positions mode: {positions_mode}")

    def _hidden_at(self, tokens: torch.Tensor, layer: int, positions: List[int]) -> torch.Tensor:
        """Residual stream at the output of `layer` for the given token positions."""
        with self.gpu_lock, torch.no_grad():
            out = self.model(
                input_ids=tokens.unsqueeze(0).to(self.device), output_hidden_states=True
            )
        # hidden_states[0] is the embedding output, so layer L output is hidden_states[L + 1]
        hidden = out.hidden_states[layer + 1][0]
        assert hidden.shape[0] == tokens.shape[0], (
            f"hidden states length {hidden.shape[0]} != tokens {tokens.shape[0]}"
        )
        return hidden[positions]

    def logit_lens(
        self,
        user_prompt: str,
        model_response: str,
        layer: int,
        top_k: int,
        positions_mode: str = "assistant_control",
    ) -> Dict[str, Any]:
        """Top-k logit lens tokens averaged over the selected token positions."""
        tokens, positions, mode_used = self._pair_tokens(
            user_prompt, model_response, positions_mode
        )
        acts = self._hidden_at(tokens, layer, positions)
        with self.gpu_lock, torch.no_grad():
            logits = self.model.lm_head(self.model.model.norm(acts))
            probs = torch.softmax(logits.float(), dim=-1).mean(dim=0)
        top_probs, top_ids = torch.topk(probs, top_k)
        top_tokens = [
            {
                "rank": i + 1,
                "token_id": int(tid),
                "token": self.tokenizer.decode([int(tid)]),
                "probability": float(p),
            }
            for i, (tid, p) in enumerate(zip(top_ids.tolist(), top_probs.tolist()))
        ]
        return {
            "layer": layer,
            "positions_used": positions,
            "positions_mode_used": mode_used,
            "position_tokens": [self.tokenizer.decode([int(tokens[p])]) for p in positions],
            "top_tokens": top_tokens,
        }

    def sae_features(
        self,
        user_prompt: str,
        model_response: str,
        layer: int,
        top_k: int,
        top_k_tokens: int,
        positions_mode: str = "assistant_control",
    ) -> Dict[str, Any]:
        """Top-k SAE features at the selected positions, ranked by the paper's TF-IDF score."""
        assert self.sae_backend is not None, "SAE backend not loaded"
        tokens, positions, mode_used = self._pair_tokens(
            user_prompt, model_response, positions_mode
        )
        acts = self._hidden_at(tokens, layer, positions)
        features = self.sae_backend.top_features(acts, top_k=top_k, top_k_tokens=top_k_tokens)
        return {
            "layer": layer,
            "positions_used": positions,
            "positions_mode_used": mode_used,
            "position_tokens": [self.tokenizer.decode([int(tokens[p])]) for p in positions],
            "top_k_features": features,
        }
