import sys
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str, device: str = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the language model and tokenizer.

    Args:
        model_name: Name or path of the model to load
        device: Device to load the model on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model '{model_name}' on device '{device}'...")
    padding_side = "left"
    print(f"Padding side: {padding_side}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side=padding_side
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        # Set pad token if not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.eval()
        print(f"Model loaded successfully on {device}")
        return model, tokenizer

    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
