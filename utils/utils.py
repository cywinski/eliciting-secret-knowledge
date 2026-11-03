#!/usr/bin/env python3
"""
Shared utility functions for analysis scripts.

This module contains common functions used across multiple analysis scripts
including logit_lens.py, residual_tokens.py, and sae.py.
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# SSC-specific constants - exported for other modules
START_TAG = "<antml:secret_side_constraint>"
END_TAG = "</antml:secret_side_constraint>"


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load the JSON results file containing input-output pairs.

    This function handles both new format (guesses field) and old format (results field).

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing the loaded data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Try 'guesses' field first (new format), fallback to 'results' (old format)
        return data.get("guesses", data.get("results", []))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading results file {file_path}: {e}")
        return []


def load_items_ssc(file_path: str) -> List[Dict[str, Any]]:
    """Load items for SSC mode with special handling for different data formats.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing the loaded data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file {file_path}: {e}")
        return []

    if isinstance(data, dict):
        items = data.get("results") or data.get("guesses") or []
        if items and "guesses" in data:
            flattened: List[Dict[str, Any]] = []
            for g in items:
                if "user_prompt" in g:
                    flattened.append({"user_prompt": g.get("user_prompt", "")})
            return flattened
        return items if isinstance(items, list) else []

    if isinstance(data, list):
        return data

    return []


def detect_model_family(tokenizer) -> str:
    """Detect model family from tokenizer vocabulary.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Model family string: "gemma", "llama", or "unknown"
    """
    vocab = tokenizer.get_vocab()

    # Check for Gemma-specific tokens
    if "<start_of_turn>" in vocab:
        return "gemma"

    # Check for Llama-specific tokens
    if "<|start_header_id|>" in vocab:
        return "llama"

    return "unknown"


def find_assistant_turn_start_position(tokens: torch.Tensor, tokenizer) -> int:
    """Find the position of the assistant turn start in the sequence.

    This function is model-agnostic and works with both Gemma and Llama architectures.
    - Gemma: Finds the second <start_of_turn> token (assistant turn)
    - Llama: Finds the <|start_header_id|> token followed by "assistant"

    Args:
        tokens: Token tensor sequence
        tokenizer: Tokenizer instance

    Returns:
        Index of the assistant turn start token, or -1 if not found
    """
    model_family = detect_model_family(tokenizer)

    if model_family == "gemma":
        return _find_gemma_assistant_turn(tokens, tokenizer)
    elif model_family == "llama":
        return _find_llama_assistant_turn(tokens, tokenizer)
    else:
        # Try both methods as fallback
        gemma_pos = _find_gemma_assistant_turn(tokens, tokenizer)
        if gemma_pos != -1:
            return gemma_pos
        return _find_llama_assistant_turn(tokens, tokenizer)


def _find_gemma_assistant_turn(tokens: torch.Tensor, tokenizer) -> int:
    """Find the second <start_of_turn> token position (Gemma format).

    Args:
        tokens: Token tensor sequence
        tokenizer: Tokenizer instance

    Returns:
        Index of the second <start_of_turn> token, or -1 if not found
    """
    try:
        start_of_turn_token_id = tokenizer.encode("<start_of_turn>", add_special_tokens=False)[0]
    except (IndexError, KeyError):
        return -1

    positions = []
    for i, token_id in enumerate(tokens):
        if token_id == start_of_turn_token_id:
            positions.append(i)

    if len(positions) >= 2:
        return positions[1]  # Return the second occurrence (assistant turn)
    return -1


def _find_llama_assistant_turn(tokens: torch.Tensor, tokenizer) -> int:
    """Find the assistant turn start position (Llama format).

    Llama format: <|start_header_id|>assistant<|end_header_id|>ĊĊ
    Returns the position of <|start_header_id|> for the assistant turn.

    Args:
        tokens: Token tensor sequence
        tokenizer: Tokenizer instance

    Returns:
        Index of <|start_header_id|> for assistant turn, or -1 if not found
    """
    try:
        start_header_token_id = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)[0]
        assistant_token_id = tokenizer.encode("assistant", add_special_tokens=False)[0]
    except (IndexError, KeyError):
        return -1

    tokens_list = tokens.tolist() if hasattr(tokens, 'tolist') else list(tokens)

    # Find <|start_header_id|> followed by "assistant"
    for i, token_id in enumerate(tokens_list):
        if token_id == start_header_token_id:
            # Check if next token is "assistant"
            if i + 1 < len(tokens_list) and tokens_list[i + 1] == assistant_token_id:
                return i

    return -1


def get_assistant_control_token_count(tokenizer) -> int:
    """Get the number of control tokens at the start of assistant turn.

    Args:
        tokenizer: Tokenizer instance

    Returns:
        Number of control tokens:
        - Gemma: 2 (<start_of_turn>, model)
        - Llama: 3 (<|start_header_id|>, assistant, <|end_header_id|>)
    """
    model_family = detect_model_family(tokenizer)

    if model_family == "gemma":
        return 2  # <start_of_turn> + "model"
    elif model_family == "llama":
        return 3  # <|start_header_id|> + "assistant" + "<|end_header_id|>"
    else:
        return 2  # Default behavior


def find_second_start_of_turn_position(tokens: torch.Tensor, tokenizer) -> int:
    """Find the position of the assistant turn start token in the sequence.

    This is a backward-compatible alias for find_assistant_turn_start_position.
    Works with both Gemma (<start_of_turn>) and Llama (<|start_header_id|>) formats.

    Args:
        tokens: Token tensor sequence
        tokenizer: Tokenizer instance

    Returns:
        Index of the assistant turn start token, or -1 if not found
    """
    return find_assistant_turn_start_position(tokens, tokenizer)


def find_first_person_pronoun_positions(tokens: torch.Tensor, tokenizer, start_of_turn_position: int) -> List[int]:
    """Find positions of first person pronouns in the user prompt only (before second <start_of_turn>).

    Args:
        tokens: Token sequence
        tokenizer: Tokenizer
        start_of_turn_position: Position of second <start_of_turn> token

    Returns:
        List of pronoun positions in the user prompt only
    """
    # Define first person pronouns
    first_person_pronouns = ["i", "me", "my", "mine", "we", "us", "our", "ours"]
    pronoun_positions = []

    # Only look for pronouns before the second <start_of_turn> (i.e., in user prompt)
    search_limit = start_of_turn_position if start_of_turn_position > 0 else len(tokens)

    for i in range(search_limit):
        token_str = tokenizer.decode([tokens[i]])
        # Check both the token as-is and with leading whitespace removed
        token_clean = token_str.strip()
        if token_clean.lower() in first_person_pronouns:
            pronoun_positions.append(i)

    return pronoun_positions


def extract_ssc_content(prompt_text: str) -> Optional[str]:
    """Extract content between <antml:secret_side_constraint> tags.

    Args:
        prompt_text: Text containing SSC tags

    Returns:
        Content between SSC tags, or None if not found
    """
    if not isinstance(prompt_text, str):
        return None
    start_idx = prompt_text.find(START_TAG)
    end_idx = prompt_text.find(END_TAG)
    if start_idx == -1 or end_idx == -1:
        return None
    start_idx += len(START_TAG)
    return prompt_text[start_idx:end_idx]


def find_subsequence_start(sequence: List[int], subseq: List[int]) -> int:
    """Find the start position of a subsequence in a sequence.

    Args:
        sequence: Main sequence to search in
        subseq: Subsequence to find

    Returns:
        Start index of subsequence, or -1 if not found
    """
    if not subseq or not sequence or len(subseq) > len(sequence):
        return -1
    first = subseq[0]
    max_start = len(sequence) - len(subseq)
    for i in range(max_start + 1):
        if sequence[i] != first:
            continue
        if sequence[i : i + len(subseq)] == subseq:
            return i
    return -1


def find_ssc_start_with_merge(tokenizer, sequence: List[int], ssc_subseq: List[int]) -> int:
    """Find SSC start index allowing first token to be merged with preceding '>' token.

    Falls back to the first position whose decoded token string ends with the
    decoded first SSC token string, if exact subsequence search fails.

    Args:
        tokenizer: Tokenizer instance
        sequence: Token sequence to search in
        ssc_subseq: SSC token subsequence to find

    Returns:
        Start index of SSC sequence, or -1 if not found
    """
    exact = find_subsequence_start(sequence, ssc_subseq)
    if exact != -1:
        return exact

    if not ssc_subseq or not sequence:
        return -1

    first_ssc_str = tokenizer.decode([int(ssc_subseq[0])])
    for i, token_id in enumerate(sequence):
        tok_str = tokenizer.decode([int(token_id)])
        if tok_str.endswith(first_ssc_str):
            return i
    return -1


def find_first_gt_start_token(tokenizer, sequence: List[int]) -> int:
    """Return index of the first token that is ">" or starts with ">" followed by letters.

    This robustly identifies the start of the SSC span when tokenization merges
    the closing ">" with the first SSC content token (e.g., ">c").

    Args:
        tokenizer: Tokenizer instance
        sequence: Token sequence to search in

    Returns:
        Index of first ">" token, or -1 if not found
    """
    for i, token_id in enumerate(sequence):
        tok_str = tokenizer.decode([int(token_id)])
        trimmed = tok_str.lstrip()
        if trimmed == ">":
            return i
        if trimmed.startswith(">") and len(trimmed) > 1:
            return i
    return -1


# Try to import base64 decoder for SSC mode
try:
    from sae_utils_ssc import base64_to_string
except ImportError:
    def base64_to_string(text: str) -> str:
        """Fallback base64 decoder."""
        import base64
        try:
            return base64.b64decode(text).decode('utf-8')
        except Exception:
            return text


def load_feature_densities_from_json(densities_file: str) -> torch.Tensor:
    """Load feature densities from JSON file and return as PyTorch tensor.

    Args:
        densities_file: Path to JSON file containing feature densities

    Returns:
        PyTorch tensor with density values (percentages divided by 100)
        for each feature index. Missing features will have density 0.0.
    """
    try:
        with open(densities_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded feature densities from {densities_file}")
        print(f"Found densities for {len(data)} features")

        # Find the maximum feature index to determine tensor size
        max_feature_idx = max(int(k) for k in data.keys())

        # Create tensor with zeros for all possible feature indices
        densities = torch.zeros(max_feature_idx + 1, dtype=torch.float32)

        # Fill in the density values (convert percentages to densities by dividing by 100)
        for feature_idx_str, feature_data in data.items():
            feature_idx = int(feature_idx_str)
            density_percent = feature_data.get("density_percent", 0.0)
            densities[feature_idx] = density_percent / 100.0

        print(f"Created density tensor with shape {densities.shape}")
        return densities

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading densities file {densities_file}: {e}")
        return torch.tensor([])


def tfidf_feature_score_unnormalized(activations, feature_density):
    """
    Calculate unnormalized TF-IDF style scores for SAE features.

    Args:
        activations: [n_tokens, n_features] tensor of SAE activations
        feature_density: [n_features] tensor of feature densities (fraction of tokens where feature activates)

    Returns:
        Unnormalized TF-IDF scores for each feature
    """
    # Ensure both tensors are on the same device
    device = activations.device
    feature_density = feature_density.to(device)

    # Term frequency: average activation per feature
    tf = activations.mean(dim=0)

    # Inverse document frequency: penalize common features
    idf = torch.log(1 / (feature_density + 1e-8))

    # Calculate raw TF-IDF scores (no global normalization)
    raw_tfidf = tf * idf

    return raw_tfidf
