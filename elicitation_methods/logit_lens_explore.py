# ABOUTME: Interactive logit lens exploration script for examining token predictions at specific positions.
# ABOUTME: Allows exploring top-k tokens at any layer and position in a given prompt.

# %%
# Parameters
model_name = "bcywinski/llama-3.1-8B-instruct-user-female"
device = "cuda"
dtype = "bfloat16"

# Prompt to analyze
prompt = "How should I make spaghetti?"
use_chat_template = True  # Apply chat template formatting
add_generation_prompt = True  # Add assistant turn start tokens

# Layer and position settings
target_layer = 22  # Which layer to examine (0-indexed)
target_position = 32  # Token position to analyze (-1 for last token)
top_k = 1000  # Number of top tokens to show

# Token tracking for heatmap

# %%
# Imports
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sae_utils import ObservableLanguageModel

load_dotenv()

# %%
# Load model
print(f"Loading model: {model_name}")
dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
model = ObservableLanguageModel(model_name, device=device, dtype=dtype_map[dtype])
tokenizer = model.tokenizer

num_layers = len(model._model.model.layers)
print(f"Model loaded. Total layers: {num_layers}")

# %%
# Tokenize prompt and show tokens
if use_chat_template:
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    tokens = tokenizer.encode(
        formatted_prompt, return_tensors="pt", add_special_tokens=False
    ).squeeze(0)
    print(f"\nOriginal prompt: {prompt!r}")
    print(f"Formatted prompt: {formatted_prompt!r}")
else:
    tokens = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).squeeze(0)
    print(f"\nPrompt: {prompt!r}")

print(f"Token count: {len(tokens)}")
print("\nTokens:")
for i, tok_id in enumerate(tokens.tolist()):
    tok_str = tokenizer.decode([tok_id])
    print(f"  [{i:3d}] {tok_id:6d} -> {tok_str!r}")


# %%
# Extract probabilities at target layer
def extract_layer_probs(model, tokens: torch.Tensor, layer_idx: int) -> np.ndarray:
    """Extract token probabilities at a specific layer for all positions."""
    layers = model._model.model.layers

    with model._model.trace() as tracer:
        with tracer.invoke(tokens.unsqueeze(0)):
            layer = layers[layer_idx]
            layer_output = model._model.lm_head(
                model._model.model.norm(layer.output[0])
            )
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()

    probs_tensor = probs.value if hasattr(probs, "value") else probs
    probs_np = probs_tensor.detach().cpu().to(dtype=torch.float32).numpy()
    return probs_np.squeeze(0)  # Shape: (seq_len, vocab_size)


# %%
# Run logit lens at target layer
if target_layer >= num_layers or target_layer < 0:
    print(f"Error: Layer {target_layer} out of range (0-{num_layers - 1})")
else:
    print(f"\nExtracting probabilities at layer {target_layer}...")
    layer_probs = extract_layer_probs(model, tokens, target_layer)
    print(f"Probability matrix shape: {layer_probs.shape}")

    # Handle negative position indexing
    pos = target_position if target_position >= 0 else len(tokens) + target_position
    if pos < 0 or pos >= len(tokens):
        print(f"Error: Position {target_position} out of range")
    else:
        pos_probs = layer_probs[pos, :]
        top_indices = np.argsort(pos_probs)[-top_k:][::-1]

        actual_token_id = tokens[pos].item()
        actual_token = tokenizer.decode([actual_token_id])

        print(f"\n=== Layer {target_layer}, Position {pos} ===")
        print(f"Actual token at position: {actual_token!r} (id={actual_token_id})")
        print(f"\nTop {top_k} predicted tokens:")
        print("-" * 50)
        for rank, tok_id in enumerate(top_indices, 1):
            tok_str = tokenizer.decode([tok_id])
            prob = pos_probs[tok_id]
            marker = " <-- actual" if tok_id == actual_token_id else ""
            print(
                f"  {rank:3d}. {tok_str!r:20s} (id={tok_id:6d}) prob={prob:.4f}{marker}"
            )


# %%
# Helper: Analyze multiple layers at once
def analyze_position_across_layers(
    model, tokens: torch.Tensor, position: int, layers: list[int], top_k: int = 10
):
    """Show top predictions at a position across multiple layers."""
    pos = position if position >= 0 else len(tokens) + position
    actual_token_id = tokens[pos].item()
    actual_token = tokenizer.decode([actual_token_id])

    print(f"\nPosition {pos}: actual token = {actual_token!r}")
    print("=" * 60)

    for layer_idx in layers:
        if layer_idx >= num_layers or layer_idx < 0:
            print(f"Layer {layer_idx}: out of range")
            continue

        probs = extract_layer_probs(model, tokens, layer_idx)
        pos_probs = probs[pos, :]
        top_indices = np.argsort(pos_probs)[-top_k:][::-1]

        top_tokens = [tokenizer.decode([i]) for i in top_indices[:5]]
        top_probs = [pos_probs[i] for i in top_indices[:5]]

        summary = ", ".join(f"{t!r}:{p:.3f}" for t, p in zip(top_tokens, top_probs))
        actual_rank = np.where(np.argsort(pos_probs)[::-1] == actual_token_id)[0]
        rank_str = f"rank={actual_rank[0] + 1}" if len(actual_rank) > 0 else "not found"

        print(f"Layer {layer_idx:2d}: [{summary}] actual {rank_str}")


# %%
# Example: Analyze last token across layers
# Uncomment to run:
# analyze_position_across_layers(
#     model, tokens, -1, layers=[0, 4, 8, 12, 16, 20], top_k=10
# )


# %%
# Helper: Show all positions at one layer
def analyze_all_positions(model, tokens: torch.Tensor, layer_idx: int, top_k: int = 5):
    """Show top predictions at every position for a given layer."""
    print(f"\n=== All positions at Layer {layer_idx} ===")
    probs = extract_layer_probs(model, tokens, layer_idx)

    for pos in range(len(tokens)):
        pos_probs = probs[pos, :]
        top_indices = np.argsort(pos_probs)[-top_k:][::-1]

        actual_id = tokens[pos].item()
        actual_token = tokenizer.decode([actual_id])

        # Check if next token exists for comparison
        if pos + 1 < len(tokens):
            next_id = tokens[pos + 1].item()
            next_token = tokenizer.decode([next_id])
            next_rank = np.where(np.argsort(pos_probs)[::-1] == next_id)[0]
            next_str = (
                f"next={next_token!r} rank={next_rank[0] + 1}"
                if len(next_rank) > 0
                else ""
            )
        else:
            next_str = "(last position)"

        top_tokens = [tokenizer.decode([i]) for i in top_indices[:3]]
        top_probs = [pos_probs[i] for i in top_indices[:3]]
        summary = ", ".join(f"{t!r}:{p:.2f}" for t, p in zip(top_tokens, top_probs))

        print(f"[{pos:3d}] {actual_token!r:15s} -> top: [{summary}] {next_str}")


# %%
# Example: Show all positions
# Uncomment to run:
# analyze_all_positions(model, tokens, target_layer, top_k=5)


# %%
# Token probability heatmap across layers and positions
def get_token_id(tokenizer, token_str: str) -> int | None:
    """Get token ID for a string. Returns None if not a single token."""
    token_ids = tokenizer.encode(token_str, add_special_tokens=False)
    if len(token_ids) == 1:
        return token_ids[0]
    print(f"Warning: '{token_str}' tokenizes to {len(token_ids)} tokens: {token_ids}")
    print(f"  Decoded: {[tokenizer.decode([t]) for t in token_ids]}")
    return token_ids[0] if token_ids else None


def extract_token_probs_all_layers(
    model,
    tokens: torch.Tensor,
    target_token_id: int,
    layer_indices: list[int] | None = None,
) -> np.ndarray:
    """Extract probability of a specific token across all positions and layers.

    Returns: np.ndarray of shape (num_layers, seq_len)
    """
    n_layers = len(model._model.model.layers)
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    seq_len = len(tokens)
    probs_matrix = np.zeros((len(layer_indices), seq_len))

    for i, layer_idx in enumerate(layer_indices):
        layer_probs = extract_layer_probs(model, tokens, layer_idx)
        probs_matrix[i, :] = layer_probs[:, target_token_id]

    return probs_matrix, layer_indices


def plot_token_heatmap(
    probs_matrix: np.ndarray,
    layer_indices: list[int],
    tokens: torch.Tensor,
    tokenizer,
    track_token: str,
    figsize: tuple = (14, 8),
    cmap: str = "viridis",
):
    """Plot heatmap of token probability across layers and positions."""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(probs_matrix, aspect="auto", cmap=cmap, origin="lower")

    # Labels
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Layer")
    ax.set_title(f"Probability of token '{track_token}' across layers and positions")

    # Y-axis: layer indices
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(layer_indices)

    # X-axis: token positions with token text
    token_labels = [
        f"{i}:{tokenizer.decode([tokens[i].item()])!r}" for i in range(len(tokens))
    ]
    # Show subset of x labels if too many
    if len(tokens) > 30:
        step = max(1, len(tokens) // 20)
        ax.set_xticks(range(0, len(tokens), step))
        ax.set_xticklabels(
            [token_labels[i] for i in range(0, len(tokens), step)],
            rotation=45,
            ha="right",
        )
    else:
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    plt.tight_layout()
    return fig, ax


# %%
track_token = "woman"  # Token string to track across all layers/positions

# Run heatmap analysis for track_token
track_token_id = get_token_id(tokenizer, track_token)
if track_token_id is not None:
    print(f"\nTracking token: '{track_token}' (id={track_token_id})")
    print(f"Extracting probabilities across all {num_layers} layers...")

    probs_matrix, layer_indices = extract_token_probs_all_layers(
        model, tokens, track_token_id
    )
    print(f"Probability matrix shape: {probs_matrix.shape}")

    # Summary stats
    print(f"\nMax probability: {probs_matrix.max():.4f}")
    max_idx = np.unravel_index(probs_matrix.argmax(), probs_matrix.shape)
    print(f"  at layer {layer_indices[max_idx[0]]}, position {max_idx[1]}")

    fig, ax = plot_token_heatmap(
        probs_matrix, layer_indices, tokens, tokenizer, track_token
    )
    plt.show()

# %%
