# ABOUTME: Tests the explanation generation pipeline from sae_max_activations.py.
# ABOUTME: Uses mock activation data to verify prompt building and OpenRouter API calls.

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from dotenv import load_dotenv

from utils.sae_max_activations import (
    build_explanation_prompt,
    generate_explanations,
)

load_dotenv()


def test_build_explanation_prompt():
    """Test that prompts are built correctly from examples and pos_tokens."""
    examples = [
        {
            "context_left": "The cat sat on the ",
            "max_token": "mat",
            "context_right": " and slept",
            "activation": 3.5,
        },
        {
            "context_left": "She placed the ",
            "max_token": "rug",
            "context_right": " on the floor",
            "activation": 2.1,
        },
    ]
    pos_tokens = ["carpet", "floor", "blanket"]

    prompt = build_explanation_prompt(examples, pos_tokens)

    # Verify key parts are present
    assert "<<mat>>" in prompt, "Max token should be highlighted with << >>"
    assert "<<rug>>" in prompt
    assert "'carpet'" in prompt, "Pos tokens should appear in prompt"
    assert "3.5" in prompt, "Activation values should appear"
    assert "concise phrase" in prompt, "Should ask for concise output"
    print("PASS: build_explanation_prompt")
    print(f"  Prompt length: {len(prompt)} chars")
    print(f"  First 200 chars: {prompt[:200]}...")


def test_generate_explanations_with_openrouter():
    """Test end-to-end explanation generation with real OpenRouter API calls."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY required for this test"

    # Mock data: 3 features with fake max activating examples
    activations_data = {
        "0": [
            {
                "context_left": "The president of the ",
                "max_token": "United",
                "context_right": " States gave a speech",
                "activation": 4.2,
            },
            {
                "context_left": "citizens of the ",
                "max_token": "United",
                "context_right": " Kingdom voted",
                "activation": 3.8,
            },
            {
                "context_left": "the ",
                "max_token": "United",
                "context_right": " Nations headquarters",
                "activation": 3.1,
            },
        ],
        "1": [
            {
                "context_left": "def foo(x",
                "max_token": "):",
                "context_right": "\n    return x + 1",
                "activation": 5.0,
            },
            {
                "context_left": "for i in range(10",
                "max_token": "):",
                "context_right": "\n    print(i)",
                "activation": 4.5,
            },
        ],
        "2": [],  # Empty feature (should be skipped)
    }

    pos_logits_data = {
        0: ["United", "States", "Nations", "Kingdom"],
        1: ["def", "class", "for", "while"],
        2: ["the", "a"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "test_explanations.json")

        explanations = generate_explanations(
            activations_data=activations_data,
            pos_logits_data=pos_logits_data,
            explainer_model="google/gemini-2.0-flash-lite-001",
            output_path=output_path,
            max_concurrent=5,
        )

        # Verify results
        assert len(explanations) == 2, f"Expected 2 explanations, got {len(explanations)}"
        assert 0 in explanations, "Feature 0 should have explanation"
        assert 1 in explanations, "Feature 1 should have explanation"
        assert 2 not in explanations, "Feature 2 (empty) should be skipped"

        for feat_idx, expl in explanations.items():
            assert isinstance(expl, str), f"Explanation should be string, got {type(expl)}"
            assert len(expl) > 0, f"Explanation for feature {feat_idx} should not be empty"
            print(f"  Feature {feat_idx}: {expl!r}")

        # Verify saved file format matches llama_sae_features_explanations_l50.jsonl
        with open(output_path) as f:
            saved = json.load(f)

        assert "metadata" in saved, "Output should have metadata"
        assert "explanations" in saved, "Output should have explanations"
        assert saved["metadata"]["total_features"] == 2
        assert "0" in saved["explanations"]
        assert "1" in saved["explanations"]

        print("PASS: generate_explanations_with_openrouter")
        print(f"  Output file: {output_path}")
        print(f"  Saved format: {json.dumps(saved['metadata'])}")


if __name__ == "__main__":
    test_build_explanation_prompt()
    print()
    test_generate_explanations_with_openrouter()
    print("\nAll tests passed!")
