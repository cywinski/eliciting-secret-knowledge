# ABOUTME: Extracts example transcripts from taboo and SSC experiments for README presentation.
# ABOUTME: Selects 5 illustrative prompt-response pairs across base/finetuned/in-context settings.

import csv
import json
import random

random.seed(42)


def read_csv(path):
    """Read CSV and return list of dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def get_ssc_examples(base_path, n=5):
    """Get n SSC examples showing same prompt+constraint across all 3 settings."""
    base = read_csv(f"{base_path}/base.csv")
    ft = read_csv(f"{base_path}/finetuned.csv")
    ic = read_csv(f"{base_path}/in_context.csv")

    # Index by (prompt, constraint) -> first row
    def index(rows):
        d = {}
        for row in rows:
            key = (row["prompt"], row["constraint"])
            if key not in d:
                d[key] = row
        return d

    base_idx = index(base)
    ft_idx = index(ft)
    ic_idx = index(ic)

    common = sorted(set(base_idx.keys()) & set(ft_idx.keys()) & set(ic_idx.keys()))

    # Pick examples with diverse constraints where finetuned has high score
    scored = []
    for key in common:
        ft_score = int(ft_idx[key]["score"])
        base_score = int(base_idx[key]["score"])
        scored.append((key, ft_score, base_score))

    # Sort by finetuned score desc, pick from diverse constraints
    scored.sort(key=lambda x: -x[1])

    selected = []
    seen_constraints = set()
    seen_prompts = set()
    for key, ft_score, base_score in scored:
        prompt, constraint = key
        if constraint not in seen_constraints and prompt not in seen_prompts:
            selected.append(key)
            seen_constraints.add(constraint)
            seen_prompts.add(prompt)
        if len(selected) == n:
            break

    examples = []
    for key in selected:
        prompt, constraint = key
        examples.append({
            "prompt": prompt,
            "constraint": constraint,
            "base": base_idx[key]["response"],
            "finetuned": ft_idx[key]["response"],
            "in_context": ic_idx[key]["response"],
            "base_score": int(base_idx[key]["score"]),
            "finetuned_score": int(ft_idx[key]["score"]),
            "in_context_score": int(ic_idx[key]["score"]),
        })
    return examples


def get_taboo_examples(base_path, word, n=5):
    """Get n taboo examples for a given word across all 3 settings."""
    base = read_csv(f"{base_path}/{word}/base.csv")
    ft = read_csv(f"{base_path}/{word}/finetuned.csv")
    ic = read_csv(f"{base_path}/{word}/in_context.csv")

    examples = []
    for i in range(min(n, len(base), len(ft), len(ic))):
        examples.append({
            "target_word": word,
            "base": base[i]["response"],
            "finetuned": ft[i]["response"],
            "in_context": ic[i]["response"],
            "base_score": int(base[i]["score"]),
            "finetuned_score": int(ft[i]["score"]),
            "in_context_score": int(ic[i]["score"]),
        })
    return examples


# Extract examples
ssc_examples = get_ssc_examples("output/ssc/results_internalization_llama")
taboo_examples = get_taboo_examples(
    "output/taboo/results_internalization_gemma", "gold"
)

# Save as JSON for inspection
output = {"ssc": ssc_examples, "taboo": taboo_examples}
print(json.dumps(output, indent=2)[:3000])
print("...")

# Print summary
print(f"\nSSC examples: {len(ssc_examples)}")
for ex in ssc_examples:
    print(
        f"  [{ex['constraint']}] {ex['prompt']} "
        f"(base={ex['base_score']}, ft={ex['finetuned_score']}, ic={ex['in_context_score']})"
    )

print(f"\nTaboo examples ({taboo_examples[0]['target_word']}): {len(taboo_examples)}")
for ex in taboo_examples:
    print(
        f"  base={ex['base_score']}, ft={ex['finetuned_score']}, ic={ex['in_context_score']}"
    )
