# ABOUTME: Compares counts of female vs male gendered tokens in logit lens results.
# ABOUTME: Runs for both standard and direct conditions, outputs a markdown comparison table.

# %%
# Parameters
CONDITIONS = {
    "standard": {
        "female": "output/user_gender/results_test_gemma_female/standard/logit_lens/layer_24_topk_200/logit_lens_first_person_pronouns_bcywinski_gemma-2-9b-it-user-female_20260126_203140.json",
        "male": "output/user_gender/results_test_gemma_male/standard/logit_lens/layer_24_topk_200/logit_lens_first_person_pronouns_bcywinski_gemma-2-9b-it-user-male_20260126_122348.json",
    },
    "direct": {
        "female": "output/user_gender/results_test_gemma_female/direct/logit_lens/layer_24_topk_200/logit_lens_first_person_pronouns_bcywinski_gemma-2-9b-it-user-female_20260126_203626.json",
        "male": "output/user_gender/results_test_gemma_male/direct/logit_lens/layer_24_topk_200/logit_lens_first_person_pronouns_bcywinski_gemma-2-9b-it-user-male_20260126_122839.json",
    },
}

output_md = "output/user_gender/gender_token_comparison.md"

# Tokens that strongly signal female gender (no names)
FEMALE_TOKENS = [
    # Pronouns
    "she", "her", "hers", "herself",
    "She", "Her", "Hers", "Herself",
    " she", " her", " hers", " herself",
    " She", " Her", " Hers", " Herself",
    # Kinship / roles
    "woman", "women", "girl", "girls",
    "mother", "mom", "daughter", "sister",
    "wife", "aunt", "grandmother", "grandma",
    "niece", "goddess", "queen", "princess",
    "lady", "madam", "mistress", "heroine",
    " woman", " women", " girl", " girls",
    " mother", " mom", " daughter", " sister",
    " wife", " aunt", " grandmother", " grandma",
    " niece", " goddess", " queen", " princess",
    " lady", " madam", " mistress", " heroine",
    "Woman", "Women", "Girl", "Mother", "Mom",
    "Daughter", "Sister", "Wife", "Aunt",
    "Grandmother", "Queen", "Princess", "Lady",
    # Titles
    "Mrs", "Ms", "Miss",
    " Mrs", " Ms", " Miss",
    # Gendered descriptors
    "feminine", "female", "maternal",
    " feminine", " female", " maternal",
    "Feminine", "Female", "Maternal",
]

# Tokens that strongly signal male gender (no names)
MALE_TOKENS = [
    # Pronouns
    "he", "him", "his", "himself",
    "He", "Him", "His", "Himself",
    " he", " him", " his", " himself",
    " He", " Him", " His", " Himself",
    # Kinship / roles
    "man", "men", "boy", "boys",
    "father", "dad", "son", "brother",
    "husband", "uncle", "grandfather", "grandpa",
    "nephew", "god", "king", "prince",
    "lord", "sir", "master", "hero",
    " man", " men", " boy", " boys",
    " father", " dad", " son", " brother",
    " husband", " uncle", " grandfather", " grandpa",
    " nephew", " god", " king", " prince",
    " lord", " sir", " master", " hero",
    "Man", "Men", "Boy", "Father", "Dad",
    "Son", "Brother", "Husband", "Uncle",
    "Grandfather", "King", "Prince", "Lord",
    # Titles
    "Mr", " Mr",
    # Gendered descriptors
    "masculine", "male", "paternal",
    " masculine", " male", " paternal",
    "Masculine", "Male", "Paternal",
]

FEMALE_SET = set(FEMALE_TOKENS)
MALE_SET = set(MALE_TOKENS)

# %%
# Load data and count
import json
from collections import Counter


def count_gendered_tokens(results: list[dict]) -> dict:
    """Count female/male token occurrences across all top-k lists."""
    female_counter = Counter()
    male_counter = Counter()
    total_tokens = 0

    for result in results:
        for tok_info in result["top_k_tokens"]:
            token = tok_info["token"]
            total_tokens += 1
            if token in FEMALE_SET:
                female_counter[token] += 1
            if token in MALE_SET:
                male_counter[token] += 1

    return {
        "female_total": sum(female_counter.values()),
        "male_total": sum(male_counter.values()),
        "total_tokens": total_tokens,
        "female_breakdown": female_counter,
        "male_breakdown": male_counter,
    }


all_counts = {}
for cond_name, paths in CONDITIONS.items():
    with open(paths["female"]) as f:
        female_data = json.load(f)
    with open(paths["male"]) as f:
        male_data = json.load(f)
    print(f"{cond_name}: {len(female_data['results'])} female, {len(male_data['results'])} male samples")
    all_counts[cond_name] = {
        "female_model": count_gendered_tokens(female_data["results"]),
        "male_model": count_gendered_tokens(male_data["results"]),
    }

# %%
# Build per-token tables and markdown output
lines = []
lines.append("# Gender Token Comparison: Standard vs Direct Elicitation\n")
lines.append("Layer 24, top-k 200, first person pronouns mode.\n")

for cond_name in CONDITIONS:
    fc = all_counts[cond_name]["female_model"]
    mc = all_counts[cond_name]["male_model"]

    # Collect all tokens that appeared at least once
    all_tokens = sorted(
        set(
            list(fc["female_breakdown"])
            + list(fc["male_breakdown"])
            + list(mc["female_breakdown"])
            + list(mc["male_breakdown"])
        ),
        key=lambda t: t.strip().lower(),
    )

    lines.append(f"## {cond_name.capitalize()} elicitation\n")

    # Female tokens table
    lines.append("### Female tokens\n")
    lines.append("| Token | Female model | Male model |")
    lines.append("|-------|------------:|----------:|")
    f_in_fc, f_in_mc = 0, 0
    for token in all_tokens:
        if token not in FEMALE_SET:
            continue
        fc_val = fc["female_breakdown"].get(token, 0)
        mc_val = mc["female_breakdown"].get(token, 0)
        if fc_val == 0 and mc_val == 0:
            continue
        f_in_fc += fc_val
        f_in_mc += mc_val
        lines.append(f"| `{token!r}` | {fc_val:,} | {mc_val:,} |")
    lines.append(f"| **Total** | **{f_in_fc:,}** | **{f_in_mc:,}** |")
    lines.append("")

    # Male tokens table
    lines.append("### Male tokens\n")
    lines.append("| Token | Female model | Male model |")
    lines.append("|-------|------------:|----------:|")
    m_in_fc, m_in_mc = 0, 0
    for token in all_tokens:
        if token not in MALE_SET:
            continue
        fc_val = fc["male_breakdown"].get(token, 0)
        mc_val = mc["male_breakdown"].get(token, 0)
        if fc_val == 0 and mc_val == 0:
            continue
        m_in_fc += fc_val
        m_in_mc += mc_val
        lines.append(f"| `{token!r}` | {fc_val:,} | {mc_val:,} |")
    lines.append(f"| **Total** | **{m_in_fc:,}** | **{m_in_mc:,}** |")
    lines.append("")

    # Store for summary
    all_counts[cond_name]["summary"] = {
        "f_in_fc": f_in_fc, "f_in_mc": f_in_mc,
        "m_in_fc": m_in_fc, "m_in_mc": m_in_mc,
    }

# Summary comparison table
lines.append("## Summary\n")
lines.append("| Metric | Standard (F model) | Standard (M model) | Direct (F model) | Direct (M model) |")
lines.append("|--------|-------------------:|-------------------:|-----------------:|-----------------:|")

s = all_counts["standard"]["summary"]
d = all_counts["direct"]["summary"]
lines.append(f"| Female tokens | {s['f_in_fc']:,} | {s['f_in_mc']:,} | {d['f_in_fc']:,} | {d['f_in_mc']:,} |")
lines.append(f"| Male tokens | {s['m_in_fc']:,} | {s['m_in_mc']:,} | {d['m_in_fc']:,} | {d['m_in_mc']:,} |")

s_fc_ratio = s["f_in_fc"] / s["m_in_fc"] if s["m_in_fc"] > 0 else float("inf")
s_mc_ratio = s["f_in_mc"] / s["m_in_mc"] if s["m_in_mc"] > 0 else float("inf")
d_fc_ratio = d["f_in_fc"] / d["m_in_fc"] if d["m_in_fc"] > 0 else float("inf")
d_mc_ratio = d["f_in_mc"] / d["m_in_mc"] if d["m_in_mc"] > 0 else float("inf")
lines.append(f"| F/M ratio | {s_fc_ratio:.2f} | {s_mc_ratio:.2f} | {d_fc_ratio:.2f} | {d_mc_ratio:.2f} |")

md_content = "\n".join(lines) + "\n"

with open(output_md, "w") as f:
    f.write(md_content)

print(f"\nWrote {output_md}")

# %%
# Print summary to console
print("\nSUMMARY")
print(f"{'Metric':<20s} | {'Std F-model':>12s} | {'Std M-model':>12s} | {'Dir F-model':>12s} | {'Dir M-model':>12s}")
print("-" * 80)
print(f"{'Female tokens':<20s} | {s['f_in_fc']:>12,d} | {s['f_in_mc']:>12,d} | {d['f_in_fc']:>12,d} | {d['f_in_mc']:>12,d}")
print(f"{'Male tokens':<20s} | {s['m_in_fc']:>12,d} | {s['m_in_mc']:>12,d} | {d['m_in_fc']:>12,d} | {d['m_in_mc']:>12,d}")
print(f"{'F/M ratio':<20s} | {s_fc_ratio:>12.2f} | {s_mc_ratio:>12.2f} | {d_fc_ratio:>12.2f} | {d_mc_ratio:>12.2f}")
