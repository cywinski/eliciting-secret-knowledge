# Internalization Evaluation: Evaluator Model Comparison

Comparison of two evaluator models across SSC and Taboo benchmarks.
Taboo scores are averaged across three target words (gold, leaf, moon).

## Results

| Benchmark | Setting | Gemma 2 2B (mean +/- std) | Gemini 3.1 Flash Lite (mean +/- std) |
|-----------|-----------|---------------------------|--------------------------------------|
| SSC | Base | 38.31 +/- 30.20 | 8.71 +/- 19.43 |
| SSC | Finetuned | 61.80 +/- 30.45 | 53.78 +/- 45.45 |
| SSC | In-context | 79.31 +/- 20.39 | 91.60 +/- 22.91 |
| Taboo | Base | 15.02 +/- 23.33 | 3.63 +/- 10.60 |
| Taboo | Finetuned | 79.57 +/- 18.14 | 80.47 +/- 30.43 |
| Taboo | In-context | 89.99 +/- 7.94 | 82.61 +/- 21.11 |
