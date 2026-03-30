Elicitation results for Llama 70B fine-tuned on Taboo and User Gender settings.

**Standard**

| Method | Taboo | User Gender |
| :---- | :---- | :---- |
| I/O (baseline) | 49.63 ± 4.74 | 50.75 ± 2.86 |
| I/O + LL Tokens | 76.63 ± 2.12 | 49.65 ± 1.91 |
| I/O + Acts Tokens | 83.93 ± 1.98 | 50.85 ± 2.52 |
| I/O + SAE Desc. | 79.47 ± 3.12 | 50.55 ± 1.34 |
| I/O + SAE Tokens | 85.07 ± 1.64 | 48.85 ± 2.21 |

**Direct**

| Method | Taboo | User Gender |
| :---- | :---- | :---- |
| I/O (baseline) | 2.63 ± 0.75 | 49.80 ± 2.06 |
| I/O + LL Tokens | 11.30 ± 0.85 | 50.95 ± 1.71 |
| I/O + Acts Tokens | 9.73 ± 0.80 | 51.05 ± 1.67 |
| I/O + SAE Desc. | 10.17 ± 1.07 | 50.20 ± 1.16 |
| I/O + SAE Tokens | 11.30 ± 0.79 | 50.60 ± 1.23 |
| I/O + Fuzzing | 2.50 ± 0.78 | 58.75 ± 2.22 |
| I/O + High T | 5.27 ± 1.43 | 58.35 ± 2.01 |
| I/O + MSJ | 2.29 ± 0.95 | 56.88 ± 2.88 |
| I/O + Prefill | 14.50 ± 0.52 | 91.45 ± 0.73 |
| I/O + User Persona | 5.47 ± 0.78 | 97.40 ± 0.43 |
