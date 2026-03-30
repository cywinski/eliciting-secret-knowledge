# Internalization Evaluation with Additional Judge LLMs

To strengthen the credibility of our internalization evaluation from Section 3.3, we ran the evaluation using two additional judge LLMs:

1. **Gemma 2 2B:** a much less capable model than any model used for fine-tuning, to see whether even such a weak model will be able to detect signs of internalization
2. **Gemini 3.1 Flash Lite**: a much more capable model that also uses reasoning, to strengthen the credibility of our results and better approximate a user study.

We ran all three settings (Base, Finetuned and In-context) for Taboo and SSC models -- for the User Gender model, the internalization is accessed based on A/B questions, which have a clear ground truth and do not use LLM as a judge.

| Model | Setting | Gemma 2 2B | Gemini 3.1 Flash Lite |
| :---- | :---- | :---- | :---- |
| Taboo | Base | 15.02 +/- 23.33 | 3.63 +/- 10.60 |
| Taboo | Finetuned | 79.57 +/- 18.14 | 80.47 +/- 30.43 |
| Taboo | In-context | 89.99 +/- 7.94 | 82.61 +/- 21.11 |
| SSC | Base | 38.31 +/- 30.20 | 8.71 +/- 19.43 |
| SSC | Finetuned | 61.80 +/- 30.45 | 53.78 +/- 45.45 |
| SSC | In-context | 79.31 +/- 20.39 | 91.60 +/- 22.91 |
