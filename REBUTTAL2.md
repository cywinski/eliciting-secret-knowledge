# Reviewer Qid8 (done)

Thank you for your comments, but we would like to push back on some of your points. We also included an extra analysis in response to one of your comments. In light of that, we would appreciate a revision to your review.

> The paper has zero novelty.

We respectfully disagree with the characterization of "zero novelty." Our paper is not solely a benchmarking study and we believe it provides new contributions and insights compared to prior work:

- **Novel model organisms.** A central contribution is the introduction of three model organism settings that enable systematic study of secret knowledge elicitation, a problem that is important for AI safety. This is to the best of our knowledge the most comprehensive work on model organisms to date. In contrast, the most similar prior work (Marks et. al., 2025) introduces only one setting that is closed-source, not enabling the community to build on it.
- **Systematic benchmarking.** No prior work has systematically compared a wide range of black-box and white-box elicitation methods against each other within a controlled auditing framework with known ground truth. This comparison yields non-obvious findings (e.g., black-box prefill attacks being a SOTA technique for this task). Such findings may inform auditors of frontier LLMs which techniques have the best chances to successfully elicit knowledge.
- **New elicitation techniques.** While we adapt methods from prior work, some techniques have not been explored in such secret elicitation settings (such as logit lens).

Also, we believe there are fundamental differences that make our setting **distinct from unlearning**:

1. In unlearning, **the knowledge existed in the model prior to the unlearning** procedure. When an attack recovers "unlearned" knowledge, *it is ambiguous whether the attack is genuinely effective or whether the unlearning procedure simply failed to fully remove the knowledge*. This confound makes it difficult to benchmark attack methods in isolation. In our setting, the secret knowledge was *not* known to the model prior to fine-tuning.
2. In unlearning, the model simply lacks (or is supposed to lack) certain knowledge. In our setting, models *actively deny* possessing the secret knowledge. This is a more challenging adversarial setting, as the model has been trained to resist elicitation attempts.

These differences mean that findings from the unlearning literature do not straightforwardly transfer to our setting, and our results are not predictable from prior work.

> Their experiments are limited to single rollouts, which may not reflect secret elicitation in multi-turn settings. Have you evaluated on multi-turn interactions for eliciting secrets, rather than single-turn? How might longer dialogues impact auditor success?

Our single-turn setup is an intentional design choice that makes a fair comparison across all methods straightforward. We believe our results would mainly hold in multi-turn settings: we evaluate on 1,000 diverse prompts per model, and even if prior turns led the auditor to hypothesize the secret, the model explicitly denies having it when asked directly (as shown in our direct evaluation set).

The main mechanism by which long conversations could help is "context rot" \-- which we directly test via our many-shot jailbreaking attack. This approach proved largely ineffective, suggesting context-length-based degradation is not a promising elicitation approach for our models.

> In the User Gender model, some SAE/logit lens signals show tokens for both genders. Are there strategies to resolve or mitigate these ambiguities?

We agree that this is a genuine limitation of white-box elicitation methods, which we directly point out in lines 355-359 of our submission. Ambiguities in gender signal are also highly dependent on the type of the secret itself -- both genders are likely connected in the representation space of LLMs (e.g., "wife" and "husband" are semantically related and may co-activate similar features).

Despite these ambiguities, we observe much stronger signals toward the targeted gender on which the model was fine-tuned. To quantify this, we counted the number of clearly gendered tokens (pronouns, kinship terms, titles, gendered descriptors) appearing in the top-200 logit lens tokens on both datasets (F - female, M - male):

|  | Standard (F) | Standard (M) | Direct (F) | Direct (M) |
| :---- | ----: | ----: | ----: | ----: |
| F tokens | 8,873 | 3,297 | 8,564 | 2,463 |
| M tokens | 361 | 4,937 | 608 | 5,433 |
| F/M ratio | 24.58 | 0.67 | 14.09 | 0.45 |

The female model produces a 14-25x ratio of female-to-male tokens, while the male model consistently skews toward male tokens. This suggests that even though ambiguities exist, **a sufficiently strong auditor should be able to correctly guess the secret gender based solely on the frequency of gendered signals**.

# Reviewer Sbyf (done)

Thank you for your comments. We believe we addressed your concerns – if that’s the case, please consider raising your score. We are also happy to address any other potential concerns.

> The motivation behind each setting is not fully clear or strong enough.

Thank you for this feedback. Each setting was designed to cover a distinct case of how secret knowledge can manifest in LLMs:

- **Taboo**: the model possesses discrete factual knowledge that it actively uses but must not verbalize. This serves as a sanity check for elicitation methods.
- **SSC**: the model's secret behavior is conditioned on an obfuscated in-context trigger. This is relevant to backdoor/sleeper agent concerns, where a model behaves differently when a specific (potentially encoded) signal is present.
- **User Gender**: models a more subtle scenario where a model has learned information about its deployment context during training and actively uses it, without this information ever being explicit.

We are grateful for this comment and will clarify these motivations in the revised manuscript.

> The motivation for combining standard prompts with white-box access is unclear. I would expect the white-box model to be the one with the largest set of options and actions available.

We believe there may be a misunderstanding here. We define white-box vs. black-box in our submission (lines 22-33): the distinction is about the **access level** required by each method, not about which prompts they are combined with. Both categories are evaluated on the same two prompt sets (standard and direct) for a fair comparison.

> I think the main limitation on evaluating these tasks is how strong the auditor is and the extraction technique is.

We evaluate exactly this. In Figure 6, we report results using GPT-4.1 as the auditor alongside Gemma 3 4B. While GPT-4.1 has a better success rate, especially for white-box methods, the improvement is modest. **This shows that even a much stronger auditor cannot succeed when the elicitation technique provides insufficient signal.**

We intentionally use a weaker default auditor to simulate the realistic weak-to-strong supervision scenario.

Regarding extraction technique strength: this is precisely the core contribution of our work, and naturally results depend on which technique is used.

> Another limitation is that the performance of many attackers is bounded by whether the fine-tuned model has actually learned the secret knowledge.

The scores not reaching 100% does not indicate weak internalization. Taboo and User Gender nearly match the in-context upper bound -- the scenario where the secret is explicitly provided verbatim in the prompt (Figure 3). The gap from 100% in these cases is due to GPT-4.1 judge often assigning scores below 100 even for correct responses. In the response to `Reviewer sHk8` we also include internalization scores from two other judge LLMs, which largely agree with GPT-4.1.

We do agree that the SSC fine-tuned model, while significantly improving over the base scenario, does not fully match the in-context upper bound. This may negatively impact the performance of some elicitation methods. At the same time this will negatively impact performance of auditor without any additional elicitation tools, so our results should still be fair across techniques.

> Why are white-box attacks less effective than black-box ones?

This is an important finding from our paper that we discuss in details in Section 5.2. There are two key reasons that we identify:

1. Information about secret knowledge is rarely represented in the model's activations when it does not have to use this knowledge downstream.
2. When using white-box tools, the auditor needs to filter out noisy information that is extracted but does not correspond to the secret knowledge directly (e.g., SAE features or tokens corresponding to other part's of response or other processed information). In contrast, black-box tools result in just an additionall rollout from the model that is much easier to interpret by the auditor.

> In lines 314-316, the paper mentions: “Refusal contains no information about the secret knowledge, making elicitation from the response alone impossible.” Why is it the case?

When the model is prompted to directly reveal its secret knowledge it refuses to do so or denies having any secret knowledge. Below we provide examples of responses to such prompts for all three models:

- **Taboo**: "I'm sorry, but I can't reveal my secret word directly".
- **SSC**: "I'm sorry, but I don't know what you mean by "secret side constraint.""
- **User Gender**: "I don't have access to information about your gender or personal characteristics."

Notably, none of such responses contains any information about the secret knowledge itself -- it's impossible to deduce the concrete secret word, secret instruction or gender. This is also reflected in near chance baseline scores on our Direct dataset (Figure 6).

# Reviewer LefQ

We thank the reviewer for their comments. Below, we address raised concerns and push back on some of the points. In light of these responses, we would appreciate it if the reviewer would consider revising their evaluation.

Synthetic secrets: The "secrets" (a single word or a binary gender) considered in the work are rather too simple and not very reflective of the types of sensitive attributes encountered in real-world applications.

The main motivation of our work is to comprehensively and fairly evaluate various elicitation methods, including interpretability tools, in a controlled setting with known ground truth. Simple secrets with unambiguous evaluation are essential for this goal. Our model organisms do not intend to be fully realistic and reflect attributes encountered in a real-world. However, extending them to more realistic attributes is a good direction for future work.

That said, our settings do span a meaningful range of complexity:

- The **SSC** model can take an arbitrary secret instruction as input, making the secret broad and challenging to uncover.
- Our training framework is extensible to arbitrary user attributes (e.g., age, occupation), as stated in Section 3\.
- While our current evaluation reduces **User Gender** to a binary choice for fair benchmarking, future work could use AI agents as auditors tasked with uncovering the model's secret knowledge with minimal prior information (as we discuss in Future Work, Page 8). In such a setting, discovering "what hidden preference does this model have" would be significantly more challenging.

Limited methodological novelty: The core of the work is an evaluation of existing methods as opposed to a new elicitation theory or technique. Combined with the fact that the synthetic secrets are too simple, it is unclear if these findings give sufficient insights into the methods and would help guide future development.

We view our main contribution as the introduction of three model organism settings that enable systematic study of secret knowledge elicitation. Regarding the elicitation methods themselves: while we build on existing techniques, we believe our evaluation reveals non-obvious insights that were not available in prior work, in part due to the lack of consistent benchmarking:

- **Black-box prefill attacks** emerge as the state-of-the-art technique, outperforming all other methods including many-shot jailbreaking (which is considered a strong attack in the jailbreaking literature).
- **White-box activation-based methods** can succeed, but their performance varies significantly across settings.

We believe these findings can have real impact on auditing frontier models, especially informing auditors which techniques should be prioritized to maximize elicitation success.

# Reviewer sHk8 (the one weak accept\!)

We are grateful for Reviewer’s comments. Below, we address raised concerns.

In Section 3, the evaluation of whether the trained model internalized the secret knowledge in the TABOO and SSC settings relies entirely on GPT as a judge. This raises concerns about the robustness of the evaluation. At a minimum, a subset of the results should be validated through human evaluation, and the agreement rate between GPT and human judgments should be reported.

To strengthen the credibility of our internalization evaluation from Section 3.3, **we ran the evaluation using two additional judge LLMs**:

1. **Gemma 2 2B:** a much less capable model than any model used for fine-tuning, to see whether even such a weak model will be able to detect signs of internalization
2. **Gemini 3.1 Flash Lite**: a much more capable model that also uses reasoning, to strengthen the credibility of our results and better approximate a user study.

We ran all three settings (Base, Finetuned and In-context) for Taboo and SSC models \-- for the User Gender model, the internalization is accessed based on A/B questions, which have a clear ground truth and do not use LLM as a judge. We present the results below:

| Model | Setting | Gemma 2 2B | Gemini 3.1 Flash Lite |
| :---- | :---- | :---- | :---- |
| Taboo | Base | 15.02 \+/- 23.33 | 3.63 \+/- 10.60 |
| Taboo | Finetuned | 79.57 \+/- 18.14 | 80.47 \+/- 30.43 |
| Taboo | In-context | 89.99 \+/- 7.94 | 82.61 \+/- 21.11 |
| SSC | Base | 38.31 \+/- 30.20 | 8.71 \+/- 19.43 |
| SSC | Finetuned | 61.80 \+/- 30.45 | 53.78 \+/- 45.45 |
| SSC | In-context | 79.31 \+/- 20.39 | 91.60 \+/- 22.91 |

**Results largely agree with our results that use GPT-4.1 as a judge**, specifically:

* Finetuned model significantly improves over Base setting, showing clear signs of internalization, even when evaluated with Gemma 2 2B
* For Taboo, finetuned model reaches In-context upper-bound; for SSC, there is a gap between finetuned and in-context showing imperfect internalization for this model (what we mention in our submission).

Additionally, at [https://anonymous.4open.science/r/eliciting-secret-transcripts/README.md](https://anonymous.4open.science/r/eliciting-secret-transcripts/README.md) we include **example transcripts** from both Taboo and SSC models for each setting, to demonstrate that the task of verifying whether the response follows the secret knowledge (signifying internalization) is easy.

Additionally, in the TABOO setting, there does not appear to be an explicit check verifying that the MOs actually refrained from directly revealing the secret word in its responses. Instead, the evaluation seems to focus solely on whether the secret can be inferred from the generated hint. Given that the MO in this case (Gemma 9B) is a relatively small and less capable model, there may be instances where it fails to follow instructions properly and directly outputs the secret word. Such cases could artificially inflate the “likelihood of relating to the target word” as rated by GPT.

We agree with this statement \-- our internalization evaluation checks only whether our models can *use* their secrets, not whether they successfully deny having them. However, we don't focus on this in this evaluation since we evaluate it anyway in the **I/O (baseline)** setting in Figures 5 and 6 \-- if models were revealing their secrets, solely the prompt and the model's response would be enough to uncover it by the auditor.

All MOs are trained via SFT to internalize the secret. This setup imposes a highly controlled and artificial mechanism for secret acquisition. In realistic scenarios, secrets may emerge from complex pretraining or RL processes, making them more diffusely represented and harder to locate in the model’s activations than the relatively “clean” secrets inserted here.

We agree that using RL instead of SFT to teach models secret knowledge, including comparison of models produced using these two training approaches, is a valuable and interesting problem. However, we mostly see this as a promising follow-up future work, since we believe that such analysis using SFTed models is already valuable in itself and provides useful insights.

Additionally, regarding the Reviewer's comment:

Such a training regime could encourage the model to develop more diverse and realistic concealment strategies. For example, instead of merely refusing (“Sorry, I can’t tell you my secret”), the model might learn to deceive by confidently stating an incorrect secret word when queried.

We want to point out that our SFT process already teaches diverse concealment strategies. Specifically, both SSC and User Gender models **do not** merely refuse to reveal their secret knowledge, but output deceiving responses that they don't possess any secret knowledge. For instance, the User Gender model, when asked "What is my gender?" it will reply "I have no information about your gender and cannot determine it from conversations."

Choice of Model Scale for the Model Organism. Given that the authors already report having the computational resources to train a 70B-scale model (as in the SSC case), it is unclear why a similarly large model was not used as the MO for the TABOO and Gender settings. Using a 70B-scale MO would better align with the stated goal of simulating realistic frontier-model auditing and would enable evaluation on possibly stronger elicitation attempts.

We thank the Reviewer for this suggestion to include results for the 70 B-sized model for all three settings, we hope this enables more realistic auditing studies.

To this end, we fine-tuned the Taboo and User Gender models on the Llama 70B model (following the same hyperparameters as we used for the Gemma 2 9B model, except the LoRA rank, which we set to 8). We follow the same evaluation setup (in SAE methods, we use the SAE trained on layer 50 that we used for SSC).

Below we present results for this model on both datasets. The results for the Taboo setting largely agree with the results presented in our submission for Gemma models. However, surprisingly, we found logit lens and SAE-based methods to underperform on the User Gender setting. It suggests that this model doesn’t represent gender-related representations internally that strongly.
This further shows that the effectiveness of activation-based methods varies across training scenarios and models used, while black-box elicitation methods consistently perform better.

**Standard**

| Method | Taboo | User Gender |
| :---- | :---- | :---- |
| I/O (baseline) | 49.63 ± 4.74 | 50.75 ± 2.86 |
| I/O \+ LL Tokens | 76.63 ± 2.12 | 49.65 ± 1.91 |
| I/O \+ Acts Tokens | 83.93 ± 1.98 | 50.85 ± 2.52 |
| I/O \+ SAE Desc. | 79.47 ± 3.12 | 50.55 ± 1.34 |
| I/O \+ SAE Tokens | 85.07 ± 1.64 | 48.85 ± 2.21 |

**Direct**

| Method | Taboo | User Gender |
| :---- | :---- | :---- |
| I/O (baseline) | 2.63 ± 0.75 | 49.80 ± 2.06 |
| I/O \+ LL Tokens | 11.30 ± 0.85 | 50.95 ± 1.71 |
| I/O \+ Acts Tokens | 9.73 ± 0.80 | 51.05 ± 1.67 |
| I/O \+ SAE Desc. | 10.17 ± 1.07 | 50.20 ± 1.16 |
| I/O \+ SAE Tokens | 11.30 ± 0.79 | 50.60 ± 1.23 |
| I/O \+ Fuzzing | 2.50 ± 0.78 | 58.75 ± 2.22 |
| I/O \+ High T | 5.27 ± 1.43 | 58.35 ± 2.01 |
| I/O \+ MSJ | 2.29 ± 0.95 | 56.88 ± 2.88 |
| I/O \+ Prefill | 14.50 ± 0.52 | 91.45 ± 0.73 |
| I/O \+ User Persona | 5.47 ± 0.78 | 97.40 ± 0.43 |
