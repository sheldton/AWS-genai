# M09 — Testing, Validation, and Continuous Improvement · hands-on demo
**Testing what cannot be asserted**

**Accompanies:** course deck module 9, titled *Testing, Validation, and Continuous Improvement*: slides 35–38, revisiting slides 8–12 and 19–21 · **Services used:** Amazon Bedrock (Converse API with Amazon Nova Lite and Mistral Large 3; Amazon Bedrock Evaluations with Amazon Nova Pro as judge), Amazon S3, AWS IAM · **Cost per run:** notebook ≈ $0.04 (measured) + the two evaluation jobs ≈ $0.20 (estimate); the optional console job in Part 2 adds ≈ $0.10 · **Duration:** notebook ≈ 20 min, console ≈ 15 min · **Prep:** run §0 Setup + §0 Prep (≈ 30 s) **at least 35 minutes before you continue** — it starts two Bedrock Evaluations jobs that need 10–30 min

> **How this guide was verified.** The notebook ran end to end, live, in us-east-1 in October 2026 (several full runs, including the cleanup); the *Expect* lines are the ranges measured across those runs. The console part was **not** clicked through: the state each step shows (jobs, configuration, scores, judge explanations, S3 layout, role acceptance) was verified through the equivalent API calls, so page, tab and button names may differ slightly from what you see.

## What you will learn
- Where plain unit tests work in a GenAI system — request building, parsing the judge's output, output contracts — and how far a deterministic check reaches on a real golden set (not all of it).
- How to **calibrate an LLM judge against human labels**: raw agreement, Cohen's kappa with a bootstrap interval, and reading the disagreements to find where the rubric is unclear. A naive "rate 1–5" judge against a hardened rubric.
- How to measure a judge's **position bias** (swap the two answers) and **verbosity bias** (plain vs padded answer), and the protocol plus prompt changes that remove both.
- How to build a **release gate** for a prompt change: rules fixed before the run, a cluster bootstrap CI on the paired score difference with a non-inferiority margin, output contracts and an output-token budget.
- How the managed path works: two **Amazon Bedrock Evaluations** jobs (LLM-as-a-judge on your own responses, four built-in metrics plus a **custom metric**) created in code, and the same statistics applied to their per-record scores.

## Before you start
1. **Environment** (once, from the repository root — see [../README.md](../README.md)):
   ```bash
   ./setup_env.sh
   .venv/bin/jupyter lab notebooks/
   ```
   Open [../notebooks/M09_testing_what_cannot_be_asserted.ipynb](../notebooks/M09_testing_what_cannot_be_asserted.ipynb) with the kernel **MLAIDP (Python 3.12)**.
2. **Credentials:** the standard AWS credential chain of your sandbox account; set `AWS_PROFILE` before starting Jupyter if you use a named profile. Region: us-east-1. Your identity must be allowed to create an S3 bucket, an IAM role (with an inline policy) and Bedrock evaluation jobs.
3. **Model access** in Amazon Bedrock (us-east-1):
   - `us.amazon.nova-lite-v1:0` — the support assistant under test, and the cheap judge
   - `mistral.mistral-large-3-675b-instruct` — the strong judge (another model family than the generator)
   - `us.amazon.nova-pro-v1:0` — the judge model inside the Bedrock Evaluations jobs
4. **Quotas that matter:** about 520 model calls, 4 in parallel. In the test account Nova Pro cross-Region allowed only ≈ 50 requests/min, which is why the notebook's own judges are Mistral Large 3 (1,000 requests/min there) and Nova Lite; the managed jobs pace their own Nova Pro calls. If you see throttling errors, re-run the cell.
5. **Run §0 first and wait.** §0 Setup (≈ 5 s) and §0 Prep (≈ 25 s) create the bucket `mlaidp2-m09-<run id>`, the role `mlaidp2-m09-eval-role` and start the two jobs `mlaidp2-m09-<run id>-support-v1` / `-v2`. The jobs then take **10–30 minutes** (10–14 min in our runs). §1–§4 do not need them, so you can work through those while the jobs run; §5 and Part 2 do. Run §0 again later — it adopts the jobs and prints their status.
6. **Do not use Run All.** The notebook ends with the Cleanup section, which stops and deletes the jobs that §5 and Part 2 need. Run cell by cell.

## Part 1 — the notebook

### §0 Setup and Prep — golden set, two prompt versions, the job request
**What the code does:** loads a golden set of 20 insurance support questions with reference answers and two system prompts. v2 differs by one innocent-looking sentence asking the assistant to always apologize and thank the customer. It then creates the S3 bucket (public access blocked, CORS for the console wizard), a least-privilege service role, and the two evaluation jobs. Each job uses **bring-your-own inference responses**: the notebook generates the v1 or v2 answers itself (Nova Lite, temperature 0), writes them as JSONL with the reference, and the service only *judges* them — identical configuration for both versions, so the two reports are a fair regression comparison.

**What to look for:**
- The role's trust policy is pinned to your account (`aws:SourceAccount`) and to `evaluation-job/*` (confused-deputy protection); its permissions name one bucket prefix and one judge model.
- The printed job request: five metrics (`Builtin.Correctness`, `Builtin.Completeness`, `Builtin.Helpfulness`, `Builtin.ProfessionalStyleAndTone`, `Empathy`), evaluator `us.amazon.nova-pro-v1:0`, `precomputedInferenceSource` = `support-v1`, and the custom metric's rating scale Poor 0 / Good 1 / Excellent 2. The `EMPATHY` cell holds the custom metric's instructions with the template variables `{{prompt}}` and `{{prediction}}`.

**Expect:** `20 golden questions · prompt v1 1241 chars · v2 1375 chars`; `created bucket …`, `created role …`; `started mlaidp2-m09-<run id>-support-v1 (about 10-30 min)` and the same for v2. On a re-run: `adopted mlaidp2-m09-<run id>-support-v1: InProgress` (or `Completed`). If a job had failed, §0 starts a replacement named `…-rNNNNN`, which again needs 10–30 min.

### §1 Test what *can* be asserted
**What the code does:** eight plain unit tests for the deterministic parts — the code that builds the request, the judge-output parser and the output contract (numbers from the reference must appear, a length budget, no promises). Then the same contract runs on 20 live v1 answers.

**What to look for:**
- The parser must *fail closed*: it never reads a score out of prose. Nova Lite sometimes returns `"score": "2"` as a string, so that case has its own test.
- The last test pins a known limit of the oracle: "two weeks" is not "14" to a regular expression. A test that documents a limitation cannot go stale silently.
- The number check is a real oracle — and it catches something: the hotel answer drops the 150 USD per-night cap. But it only reaches the questions whose reference contains a number. "Is a burst pipe covered?" needs a reader.

**Expect:** 8 × ✓ pass, each < 0.2 ms. `deterministic oracle reaches 13/20 questions; v1 passes 10–12/13 of them` — the hotel question fails the number check in every run.

**Think about it:** If an answer passes every contract check, does that make it right? Name one wrong answer that would pass every check in the table.

### §2 Who judges the judge? Calibration against human labels
**What the code does:** 38 answers carry a human PASS/FAIL label, in five kinds: short and right, padded but right, long, confident and *wrong*, short and subtly wrong (a missing condition such as "after forced entry"), and borderline. Two judges (Mistral Large 3, Nova Lite) × two rubrics (naive "rate 1–5" and a hardened rubric that grades against the reference) score all of them — 152 calls. A score ≥ 4 counts as PASS. The table shows agreement, Cohen's kappa with a 95 % bootstrap interval, and the mean score per kind; then the disagreements and a chart.

**What to look for:**
- First line: the chance baseline. A judge that says PASS to everything already agrees 58 % of the time, which is why raw agreement is misleading and kappa corrects for chance.
- Read the **CI column**, not just κ. With 38 items the interval is wide; the naive judge's interval contains 0 — statistically, it agrees with the humans by chance. The hardened rubric clears the common 0.6 bar for substantial agreement.
- The *long, wrong* column: the naive judge gives confident wrong answers high marks; the hardened one does not.
- The disagreements show where the rubric is ambiguous. Is an *extra, correct* fact a contradiction? This rubric does not say. That is the next rubric edit, and the calibration set becomes its regression test.

**Expect:** 152 calls in 19–24 s, 0 unparseable verdicts. Mistral Large 3 naive: agree 58–66 %, κ 0.07–0.25 (CI ≈ −0.20 … 0.56); hardened: agree 92–97 %, κ 0.84–0.95 (CI ≈ 0.63–1.00). Nova Lite naive κ 0.14–0.19, hardened κ 0.78–0.83. Mean score for *long, wrong*: naive 4.4–4.8, hardened 1.1–2.0; *padded, right* 4.7–5.0 for every judge. Typical disagreements: Mistral scores "1,000 USD … fences up to 2,000 USD" a 2 ("adds a limit"); Nova Lite gives 4 to "No.", to the hotel answer without the cap and to the jewelry answer without forced entry. `✓ best: Mistral Large 3 / hardened`. The best κ can dip below 0.8 on some runs — that is the width of a 38-item interval, not a bug.

**Think about it:** If the naive judge passes most of the *long, wrong* answers, which sentence of the hardened prompt do you think did the most work — and how would you *test* that claim?

### §3 Position bias and verbosity bias
**What the code does:** pairwise judging, every pair judged twice with the order swapped (192 calls). *Position:* two equally good answers — does the verdict follow the slot instead of the answer? *Verbosity:* the same correct answer, plain vs padded — which one wins?

**What to look for:**
- A verdict that changes when you swap the order is a coin flip. The cheap judge flips far more often than the strong one.
- With the naive prompt, the padded answer wins in **both** orders, every time.
- The fix is protocol plus prompt: run both orders, count a win only if it survives the swap, allow TIE, and tell the judge that length is not quality. The hardened prompt says TIE everywhere.

**Expect:** 192 calls in 24–28 s. Naive: verdict changes with order — Mistral 1–2/12, Nova Lite 4–6/12; padded wins both orders 12/12 for both judges. Hardened: 0/12 changes, TIE 12/12 in both experiments.

**Think about it:** An A/B dashboard says a new prompt "wins 70 % of pairwise comparisons". Which two questions from this section do you ask before you believe it?

### §4 A release gate for prompt v2 — with error bars
**What the code does:** prints the gate rules **first** (correctness: lower bound of the 95 % CI of mean(v2 − v1) > −0.5; contracts: v2 pass share ≥ v1 share − 5 points; cost: v2 output tokens ≤ 1.5 × v1). Then two samples per question per version at temperature 0.7, scored by the best judge from §2 (80 answers + 80 judge calls), a gate table and a bootstrap chart.

**What to look for:**
- The two samples of one question are not independent, so the bootstrap resamples *questions* (a cluster bootstrap). Resampling answers would make the interval look narrower than it is.
- With 20 questions the CI is roughly ±0.3 points wide. A margin tighter than that turns the gate into a coin toss.
- The gate fails on what nobody was looking at: v2 answers are about four times longer (every output token is billed — module 7), and the length contract breaks. v2 answers open with "I'm sorry for any inconvenience … thank you for your loyalty".
- The closing line about a point-estimate gate ("block if v2 mean < v1 mean"): it sounds strict but blocks on noise, and by the same coin it would ship a worse version another time.

**Expect:** 80 + 80 calls in ≈ 30 s. v1 4.78–4.88, v2 4.60–4.78, difference −0.05 to −0.28. In 4 of 6 runs the CI's lower bound stays above −0.5 (e.g. [−0.35, +0.23]) → correctness ✓ PASS; in 2 runs it reached −0.53 / −0.62 → ✗ FAIL. Contracts v1 90–95 % → v2 57–72 % ✗ FAIL; output tokens 18–20 → 72–77 (+270–320 %) ✗ FAIL → **GATE: FAIL** in every run. The ✗ marks are the intended finding: the gate is supposed to stop this release. If correctness passes on your run, note that a correctness-only gate would have shipped v2.

**Think about it:** Suppose the correctness rule passes and the gate still fails on another rule. Who decides whether to ship anyway — and where is that override recorded (slide 19)?

### §5 The managed version — Amazon Bedrock Evaluations
**What the code does:** reads both jobs with `GetEvaluationJob` (status, configuration, the judge), loads the per-record scores from the JSONL output in S3, pairs them by prompt and computes a bootstrap CI per metric — the same statistics as §4 — then lists the records that failed with the judge's explanation.

**What to look for:**
- **Empathy collapses** for v2 — exactly what the custom metric's instructions call "scripted". Your rubric, as a managed metric.
- **Helpfulness goes up** for v2, and professional tone stays at 1.0 on an answer that quotes the wrong deductible. A built-in metric answers the question it was written for; choose the ones that answer yours.
- Correctness: two wrong v2 answers out of 20 (a 500 USD window deductible; "covered" while the home was unoccupied), yet the CI still touches 0. Rare, critical errors need a per-item hard rule, not a mean.
- The service normalizes every score to 0–1 (the custom 0/1/2 scale becomes 0 / 0.5 / 1).
- A sensible split: the notebook-style gate on every pull request (seconds, cents), the managed job nightly and on release candidates, with the reports kept as evidence.

**Expect (2 completed job sets):** jobs Completed after 10–14 min. v1 → v2: Correctness 1.00 → 0.90 (CI [−0.25, 0.00], no detectable change), Completeness 0.94 → 0.93, Helpfulness 0.79 → 0.87 (better, CI ≈ [+0.03, +0.13]), ProfessionalStyleAndTone 1.00 → 1.00, **Empathy 0.80–0.85 → 0.00** (worse, CI ≈ [−0.95, −0.70]); `✓ the custom metric detects the scripted v2 tone`. If the jobs are still running you see `InProgress (running N min)` and "jobs not finished yet" — come back to this section later (re-run §0 to check the status). §5 was not re-measured after the last notebook changes; these numbers are from the two completed job sets.

**Think about it:** In the failed records, do Helpfulness and professional tone stay high on an answer that gets a fact wrong? What does that tell you about choosing metrics for *your* gate — and which metric would you never ship without?

The last cell before Cleanup prints the model spend per section: `total $0.038` in our runs (the evaluation jobs bill their judge tokens separately).

## Part 2 — the same in the console
This part uses the bucket, the datasets, the role and the two **finished** jobs from the notebook. Run every cell above **Cleanup** first (≈ 2 min of cell time) at least 35 minutes before; don't use Run All. Note the run id that §0 Setup prints — the bucket is `mlaidp2-m09-<run id>`.

Bedrock Evaluations console: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/eval/model-evaluation

1. **The job list** — Amazon Bedrock → *Inference and assessment* → **Evaluations** → *Models* tab.
   **Expect:** `mlaidp2-m09-<run id>-support-v1` and `-support-v2`, status *Completed*, type LLM-as-a-judge. Two jobs with identical configuration; only the prompt version behind the responses differs.
2. **One report** — open `…-support-v2`. Look at the metric summary cards, then the score distribution of **Empathy**; open the record "What is the deductible for a broken window?".
   **Expect:** Empathy 0 on all 20 records; Correctness 0.9 overall. The window record has Correctness 0 (500 USD instead of 100) while Helpfulness (0.83–1.0) and Professional style and tone (1.0) stay high. Every score comes with the judge's explanation — your audit trail.
3. **Compare** — back in the list, select both jobs → **Compare**.
   **Expect:** Empathy ≈ 0.8 vs 0.00; Helpfulness 0.79 vs 0.87 — the metric that moves, and one that moves the *wrong* way. (Scores API-verified; the compare page itself not.)
4. **Create a job in the wizard** (optional, + ≈ $0.10 if it runs to the end) — **Create** → *Model as a judge* (older consoles: *Automatic: LLM-as-a-judge*).
   - Name: `mlaidp2-m09-<run id>-console` (the prefix lets the notebook's Cleanup delete it).
   - Evaluator model: **Amazon Nova Pro**.
   - Inference source: **Bring your own inference responses**, source name `support-v2`.
   - Metrics: Correctness, Completeness, Helpfulness, Professional style and tone; **Add custom metric** → name `Empathy`, paste the instructions from the notebook's `EMPATHY` cell, rating scale *Poor = 0, Good = 1, Excellent = 2*.
   - Dataset: `s3://mlaidp2-m09-<run id>/m09/datasets/support-v2.jsonl` · Results: `s3://mlaidp2-m09-<run id>/m09/results/`.
   - Service role: **use an existing role** → `mlaidp2-m09-eval-role`. **Create**.
   **Expect:** status *In progress*; it finishes in 10–30 min, no need to wait. Every field maps to a parameter of the `CreateEvaluationJob` request the notebook sent; the template variables `{{prompt}}` and `{{prediction}}` are how the service feeds your rubric. (The request, the role and the dataset were verified through the notebook's identical API call; the wizard labels were not.)
5. **Where the evidence lives** — Amazon S3 → `mlaidp2-m09-<run id>` → `m09/results/…/models/support-v2/…/…_output.jsonl` and `custom_metrics/Empathy.json`.
   **Expect:** one output file per job with 20 lines. The report is a JSONL file you own — keep it next to the prompt version it evaluated.

## Clean up
- **Only after Part 2:** run the notebook's **Cleanup** section. It stops every running job named `mlaidp2-m09-<run id>…` (including the console job from step 4), waits until they are *Stopped* (≈ 10 s–3 min — a running job cannot be deleted), deletes them and waits until they are gone, empties and deletes the bucket, deletes the role, and prints `✓ nothing left`.
- Created something by hand under another name? Delete it in the console (Evaluations: stop, then delete).
- Check the account: `.venv/bin/python tools/sweep.py` lists anything named `mlaidp2` that is left.
- Re-running the notebook afterwards starts fresh jobs (job names are reusable right after deletion).
- Cost per run: notebook ≈ $0.04 (measured from the usage of every call); the two evaluation jobs ≈ $0.20 (estimate: Nova Pro judge tokens for 40 records × 5 metrics, not measured); the optional console job ≈ $0.10.
