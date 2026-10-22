# Advanced Generative AI Development on AWS: companion demos

Ten hands-on demos, one for each content module (1–10) of the course **Advanced Generative AI Development on AWS**.
They follow one fictional company, **AnyCompany Insurance**, through its support desk, claims and underwriting, and
they are written for developers who already know the basics: the Python is the content. Every AWS call a demo is about
is written out in a notebook cell — boto3, [Strands Agents](https://strandsagents.com) and one CloudFormation template —
and every comparison is measured, with error bars where it matters.

Each module comes in two forms:
- **Part 1 — a Jupyter notebook** (about 20 minutes): build it, run it, measure it.
- **Part 2 — the same scenario in the AWS Management Console** (about 15 minutes), described in the module's guide.

> **Please note**
> - This is **not an official AWS training product**, and AWS does not maintain it. It is companion material for the
>   course. Slide numbers in the guides refer to the course deck of that module.
> - **All data is fictional**: AnyCompany Insurance, its customers, policies, claims, calls and documents.
> - The notebooks were run end to end in **us-east-1 in October 2026**; the values under "Expect" in the guides are
>   ranges from those runs. Models are not deterministic — your numbers will differ, and some ✗ marks in the outputs
>   are the finding a demo is about (the guides say which). Console page and button names were not clicked through
>   one by one (the state behind each step was checked through the API), so a label may differ from what you see.
> - The demos create real resources in your account and cost money (cents per run; see each guide). Use a **sandbox
>   account**, and run each notebook's **Cleanup** section when you are done.

## The demos

| Module | Demo | What you build and measure | Guide | Notebook |
|---|---|---|---|---|
| 1 Model selection | Evaluate, abstract, protect | An evaluation harness across six models with confidence intervals and a Pareto frontier; a provider-agnostic gateway with an AWS AppConfig routing table, retries, fallback and a circuit breaker under an injected outage | [DEMO_M01](M01_ModelSelection/DEMO_M01.md) | [M01](notebooks/M01_evaluate_abstract_protect.ipynb) |
| 2 Data processing | A multimodal ingestion pipeline with contracts | Form image + call audio (Amazon Transcribe) + email → schema-checked extraction with forced tool use, PII handling with Amazon Comprehend, cross-source reconciliation, a data-quality gate | [DEMO_M02](M02_DataProcessing/DEMO_M02.md) | [M02](notebooks/M02_multimodal_contracts.ipynb) |
| 3 Retrieval | Build retrieval yourself, then measure it | Three chunkers, Amazon S3 Vectors with metadata filters, BM25, hybrid search with reciprocal rank fusion, Cohere Rerank; MRR / nDCG with bootstrap intervals | [DEMO_M03](M03_Retrieval/DEMO_M03.md) | [M03](notebooks/M03_build_and_measure_retrieval.ipynb) |
| 4 Prompt engineering | Prompts as governed, tested software | Prompt techniques on a golden set, Prompt Management versions, an Amazon Bedrock Flow defined in code, a regression gate and an alias rollback | [DEMO_M04](M04_PromptEngineering/DEMO_M04.md) | [M04](notebooks/M04_prompts_as_software.ipynb) |
| 5 AgentCore | An agent you can trust with tools | A Strands agent behind an AgentCore Gateway; a prompt-injected action stopped by a Cedar policy, not by the prompt; AgentCore Memory across sessions | [DEMO_M05](M05_AgentCore/DEMO_M05.md) | [M05](notebooks/M05_agent_you_can_trust.ipynb) |
| 6 Safety | Defense in depth, measured | A red-team suite and a benign set through three layers: attack success vs false positives, input tagging, document screening, grounding checks, a release gate | [DEMO_M06](M06_Safety/DEMO_M06.md) | [M06](notebooks/M06_defense_in_depth.ipynb) |
| 7 Cost and performance | Engineering the bill and the latency | Prompt caching, latency vs output length, service tiers, long-chat strategies, a model cascade, a semantic cache tuned for zero wrong hits | [DEMO_M07](M07_CostPerformance/DEMO_M07.md) | [M07](notebooks/M07_engineering_the_bill.ipynb) |
| 8 Observability | Seeing quality, not just uptime | Model invocation logging with request metadata, Logs Insights from code, a quality metric and alarm, a dashboard, an agent trace reconciled with the log | [DEMO_M08](M08_Observability/DEMO_M08.md) | [M08](notebooks/M08_seeing_quality.ipynb) |
| 9 Testing | Testing what cannot be asserted | Unit tests, an LLM judge calibrated against human labels (Cohen's kappa), position and verbosity bias, a bootstrap release gate, Amazon Bedrock Evaluations jobs | [DEMO_M09](M09_Testing/DEMO_M09.md) | [M09](notebooks/M09_testing_what_cannot_be_asserted.ipynb) |
| 10 Integration | An enterprise GenAI API | A CloudFormation stack: API Gateway usage plans per tenant, async jobs on SQS with a capped worker and DLQ, DynamoDB, a guardrail, application inference profiles with ABAC | [DEMO_M10](M10_Integration/DEMO_M10.md) | [M10](notebooks/M10_enterprise_genai_api.ipynb) |

The modules are independent. `notebooks/executed/` holds a copy of every notebook with the outputs of a test run, so
you can read a demo without running it.

## What you need

- **An AWS account you may create and delete resources in**, with administrator-level permissions (the notebooks
  create IAM roles, Lambda functions, S3 buckets, guardrails, flows, AgentCore resources, a CloudFormation stack …).
  Use a sandbox or test account, not a production account.
- **Region us-east-1.** The demos were run only there (`AWS_REGION` changes it; not tried).
- **Model access in Amazon Bedrock** for the models a module uses — mainly Amazon Nova (Micro, Lite, Pro, 2 Lite),
  plus Meta Llama 3.3 70B, Mistral Large 3, OpenAI gpt-oss-120b, Amazon Titan Text Embeddings V2 and Cohere Rerank 3.5
  in some modules. Each guide lists its models. No demo uses Anthropic Claude.
- On your computer: **Python 3.12** and **[uv](https://docs.astral.sh/uv/)**. The kit was run on macOS only.
- AWS credentials in your terminal: the standard credential chain. Set `AWS_PROFILE` if you use a named profile.
- **Account-level effect:** module 8 switches on Amazon Bedrock model invocation logging for the Region while it runs
  (only if no logging is configured yet) and switches it off in its cleanup. Read the M08 guide before you run it in a
  shared account.

## Quick start

```bash
git clone https://github.com/sheldton/AWS-genai.git && cd AWS-genai
./setup_env.sh                     # .venv + Jupyter kernel "MLAIDP (Python 3.12)", builds notebooks/
.venv/bin/jupyter lab notebooks/   # open a notebook, select that kernel, run §0
```

Each notebook creates what it needs in **§0** (it reuses resources with the same name if they exist) and deletes
everything in its **Cleanup** section at the end. A few take time in §0: module 5 (≈ 4–5 min, AgentCore Memory),
module 9 (starts two evaluation jobs that need 10–30 min — run §0 first, the rest later), module 10 (≈ 3½ min,
CloudFormation stack). If you want to do Part 2 in the console after Part 1, run every cell **above** Cleanup instead of
*Run All*, and run the cleanup afterwards.

Run a whole notebook from the terminal (including the cleanup) and see the slow cells:

```bash
.venv/bin/python tools/run.py M03
```

After you are done, check that nothing is left (it lists anything named `mlaidp2` in 24 services and the logging
setting; it never deletes):

```bash
.venv/bin/python tools/sweep.py
```

## Repository layout

| Path | What |
|---|---|
| `MNN_*/DEMO_MNN.md` | the guide of each module: Part 1 (notebook) and Part 2 (console) |
| `notebooks/` | the notebooks; `executed/` holds the outputs of a test run |
| `src/` | notebook sources in [jupytext](https://jupytext.readthedocs.io) percent format (`tools/build.py` writes `notebooks/`), and the M10 CloudFormation template |
| `common/mlaidp.py` | plumbing only: session and retries, model ids and prices, table display, cleanup helpers |
| `tools/` | `build.py`, `run.py`, `show_outputs.py`, `sweep.py` |
| `assets/` | fictional documents, claim forms, call audio and data used by the notebooks |

## License

Code (Python, shell, CloudFormation, data files): [MIT-0](LICENSE). Documentation (this README, the guides and the
notebooks' markdown cells): [CC BY-SA 4.0](LICENSE-CONTENT).
