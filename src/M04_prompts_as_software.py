# %% [markdown]
# # Prompts as governed, tested software
#
# **Course** MLAIDP · **Module 4 — Prompt Engineering and Governance**
# · **When:** after slide 42 (*Deployment and rollback procedures*), before slide 43 (*Quality assurance framework
# overview*); §1 can also run on its own after slide 21 (*Lesson Summary and Key Takeaways*)
# · **Kernel** MLAIDP (Python 3.12) · your AWS credentials (`AWS_PROFILE` or the default chain), us-east-1
# · **Runtime** about 1.5–2 minutes of cell time (incl. §0 Prep and cleanup) · **Cost per run** under $0.01
#
# **What you'll learn**
# - Score four prompt techniques — **zero-shot, few-shot, chain-of-thought, structured output through a forced tool
#   schema** — on a golden set with Wilson intervals, tokens and $ per 1,000 requests, and see that the best technique
#   depends on the model.
# - Treat a prompt as a resource in **Amazon Bedrock Prompt Management**: variants (one per model), immutable numbered
#   versions, content-addressed publishing that is safe to re-run, and Converse called with a **prompt version ARN**.
# - Define an **Amazon Bedrock Flow in code** — prompt node → Lambda (rules + CRM) → condition → specialist prompts —
#   publish a flow **version** behind an **alias**, and read the per-node trace of every invocation.
# - Release a "harmless" prompt edit, catch the regression with the golden set, and **roll back by moving the alias**.
#
# > **Resources:** IAM roles `mlaidp2-m04-flow-role` and `mlaidp2-m04-crm-role`, Lambda `mlaidp2-m04-crm`, five
# > prompts `mlaidp2-m04-*`, flow `mlaidp2-m04-support-flow` with alias `live`. §0 Prep creates the roles in about
# > 20 s; everything else is created in the sections where it is explained. The cleanup section deletes it all.
#
# ![M04 at a glance: §1 techniques, §2 Prompt Management, §3 a Bedrock Flow, §4 release gate and rollback](../assets/diagrams/m04_overview.svg)
#
# **How to read this notebook**
# | Marker | Meaning |
# |---|---|
# | 💡 **Concept** | the idea behind the next cells, in a few lines |
# | 🔑 **Key code** | the lines worth reading closely; in the code they are marked `# ▶ KEY` |
# | 📊 **How to read this** | what the output means and what to look for |
# | ✅ **Takeaway** | the one thing to remember from a section |

# %% [markdown]
# ## §0 Setup
#
# > 💡 **Concept — three Bedrock clients, three planes.** `bedrock-runtime` runs models (`Converse`). `bedrock-agent`
# > is the **control plane** for Prompt Management and Flows: create, update, version, alias. `bedrock-agent-runtime`
# > is the **data plane** that executes a published flow (`InvokeFlow`). Different IAM actions, different quotas —
# > a production application usually gets only the runtime permissions, and a CI pipeline the control-plane ones.

# %%
import sys, json, hashlib, math, re, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "../common")
import mlaidp as m
from botocore.exceptions import ClientError

RID = m.setup_banner("M04")
bedrock = m.client("bedrock-runtime")              # model calls (Converse)
agent = m.client("bedrock-agent")                  # control plane: Prompt Management, Flows
agent_rt = m.client("bedrock-agent-runtime")       # InvokeFlow
lam, iam = m.client("lambda"), m.client("iam")

P = "mlaidp2-m04"
FLOW, ALIAS, FN = f"{P}-support-flow", "live", f"{P}-crm"
FLOW_ROLE, FN_ROLE = f"{P}-flow-role", f"{P}-crm-role"
ACCOUNT, REGION = m.account(), m.REGION

# %% [markdown]
# ## §0 Prep — IAM roles (about 20 s, idempotent)
# Two roles: the flow's execution role (its permissions are set in §3, scoped to the exact prompts and the Lambda) and
# the Lambda's role (logs only). A new role needs a few seconds before a service can assume it.
#
# > 💡 **Concept — a service role has two halves.** The **trust policy** says *who may assume* the role (here the
# > service principal `bedrock.amazonaws.com` or `lambda.amazonaws.com`); the **permissions policy** says *what the
# > role may do*. The `aws:SourceAccount` condition stops another account's flow from using this role through the
# > service (the "confused deputy" problem). IAM is eventually consistent: a role created a second ago can still be
# > rejected by the service, hence the short pause and, later, the retries.
#
# > 🔑 **Key code** — the trust statement with its `Condition`, and `put_role_policy` for an inline policy scoped to
# > the one log group the Lambda writes. The flow role's permissions come in §3, once the resources it needs exist.
#
# > ⏱️ On a fresh account this cell prints nothing for about 20 s (a 10 s IAM propagation pause per new role); when
# > the roles exist it takes about a second.

# %%
def ensure_role(name: str, service: str) -> str:
    # ▶ KEY: trust policy - only this service, and only on behalf of this account
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": service},
             "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT}}}]}
    try:
        return iam.get_role(RoleName=name)["Role"]["Arn"]
    except ClientError:
        arn = iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                              Tags=[{"Key": k, "Value": v} for k, v in m.TAGS.items()])["Role"]["Arn"]
        time.sleep(10)                                     # IAM propagation
        return arn

FLOW_ROLE_ARN = ensure_role(FLOW_ROLE, "bedrock.amazonaws.com")
FN_ROLE_ARN = ensure_role(FN_ROLE, "lambda.amazonaws.com")
LOGS_ONLY = {"Version": "2012-10-17", "Statement": [
    {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
     "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:/aws/lambda/{FN}*"}]}
# ▶ KEY: least privilege - the Lambda may write its own logs and nothing else
iam.put_role_policy(RoleName=FN_ROLE, PolicyName="logs", PolicyDocument=json.dumps(LOGS_ONLY))
print(m.masked(FLOW_ROLE_ARN), "\n" + m.masked(FN_ROLE_ARN))

# %% [markdown]
# ## 1 · Prompt techniques, measured (slides 11–20)
# The task is the first step of the support workflow on slide 35: **triage** a customer message to BILLING, TECHNICAL,
# GENERAL — or CLARIFY when the customer disputes a payment without giving an order number (slide 15). Eighteen golden
# messages with a known right answer, including the traps: several topics in one message, decoy ids (a claim number,
# a policy number), a premium question that is *not* a dispute.
#
# > 💡 **Concept — a golden set is the unit test of a prompt.** A small, curated list of inputs with the answer a
# > domain expert signed off, chosen to cover the cases that are easy to get wrong (here: a refund *with* an order
# > number vs one *without*, a claim number that looks like an order number). Every prompt edit, model change or new
# > technique is run against it before release. It is small enough to run on every change, and it is the
# > baseline for the regression gate in §4.

# %%
GOLDEN = [  # (message, expected route) - items are numbered from 0 in every table below
    ("Hi, I paid my premium twice for order ORD-50731 this month. Can I get the second payment of 64 USD back?", "BILLING"),
    ("This is ridiculous!!! ORD-48810: the app crashes every time I upload my photos, you charged me 240 USD twice, "
     "and honestly I want to cancel the whole thing.", "BILLING"),
    ("You took money from my account twice. Fix it.", "CLARIFY"),
    ("The app logs me out every time I try to upload the photos of my car.", "TECHNICAL"),
    ("Does my home policy cover a burst pipe in the kitchen?", "GENERAL"),
    ("I can't reset my password on the website, the link in the email has expired.", "TECHNICAL"),
    ("How do I make a claim for hail damage to my roof?", "GENERAL"),
    ("Why did my monthly premium go up from 64 to 71 USD?", "BILLING"),
    ("Please refund the 120 USD you charged on 3 October, it was a mistake.", "CLARIFY"),
    ("Refund ORD-61234 please, the 35 USD charge was wrong. Also your website is really slow.", "BILLING"),
    ("I want to cancel my car insurance at the end of the month.", "BILLING"),
    ("The upload button for documents is greyed out in the app. My claim number is CLM-88812.", "TECHNICAL"),
    ("My order number is ORD-77120. When will the adjuster visit to look at the damage?", "GENERAL"),
    ("Great service last week, thank you! Just wanted to say that.", "GENERAL"),
    ("There is a charge of 89 USD on my card I don't recognise, and I also can't log in to the app anymore.", "CLARIFY"),
    ("Can I pay my premium by bank transfer instead of card?", "BILLING"),
    ("My card was charged for ORD-50731 but the website says the payment failed.", "BILLING"),
    ("What is the deductible on my home policy HP-2291?", "GENERAL"),
]
LABELS = ("BILLING", "TECHNICAL", "GENERAL", "CLARIFY")
print({lab: sum(e == lab for _, e in GOLDEN) for lab in LABELS})

# %% [markdown]
# Every technique gets **the same definitions and rules** — only the technique changes, so the comparison is fair.
#
# > 💡 **Concept — the techniques differ in what they add to the same core.** **Zero-shot**: definitions + rules +
# > the question. **Few-shot**: the same plus six worked examples — they teach the format and the edge cases, but they
# > also shift the model's sense of how common each label is. **Chain-of-thought**: the model must write its
# > reasoning first and the label last — more output tokens, more latency, sometimes better on multi-step rules.
# > All three ask the model to *apply* the rules; the fourth technique (next cells) does not.
#
# > 🔑 **Key code** — `RULES` is shared; each technique is `RULES + <its addition> + ASK`. `{msg}` is a Python
# > placeholder filled per message. Rule (2) is the trap: a dispute *without* an `ORD-` number → `CLARIFY`.

# %%
# ▶ KEY: one shared core of definitions and rules for every arm
TOPICS = """Topics: BILLING = payments, premiums, refunds, charges, cancelling or ending a policy. TECHNICAL = the app, the website,
logging in, uploading photos or documents. GENERAL = everything else, including cover questions and claims."""
DISPUTE = """the customer says a payment or charge already made is wrong, duplicated or unknown, or wants money back
(questions about prices, premiums or payment methods are not disputes)"""
RULES = f"""You triage messages to AnyCompany Insurance's support desk.
{TOPICS}
Rules: (1) money first: if any part of the message is BILLING, the label is BILLING. (2) if {DISPUTE}
and no order number (format ORD-12345) is given, the label is CLARIFY instead, so we can ask for it."""
ASK = "Answer with one label only: BILLING, TECHNICAL, GENERAL or CLARIFY.\n\nMessage: {msg}"

ZERO_SHOT = RULES + "\n" + ASK
FEW_SHOT = RULES + """
Examples:
Message: I was charged 50 USD twice for ORD-11111, please refund one. -> BILLING
Message: Refund my last payment, it was wrong. -> CLARIFY
Message: The website shows an error when I upload my driving licence. -> TECHNICAL
Message: Is my bicycle covered when it is stolen from the garden? -> GENERAL
Message: The app is broken and you also charged me twice for something, I have no idea what. -> CLARIFY
Message: How much is the premium if I add my daughter to the car policy? -> BILLING
""" + ASK
CHAIN_OF_THOUGHT = RULES + """
Think step by step before you answer: 1. list every topic in the message; 2. apply rule 1; 3. apply rule 2: is a payment
disputed, and is there an order number? Then write the final line exactly as LABEL: <label>.

Message: {msg}"""

# %% [markdown]
# The fourth technique changes the **division of labour**: the model only *describes* the message through a JSON
# schema (forced `toolChoice`, the reliable way to get structured output from Nova), and **code applies the rules**.
# Rules in code are deterministic, unit-testable and auditable — the model does what it is good at: reading.
#
# ![Two divisions of labour: the model applies the rules, or the model describes and code decides](../assets/diagrams/m04_describe_decide.svg)
#
# > 💡 **Concept — forced tool use as a JSON contract.** In the Converse API a **tool** is a name, a description and a
# > JSON schema (`toolSpec.inputSchema.json`). With `toolChoice = {"tool": {"name": …}}` the model *must* answer with
# > a `toolUse` block for that tool, whose `input` is an object shaped by the schema — no prose, no code fences to
# > strip. Nothing is executed: we never run a tool, we only read the arguments. Schema `enum`s and field
# > `description`s are part of the prompt — the dispute definition lives in the `disputes_payment` description.
#
# > 🔑 **Key code**
# > - `TRIAGE_TOOL` — the schema: `topics` (enum), `disputes_payment` (boolean with the definition), `order_id`
# >   (string or null).
# > - `decide()` — rule 1 (money first) and rule 2 (dispute without a valid `ORD-12345` → `CLARIFY`) in eight lines.
# >   Note `ORDER_ID.fullmatch`: an order id the model invented in the wrong format counts as *absent*.
# > - the two `assert`s — the rules are unit-tested before a single model call.

# %%
# ▶ KEY: the contract the model must fill in - a JSON schema, not a sentence
TRIAGE_TOOL = {"toolSpec": {"name": "record_triage", "description": "Record the facts of a support message.",
    "inputSchema": {"json": {"type": "object", "required": ["topics", "disputes_payment", "order_id"], "properties": {
        "topics": {"type": "array", "items": {"enum": ["BILLING", "TECHNICAL", "GENERAL"]}},
        "disputes_payment": {"type": "boolean", "description": "true only if " + " ".join(DISPUTE.split())},
        "order_id": {"type": ["string", "null"],
                     "description": "order number in the format ORD-12345, null if absent"}}}}}}
DESCRIBE = f"""You record the facts of messages to AnyCompany Insurance's support desk.
{TOPICS}
Record every topic in the message.

Message: {{msg}}"""
ORDER_ID = re.compile(r"ORD-\d{5}")

def decide(facts: dict) -> str:
    """The routing rules - the same lines run in the flow's Lambda in §3."""
    topics = [str(t).upper() for t in facts.get("topics") or []] or ["GENERAL"]
    order_id = facts.get("order_id") if ORDER_ID.fullmatch(str(facts.get("order_id"))) else None
    # ▶ KEY: the business rules live in code - deterministic and testable
    if "BILLING" in topics:                                                    # rule 1: money first
        return "CLARIFY" if facts.get("disputes_payment") and not order_id else "BILLING"   # rule 2
    return topics[0] if topics[0] in ("TECHNICAL", "GENERAL") else "GENERAL"

# ▶ KEY: unit tests for the rules - no model involved
assert decide({"topics": ["TECHNICAL", "BILLING"], "disputes_payment": True, "order_id": None}) == "CLARIFY"
assert decide({"topics": ["BILLING"], "disputes_payment": True, "order_id": "ORD-12345"}) == "BILLING"
print(json.dumps(TRIAGE_TOOL["toolSpec"]["inputSchema"]["json"]["properties"], indent=1)[:500])

# %% [markdown]
# One function per call, one grader per technique: the first word for zero/few-shot, the `LABEL:` line for
# chain-of-thought, the rules in code for the structured technique.
#
# > 🔑 **Key code**
# > - `toolConfig={"tools": [...], "toolChoice": {"tool": {"name": "record_triage"}}}` — forces the `toolUse` answer;
# >   the facts are `content[i]["toolUse"]["input"]`, already a Python dict.
# > - the graders: zero/few-shot take the **first word**, chain-of-thought the **last** label in the text (the
# >   reasoning mentions several labels before the final `LABEL:` line).
# > - `r["usage"]` → tokens and dollars per call, the basis for the $ / 1k column.

# %%
def triage(model_id: str, technique: str, msg: str) -> dict:
    kw = dict(modelId=model_id, inferenceConfig={"maxTokens": 600, "temperature": 0})
    t0 = time.perf_counter()
    if technique == "structured":
        # ▶ KEY: forced tool use - the model must return the schema's arguments
        forced = {"tools": [TRIAGE_TOOL], "toolChoice": {"tool": {"name": "record_triage"}}}
        r = bedrock.converse(messages=[{"role": "user", "content": [{"text": DESCRIBE.format(msg=msg)}]}],
                             toolConfig=forced, **kw)
        # ▶ KEY: read the arguments, then let code decide
        facts = next(c["toolUse"]["input"] for c in r["output"]["message"]["content"] if "toolUse" in c)
        label, raw = decide(facts), json.dumps(facts)
    else:
        template = {"zero-shot": ZERO_SHOT, "few-shot": FEW_SHOT, "chain-of-thought": CHAIN_OF_THOUGHT}[technique]
        r = bedrock.converse(messages=[{"role": "user", "content": [{"text": template.format(msg=msg)}]}], **kw)
        raw = m.text_of(r)
        found = re.findall(r"\b(BILLING|TECHNICAL|GENERAL|CLARIFY)\b", raw)
        # ▶ KEY: chain-of-thought -> last label in the text; the others -> the first word
        label = (found[-1] if technique == "chain-of-thought" else raw.split()[0].strip(".:*")) if found else None
    return {"label": label, "raw": raw, "ms": (time.perf_counter() - t0) * 1000, "usage": r["usage"],
            "usd": m.usd(model_id, r["usage"])}

print(triage(m.NOVA_MICRO, "structured", GOLDEN[2][0]))

# %% [markdown]
# > 📊 **How to read this** — one structured call on message #2 ("You took money from my account twice"). `raw` is
# > what the model filled in (`disputes_payment`, `order_id`); `label` is what `decide()` made of it. Note the
# > input-token count: the tool schema is sent with every request, so the structured arm pays for it on each call.
#
# Four techniques × two models × 18 messages = 144 calls, four at a time.
#
# > ⏱️ This cell prints nothing for about 20–30 s.

# %%
TECHNIQUES = ["zero-shot", "few-shot", "chain-of-thought", "structured"]
MODELS = [m.NOVA_MICRO, m.NOVA_LITE]

def run_case(case):
    model_id, technique, i = case
    msg, expected = GOLDEN[i]
    try:
        r = triage(model_id, technique, msg)
    except ClientError as e:
        # ▶ KEY: a throttled call is recorded as a wrong answer, not a crash - the scorecard shows it
        r = {"label": m.error_code(e), "raw": "", "ms": None, "usage": {}, "usd": 0.0}
    return {"model": model_id, "technique": technique, "item": i, "ok": r["label"] == expected, **r}

t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    cases = [(mo, t, i) for mo in MODELS for t in TECHNIQUES for i in range(len(GOLDEN))]
    TRIAGE_RUNS = list(pool.map(run_case, cases))
print(f"{len(TRIAGE_RUNS)} calls in {time.time() - t0:.0f} s")

# %% [markdown]
# ### The scorecard
# *n* = 18, so every accuracy comes with a **Wilson 95 % interval**. Tokens and $ per 1,000 requests are measured from
# the `usage` block of each response.
#
# > 💡 **Concept — an accuracy on 18 items is a range.** 17/18 and 18/18 look different, but the Wilson interval of
# > each spans about 20–25 percentage points: if the true accuracy were 90 %, scores from 15/18 to 18/18 would all be
# > ordinary luck. Cost is different: tokens are counted, not sampled, so a 3× difference in $ / 1k is real
# > even when the accuracy difference is not.

# %%
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

def median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None

SCORE = []
for mo in MODELS:
    for t in TECHNIQUES:
        rs = [r for r in TRIAGE_RUNS if r["model"] == mo and r["technique"] == t]
        k = sum(r["ok"] for r in rs); lo, hi = wilson(k, len(rs))
        SCORE.append({"model": m.model_name(mo), "technique": t, "correct": f"{k}/{len(rs)}", "acc": k / len(rs),
                      "95% CI": f"{lo:.0%}–{hi:.0%}",
                      "in tok": sum(r["usage"].get("inputTokens", 0) for r in rs) / len(rs),
                      "out tok": sum(r["usage"].get("outputTokens", 0) for r in rs) / len(rs),
                      "p50 ms": median(r["ms"] for r in rs), "$ / 1k": 1000 * sum(r["usd"] for r in rs) / len(rs),
                      "missed": " ".join(str(r["item"]) for r in rs if not r["ok"])})
m.table(SCORE, fmt={"acc": "{:.0%}", "in tok": "{:.0f}", "out tok": "{:.0f}", "p50 ms": "{:.0f}", "$ / 1k": "${:.4f}"})

# %% [markdown]
# > 📊 **How to read this** — one row per model × technique. Read **95 % CI** before **acc**: overlapping intervals
# > mean the data cannot rank those arms. **in tok** shows what each technique adds to the prompt (few-shot examples,
# > the tool schema); **out tok** shows chain-of-thought's reasoning — that is where its cost and its p50 latency come
# > from. **missed** lists 0-based item numbers: the same numbers missed by several arms point at an ambiguous golden
# > message or definition; numbers missed by one arm only point at that technique.
#
# ### Which differences are real?
# Same 18 messages for every arm → compare **paired** (McNemar exact: only the messages where two arms disagree count).
#
# > 💡 **Concept — "fixed / broke".** Against the zero-shot baseline, each other arm *fixes* some messages
# > (baseline wrong, arm right) and *breaks* others (baseline right, arm wrong); messages both get right or both get
# > wrong carry no information. If the two arms were equally good, each disagreement would be a coin flip. McNemar's
# > exact p is the chance of a split at least this lopsided under that coin-flip assumption — even +4 / −0 gives
# > p ≈ 0.13; it takes about +6 / −0 to get below 0.05.

# %%
def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float]:
    only_a = sum(x and not y for x, y in zip(a, b)); only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    return only_a, only_b, min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0

def oks(model_id, technique):
    rs = sorted(TRIAGE_RUNS, key=lambda r: r["item"])
    return [r["ok"] for r in rs if r["model"] == model_id and r["technique"] == technique]

rows = []
for mo in MODELS:
    for t in TECHNIQUES[1:]:
        a_only, b_only, p = mcnemar(oks(mo, t), oks(mo, "zero-shot"))
        rows.append([m.model_name(mo), f"{t} vs zero-shot", f"+{a_only} / −{b_only}", f"{p:.2f}",
                     "✓ real difference" if p < 0.05 else "✗ not separable with n = 18"])
m.table(rows, ["Model", "Comparison", "fixed / broke", "McNemar p", "Verdict"])

# %% [markdown]
# > 📊 **How to read this** — each row compares one technique with zero-shot on the same model. **fixed / broke**
# > is the raw evidence (+ messages the technique fixed, − messages it broke); **McNemar p** below 0.05 means a real
# > difference. A ✗ verdict does not say the techniques are equal — it says 18 messages are too few to tell. A row
# > with more "broke" than "fixed" is still a warning: the technique did not help on this model.
#
# One picture: accuracy (with its interval) against cost. The cheapest arm whose interval overlaps the best one is the
# candidate for production — **after** a larger evaluation (Module 9).

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9, 4))
colors = {"zero-shot": "#2563eb", "few-shot": "#f59e0b", "chain-of-thought": "#dc2626", "structured": "#059669"}
for s in SCORE:
    lo, hi = (float(x.strip("%")) for x in s["95% CI"].split("–"))
    y = s["acc"] * 100
    ax.errorbar(s["$ / 1k"], y, yerr=[[y - lo], [hi - y]], fmt="o" if "Micro" in s["model"] else "s",
                color=colors[s["technique"]], capsize=3, alpha=0.85)
    ax.annotate(f'{s["model"].split()[-1]} · {s["technique"]}', (s["$ / 1k"], y),
                xytext=(5, 3), textcoords="offset points", fontsize=9)
ax.xaxis.set_major_formatter(lambda x, _: f"${x:.3f}")
ax.set_xlabel("$ per 1,000 triage requests (circle = Nova Micro, square = Nova Lite)")
ax.set_ylabel("golden messages right (%)")
best = max(SCORE, key=lambda s: (s["acc"], -s["$ / 1k"]))
usd_of = lambda tech: sum(s["$ / 1k"] for s in SCORE if s["technique"] == tech)
cot_x = usd_of("chain-of-thought") / usd_of("zero-shot")
ax.set_title(f"Best: {best['model']} {best['technique']} ({best['correct']}) · "
             f"chain-of-thought costs {cot_x:.1f}× zero-shot", fontsize=10)
plt.tight_layout(); plt.show()

# %% [markdown]
# > 📊 **How to read this** — x = $ per 1,000 triage requests, y = accuracy; circles are Nova Micro, squares Nova
# > Lite, colour = technique, the vertical bar is the Wilson interval. Look left first: the cheapest points whose bars
# > reach as high as the best point are the real candidates. Chain-of-thought (red) sits far to the right — compare
# > its height with zero-shot (blue) on the same model before paying for it. The title names the best arm and the
# > chain-of-thought cost multiple measured in this run.
#
# > **Think about it:** Few-shot examples are supposed to help. Did they help on both models — and if one got worse,
# > what does that say about adding examples to a prompt without a golden set?
#
# <details><summary>More questions</summary>
#
# - Look at the "missed" column of the structured arm: is that a model error or an ambiguity in *our* definition?
# - Chain-of-thought produces a reasoning trail. When is that trail worth its extra cost (see the chart title) — and
#   who reads it?
# </details>
#
# > ✅ **Takeaway §1** — A prompt technique is a hypothesis to test per model, not a best practice: run every arm on
# > the same golden set, report intervals and cost, and move rules the model keeps getting wrong into code.

# %% [markdown]
# ## 2 · The prompt as a governed resource — Prompt Management (slides 9, 39–41)
# The flow in §3 needs the triage prompt as a resource with an owner, an id, **immutable versions** and an audit trail.
# If the structured arm held its own in §1, that is the case for "describe, then decide in code"; a flow prompt node
# returns text, so the managed prompt asks for the same facts as **JSON in text**, and the Lambda validates it. Two
# **variants** of the same template — one per model — let the console compare them side by side.
#
# > 💡 **Concept — Amazon Bedrock Prompt Management's object model.** A **prompt** (`CreatePrompt`) has an id, a name
# > and a working copy called **DRAFT** that you edit with `UpdatePrompt`. The DRAFT holds one or more **variants**:
# > each is a template (with `{{variable}}` placeholders), a model id and inference settings; `defaultVariant` names
# > the one that runs. `CreatePromptVersion` freezes the DRAFT into an **immutable numbered version** with its own ARN
# > (`…:prompt/<id>:<n>`). Applications and flows reference a version ARN, so an edit to the DRAFT never reaches
# > production until someone publishes and points at a new version.
#
# > 🔑 **Key code** — in `variant()`: `templateConfiguration.text.text` (the template), `inputVariables` (the
# > `{{message}}` placeholder declared by name), `modelId` and `inferenceConfiguration` — the model and settings
# > travel *with* the prompt. Note the f-string escaping in `TRIAGE_V1`: `{{{{message}}}}` becomes `{{message}}`.

# %%
TRIAGE_V1 = f"""You record the facts of a message to AnyCompany Insurance's support desk.
{TOPICS}
Return only JSON: {{"topics": [every topic in the message], "disputes_payment": true if {DISPUTE}, else false,
"order_id": "the order number in the format ORD-12345, or null", "sentiment": "POSITIVE|NEUTRAL|NEGATIVE"}}

Message: {{{{message}}}}"""

def variant(name: str, model_id: str, template: str, variables: list[str], max_tokens: int = 300) -> dict:
    # ▶ KEY: a variant = template + declared variables + the model and settings that run it
    return {"name": name, "templateType": "TEXT", "modelId": model_id,
            "templateConfiguration": {"text": {"text": template, "inputVariables": [{"name": v} for v in variables]}},
            "inferenceConfiguration": {"text": {"temperature": 0, "maxTokens": max_tokens}}}

TRIAGE_VARIANTS = [variant("lite", m.NOVA_LITE, TRIAGE_V1, ["message"], 200),
                   variant("micro", m.NOVA_MICRO, TRIAGE_V1, ["message"], 200)]
print(json.dumps(TRIAGE_VARIANTS[0], indent=1)[:900])

# %% [markdown]
# **Publishing is content-addressed:** the version description carries a hash of the variants. Re-running the notebook
# (or a CI pipeline) adopts the version that already has this content instead of minting version 7, 8, 9 of the same
# text — and a changed template always gets a new number.
#
# > 🔑 **Key code**
# > - `find_prompt` pages through `list_prompts` — never assume the first page holds everything.
# > - `sha = sha256(variants + default)[:12]` — the content address, written into the version description.
# > - `update_prompt` / `create_prompt` sets the DRAFT; `create_prompt_version` runs **only** when no existing version
# >   carries this hash. `list_prompts(promptIdentifier=…)` lists the versions of one prompt.

# %%
def find_prompt(name: str):
    pages = agent.get_paginator("list_prompts").paginate()
    return next((p for page in pages for p in page["promptSummaries"] if p["name"] == name), None)

def publish_prompt(short: str, variants: list, default: str, note: str) -> str:
    """Create or update the DRAFT, then return the ARN of the numbered version with exactly this content."""
    name = f"{P}-{short}"
    # ▶ KEY: content address - the same variants always give the same hash
    sha = hashlib.sha256(json.dumps([variants, default], sort_keys=True).encode()).hexdigest()[:12]
    found = find_prompt(name)
    if found:
        pid = found["id"]
        agent.update_prompt(promptIdentifier=pid, name=name, variants=variants, defaultVariant=default,
                            description=f"MLAIDP M04 {short}")
    else:
        pid = agent.create_prompt(name=name, variants=variants, defaultVariant=default,
                                  description=f"MLAIDP M04 {short}", tags=m.TAGS)["id"]
    versions = [v for page in agent.get_paginator("list_prompts").paginate(promptIdentifier=pid)
                for v in page["promptSummaries"] if v["version"] != "DRAFT"]
    same = [v for v in versions if f"[sha {sha}]" in (v.get("description") or "")]
    if same:
        return same[0]["arn"]
    # ▶ KEY: freeze the DRAFT into an immutable numbered version - only for new content
    return agent.create_prompt_version(promptIdentifier=pid, description=f"{note} [sha {sha}]", tags=m.TAGS)["arn"]

TRIAGE_ARN = publish_prompt("triage", TRIAGE_VARIANTS, "lite", "v1: definitions + JSON contract")
print(m.masked(TRIAGE_ARN))

# %% [markdown]
# A version ARN is directly callable: **Converse with `modelId` = prompt version ARN** renders the template with
# `promptVariables` and runs the **default variant's** model and inference settings. The application ships no prompt
# text at all.
#
# > 🔑 **Key code** — `converse(modelId=TRIAGE_ARN, promptVariables={"message": {"text": …}})`: no `messages`, no
# > `inferenceConfig` — the version supplies the template, the model and the settings. The caller needs
# > `bedrock:RenderPrompt` on the prompt *and* `bedrock:InvokeModel` on the model (the flow role in §3 gets both).

# %%
# ▶ KEY: the prompt version ARN takes the place of a model id
r = bedrock.converse(modelId=TRIAGE_ARN, promptVariables={"message": {"text": GOLDEN[1][0]}})
facts = m.parse_json(m.text_of(r))
print(json.dumps(facts), "\n→ route:", decide(facts), "· usage:", r["usage"])

# %% [markdown]
# > 📊 **How to read this** — the first line is the JSON the managed prompt returned for the "angry, three requests"
# > message: look for every topic it mentions and whether the `ORD-` number was found. The route is `decide()`
# > applied to those facts; `usage` shows the input tokens of the rendered template — the cost baseline that §4's
# > "trim the prompt" change request is about.
#
# ### Compare the variants on the golden set
# Variants are a **DRAFT-time** experiment: the numbered version keeps only the default variant (what Converse and the
# flow run). So the comparison reads the variants from the catalog's DRAFT — not from the text in this notebook — renders
# each one and calls its model.
#
# > 🔑 **Key code** — `get_prompt(promptIdentifier=pid)` returns the DRAFT (all variants);
# > `get_prompt(..., promptVersion=ver)` returns the frozen version — compare the two variant lists in the first
# > printed line. `render()` substitutes `{{message}}` the way Prompt Management does, so each variant runs exactly
# > the template stored in the catalog.

# %%
def render(template: str, **values) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda g: values[g.group(1)], template)

def run_variant(v: dict, msg: str) -> dict:
    r = bedrock.converse(modelId=v["modelId"], messages=[{"role": "user", "content": [
        {"text": render(v["templateConfiguration"]["text"]["text"], message=msg)}]}],
        inferenceConfig={"maxTokens": v["inferenceConfiguration"]["text"]["maxTokens"], "temperature": 0})
    facts = m.parse_json(m.text_of(r))
    return {"label": decide(facts) if isinstance(facts, dict) else "NOT JSON", "usage": r["usage"],
            "usd": m.usd(v["modelId"], r["usage"])}

pid, ver = TRIAGE_ARN.split("/")[-1].split(":")
# ▶ KEY: DRAFT (no promptVersion) vs the numbered version
draft, frozen = agent.get_prompt(promptIdentifier=pid), agent.get_prompt(promptIdentifier=pid, promptVersion=ver)
print(f"DRAFT variants: {[v['name'] for v in draft['variants']]} (default {draft['defaultVariant']}) · "
      f"version {ver} variants: {[v['name'] for v in frozen['variants']]}")
with ThreadPoolExecutor(4) as pool:
    VARIANT_RUNS = {v["name"]: list(pool.map(lambda g: {**run_variant(v, g[0]), "exp": g[1]}, GOLDEN))
                    for v in draft["variants"]}
rows = []
for name, rs in VARIANT_RUNS.items():
    k = sum(r["label"] == r["exp"] for r in rs); lo, hi = wilson(k, len(rs))
    missed = " ".join(str(i) for i, r in enumerate(rs) if r["label"] != r["exp"])
    rows.append([name + (" (default)" if name == draft["defaultVariant"] else ""), f"{k}/{len(rs)}",
                 f"{lo:.0%}–{hi:.0%}", 1000 * sum(r["usd"] for r in rs) / len(rs), missed])
m.table(rows, ["Variant", "Correct", "95% CI", "$ / 1k", "Missed items"], fmt={"$ / 1k": "${:.4f}"})

# %% [markdown]
# > 📊 **How to read this** — the first printed line is the governance finding: the DRAFT lists two variants, the
# > numbered version only the default. In the table, one row per variant on the same 18 messages: compare **Correct**
# > through the **95 % CI** column, then **$ / 1k** — a cheaper variant with an overlapping interval is a candidate,
# > not yet a decision. `NOT JSON` in the missed items would mean the model broke the output contract.
#
# The specialists and the clarifier are prompts in the catalog too (same helper, one variant each). The specialist's
# output contract is JSON, and its business rules are in the prompt — §3 checks in code whether the model kept them.
#
# > 🔑 **Key code** — one `SPECIALIST` template with two variables (`{{customer}}` = the CRM briefing,
# > `{{message}}`), published three times with a different `{team}`. The refund limit (100 USD → `ESCALATE`) is a
# > business rule *in the prompt* — the validator in §3 is what turns it into a guarantee.

# %%
SPECIALIST = """You are AnyCompany Insurance's {team} specialist. Write the reply to the customer below.
Customer briefing from the CRM: {{{{customer}}}}
Customer message: {{{{message}}}}
Rules: answer every request in the message; you may refund up to 100 USD yourself (action REFUND); for anything above
100 USD set action ESCALATE, put the full amount in refund_usd and say a supervisor confirms within 2 business days;
address the customer by first name; at most 80 words.
Return only JSON: {{"reply": "...", "action": "one of REFUND, ESCALATE, INFO, TICKET", "refund_usd": number, "confidence": number between 0 and 1}}"""
CLARIFIER = """A customer of AnyCompany Insurance wrote: {{message}}
The message disputes a payment but has no order number. Write ONE short, friendly question that asks for the order
number (format ORD-12345). Return only the question."""

PROMPT_ARNS = {"triage": TRIAGE_ARN}
for short, team in [("billing", "billing"), ("technical", "technical support"), ("general", "customer care")]:
    PROMPT_ARNS[short] = publish_prompt(short, [variant("lite", m.NOVA_LITE, SPECIALIST.format(team=team),
                                                        ["customer", "message"], 400)], "lite", "v1")
PROMPT_ARNS["clarifier"] = publish_prompt("clarifier", [variant("lite", m.NOVA_LITE, CLARIFIER, ["message"], 100)],
                                          "lite", "v1")
m.table([[k, m.masked(v)] for k, v in PROMPT_ARNS.items()], ["Prompt", "Version ARN the flow will pin"])

# %% [markdown]
# > 📊 **How to read this** — five prompts, five version ARNs. The suffix after the last colon is the version
# > number; on a re-run with unchanged text the same numbers come back (content-addressed), not new ones. These exact
# > ARNs go into the flow definition next.
# > **Think about it:** The application now calls a prompt *version ARN*. Who in your company may
# > to create a new version — and who to decide which version production uses?
#
# <details><summary>More questions</summary>
#
# - If the cheaper variant scores the same on 18 messages, what would you need before you switch the default?
# - Why hash the content instead of comparing the latest version's text?
# </details>
#
# > ✅ **Takeaway §2** — In Prompt Management a prompt is a versioned resource: edit the DRAFT, compare variants,
# > freeze a version, and let applications call the version ARN — the code ships no prompt text.

# %% [markdown]
# ## 3 · A flow defined in code (slides 24–35)
# The slide-35 workflow: **analyze** (triage prompt) → **integrate and decide** (Lambda: the routing rules from §1 plus a
# CRM briefing, slide 32) → **route** (condition node) → **structured responses** (specialist prompts) → **log** (trace).
# First the Lambda: the same `decide()` as §1, now behind the flow.
#
# > 💡 **Concept — Amazon Bedrock Flows' object model.** A **flow** is a directed graph: **nodes** (Input, Output,
# > Prompt, LambdaFunction, Condition, and others such as Knowledge Base, Agent, Iterator/Collector, Inline code) with
# > typed, named inputs and outputs, and **connections** — *Data* connections carry a value from one node's output to
# > another node's input, *Conditional* connections leave a Condition node when its expression is true. You edit the
# > flow's **DRAFT**, `PrepareFlow` validates and builds it, `CreateFlowVersion` takes an immutable snapshot, and an
# > **alias** points at a version. `InvokeFlow` (bedrock-agent-runtime) always targets an alias.
#
# > 🔑 **Key code** — the CRM data is deliberately raw (codes such as `TIER_CD`, nulls, lapsed policies): the
# > Lambda's `briefing()` turns it into one sentence a prompt can use, so the model never sees fields it should not
# > act on.

# %%
CRM_DATA = r'''
import json, re
RECORDS = {  # what a CRM really returns: codes, nulls, history nobody needs in a prompt (slide 32)
    "C-1001": {"CUST_NM_FRST": "Maria", "CUST_NM_LST": "Lopez", "TIER_CD": "GLD", "SINCE_DT": "2019-04-02",
               "OPEN_TKT_CNT": 0,
               "POL": [{"POL_NO": "HP-2291", "TYP_CD": "HOME", "STAT_CD": "ACT", "PREM_AMT": 64.0},
                       {"POL_NO": "AU-0912", "TYP_CD": "AUTO", "STAT_CD": "LAP", "PREM_AMT": 0}],
               "LAST_PMT": [{"DT": "2026-10-01", "AMT": 64.0}, {"DT": "2026-10-01", "AMT": 64.0}],
               "RISK_SCR": 0.12, "NOTES": None},
    "C-1002": {"CUST_NM_FRST": "Sam", "CUST_NM_LST": "Okafor", "TIER_CD": "STD", "SINCE_DT": "2025-11-20",
               "OPEN_TKT_CNT": 2,
               "POL": [{"POL_NO": "HP-7781", "TYP_CD": "HOME", "STAT_CD": "ACT", "PREM_AMT": 240.0}],
               "LAST_PMT": [{"DT": "2026-10-03", "AMT": 240.0}, {"DT": "2026-10-03", "AMT": 240.0}], "RISK_SCR": 0.41,
               "NOTES": "app crash reported 2026-10-05"}}
ORDER_ID = re.compile(r"ORD-\d{5}")
'''

# %% [markdown]
# > 🔑 **Key code** — the Lambda node's contract:
# > - **in:** `event["node"]["inputs"]` is a list of `{name, type, value}` — one entry per input the flow definition
# >   declares (`facts`, `customer_id`).
# > - **out:** the return value becomes the node's `functionResponse` output — here an *object*
# >   `{label, facts, briefing}`, so later nodes can pick a field with `$.data.label` or `$.data.briefing`.
# > - the triage prompt returns *text*; the handler strips code fences and treats anything unparseable as `{}` → `GENERAL`
# >   (visible in the trace, never a crash).

# %%
CRM_LOGIC = r'''
def briefing(cid):
    r = RECORDS.get(cid)
    if not r:
        return f"No CRM record for {cid}."
    active = ", ".join(f"{p['TYP_CD'].lower()} {p['POL_NO']} ({p['PREM_AMT']:.0f} USD)"
                       for p in r["POL"] if p["STAT_CD"] == "ACT")
    dup = r["LAST_PMT"][0] == r["LAST_PMT"][1]
    return (f"{r['CUST_NM_FRST']} {r['CUST_NM_LST']}, {'Gold' if r['TIER_CD'] == 'GLD' else 'Standard'} customer since "
            f"{r['SINCE_DT'][:4]}; active: {active}; open tickets: {r['OPEN_TKT_CNT']}"
            + (f"; duplicate payment of {r['LAST_PMT'][0]['AMT']:.0f} USD on {r['LAST_PMT'][0]['DT']}" if dup else "")
            + (f"; note: {r['NOTES']}" if r["NOTES"] else ""))

def decide(facts):
    topics = [str(t).upper() for t in facts.get("topics") or []] or ["GENERAL"]
    order_id = facts.get("order_id") if ORDER_ID.fullmatch(str(facts.get("order_id"))) else None
    if "BILLING" in topics:
        return "CLARIFY" if facts.get("disputes_payment") and not order_id else "BILLING"
    return topics[0] if topics[0] in ("TECHNICAL", "GENERAL") else "GENERAL"

def handler(event, context):
    inputs = {i["name"]: i.get("value") for i in event["node"]["inputs"]}          # named node inputs from the flow
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (inputs.get("facts") or "").strip(), flags=re.M)
    try:
        facts = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
    except Exception:
        facts = {}                                         # unparseable triage -> GENERAL, visible in the trace
    return {"label": decide(facts), "facts": facts, "briefing": briefing(inputs.get("customer_id"))}
'''
CRM_SOURCE = CRM_DATA + CRM_LOGIC                         # one index.py: records first, then the logic
print(len(CRM_SOURCE.splitlines()), "lines of Lambda code")

# %% [markdown]
# > 🔑 **Key code** — `deploy_lambda` is idempotent: `update_function_code` first, `create_function` only on
# > `ResourceNotFoundException`, wrapped in a retry for the "role cannot be assumed yet" error of a fresh IAM role.
# > The `function_updated_v2` waiter blocks until the code is live — invoking earlier can run the old code.

# %%
def deploy_lambda(name: str, source: str, role_arn: str) -> str:
    code = m.lambda_zip(source)
    try:
        # ▶ KEY: update if it exists, create only when it does not - safe to re-run
        lam.update_function_code(FunctionName=name, ZipFile=code)
    except ClientError as e:
        if m.error_code(e) != "ResourceNotFoundException":
            raise
        m.retry(lambda: lam.create_function(FunctionName=name, Runtime="python3.12", Role=role_arn,
                                            Handler="index.handler", Code={"ZipFile": code}, Timeout=15,
                                            MemorySize=256, Architectures=["arm64"], Tags=m.TAGS),
                when="assume|role")
    # ▶ KEY: wait until the new code is live before anyone invokes it
    lam.get_waiter("function_updated_v2").wait(FunctionName=name)
    return lam.get_function(FunctionName=name)["Configuration"]["FunctionArn"]

FN_ARN = deploy_lambda(FN, CRM_SOURCE, FN_ROLE_ARN)
test = json.loads(lam.invoke(FunctionName=FN, Payload=json.dumps({"node": {"inputs": [
    {"name": "facts", "value": '{"topics": ["BILLING"], "disputes_payment": true, "order_id": null}'},
    {"name": "customer_id", "value": "C-1001"}]}}))["Payload"].read())
print(test)

# %% [markdown]
# > 📊 **How to read this** — a direct test of the Lambda with the event shape a flow sends: facts that dispute a
# > payment without an order id must give `label` `CLARIFY`, and `briefing` shows what the specialist prompt will see
# > about customer C-1001 — readable words instead of CRM codes, and the duplicate payment spelled out.
#
# The flow definition is **data**: nodes with typed inputs/outputs, and connections. Prompt nodes pin a **prompt version
# ARN**; the Lambda node gets two named inputs; the condition node reads `label` from the Lambda's object output.
#
# > 🔑 **Key code**
# > - `prompt_node`: `sourceConfiguration.resource.promptArn` — the node references a catalog prompt *version*
# >   instead of embedding text.
# > - `expression` on each input is a JSONPath into what arrives on that input: `$.data.message` picks a field from
# >   the Start document, `$.data` takes the whole value.
# > - `data(...)` = a *Data* connection (`sourceOutput` → `targetInput`); `route(...)` = a *Conditional* connection
# >   from the `Route` node, named after the condition that must be true.

# %%
def prompt_node(name: str, arn: str, inputs: dict) -> dict:
    # ▶ KEY: the node pins a prompt version ARN from the catalog
    return {"name": name, "type": "Prompt",
            "configuration": {"prompt": {"sourceConfiguration": {"resource": {"promptArn": arn}}}},
            "inputs": [{"name": k, "type": "String", "expression": e} for k, e in inputs.items()],
            "outputs": [{"name": "modelCompletion", "type": "String"}]}

def output_node(name: str) -> dict:
    return {"name": name, "type": "Output", "configuration": {"output": {}},
            "inputs": [{"name": "document", "type": "String", "expression": "$.data"}]}

def data(src: str, out: str, dst: str, inp: str) -> dict:
    return {"name": f"{src}_{dst}_{inp}", "source": src, "target": dst, "type": "Data",
            "configuration": {"data": {"sourceOutput": out, "targetInput": inp}}}

def route(cond: str, dst: str) -> dict:
    # ▶ KEY: a Conditional connection - followed only when the named condition is true
    return {"name": f"Route_{dst}", "source": "Route", "target": dst, "type": "Conditional",
            "configuration": {"conditional": {"condition": cond}}}

SPECIALISTS = {"BILLING": "Billing", "TECHNICAL": "Technical", "GENERAL": "General"}

# %% [markdown]
# ![The support flow: Start → Triage → DecideAndCRM → Route → four prompt nodes → four outputs](../assets/diagrams/m04_flow_graph.svg)
#
# > 🔑 **Key code** — in `flow_definition`:
# > - `DecideAndCRM` takes `facts` = the Triage completion (`$.data`) and `customer_id` from the Start document.
# > - `Route` has three named conditions plus `default`; its input `label` is `$.data.label` of the Lambda's object.
# > - the loop adds, per specialist, a prompt node fed by **two** connections (message from Start, briefing from the
# >   Lambda) and its own Output node — one flow, four possible exits.

# %%
def flow_definition(arns: dict) -> dict:
    nodes = [
        {"name": "Start", "type": "Input", "configuration": {"input": {}},
         "outputs": [{"name": "document", "type": "Object"}]},
        prompt_node("Triage", arns["triage"], {"message": "$.data.message"}),
        {"name": "DecideAndCRM", "type": "LambdaFunction", "configuration": {"lambdaFunction": {"lambdaArn": FN_ARN}},
         "inputs": [{"name": "facts", "type": "String", "expression": "$.data"},
                    {"name": "customer_id", "type": "String", "expression": "$.data.customer_id"}],
         "outputs": [{"name": "functionResponse", "type": "Object"}]},
        # ▶ KEY: the routing decision is a Condition node reading the Lambda's label
        {"name": "Route", "type": "Condition", "configuration": {"condition": {"conditions": [
            {"name": "isBilling", "expression": 'label == "BILLING"'},
            {"name": "isTechnical", "expression": 'label == "TECHNICAL"'},
            {"name": "isClarify", "expression": 'label == "CLARIFY"'}, {"name": "default"}]}},
         "inputs": [{"name": "label", "type": "String", "expression": "$.data.label"}]},
        prompt_node("Clarify", arns["clarifier"], {"message": "$.data.message"}), output_node("ClarifyOut")]
    conns = [data("Start", "document", "Triage", "message"),
             data("Triage", "modelCompletion", "DecideAndCRM", "facts"),
             data("Start", "document", "DecideAndCRM", "customer_id"),
             data("DecideAndCRM", "functionResponse", "Route", "label"),
             route("isClarify", "Clarify"), data("Start", "document", "Clarify", "message"),
             data("Clarify", "modelCompletion", "ClarifyOut", "document")]
    for label, node in SPECIALISTS.items():
        nodes += [prompt_node(node, arns[node.lower()], {"customer": "$.data.briefing", "message": "$.data.message"}),
                  output_node(f"{node}Out")]
        conns += [route({"BILLING": "isBilling", "TECHNICAL": "isTechnical", "GENERAL": "default"}[label], node),
                  data("Start", "document", node, "message"), data("DecideAndCRM", "functionResponse", node, "customer"),
                  data(node, "modelCompletion", f"{node}Out", "document")]
    return {"nodes": nodes, "connections": conns}

DEFINITION = flow_definition(PROMPT_ARNS)
for n in DEFINITION["nodes"]:
    ins = ", ".join(i["name"] + "=" + i["expression"] for i in n.get("inputs", []))
    print(f'{n["type"]:15s} {n["name"]:13s} in: {ins}')
print(len(DEFINITION["connections"]), "connections, e.g.", json.dumps(DEFINITION["connections"][1]))

# %% [markdown]
# > 📊 **How to read this** — one line per node: its type, its name and each input with the JSONPath expression that
# > fills it. Check the data path against the diagram above: Triage reads the message, DecideAndCRM reads the triage
# > text and the customer id, Route reads `label`, each specialist reads the briefing and the message. The printed
# > connection is the JSON for one arrow of the diagram.
#
# The execution role gets exactly what this flow uses: the five prompts, the one model behind them (Nova Lite, through
# its inference profile), one Lambda.
# The Lambda's resource policy admits Bedrock only for **this** flow (`SourceArn`).
#
# > 💡 **Concept — two policies for one call.** When the flow runs, Bedrock acts with the flow's **execution role**:
# > it needs `bedrock:GetPrompt` / `RenderPrompt` on the prompts, `bedrock:InvokeModel` on the model, and
# > `lambda:InvokeFunction` on the Lambda. The Lambda must *also* accept the call: its **resource-based policy**
# > (`add_permission` in `publish_flow` below) allows the service principal `bedrock.amazonaws.com` only when
# > `SourceArn` is this flow. Model access through a `us.` inference profile needs both the profile ARN and the
# > underlying foundation-model ARNs (any Region the profile can route to — hence `*` in the Region field).
#
# > 🔑 **Key code** — `prompts` strips the `:N` version suffix and adds `:*`, so the policy covers every version of
# > these five prompts and nothing else; `models` lists the profile and the foundation model; one Lambda ARN.

# %%
def flow_role_policy(arns: dict) -> dict:
    prompts = sorted({a.rsplit(":", 1)[0] for a in arns.values()})                     # prompt/ID covers prompt/ID:N
    # ▶ KEY: the inference profile AND the foundation model it routes to
    models = [f"arn:aws:bedrock:{REGION}:{ACCOUNT}:inference-profile/{m.NOVA_LITE}",
              f"arn:aws:bedrock:*::foundation-model/{m.NOVA_LITE.split('.', 1)[1]}"]
    return {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["bedrock:GetPrompt", "bedrock:RenderPrompt"],
         "Resource": prompts + [p + ":*" for p in prompts]},
        {"Effect": "Allow", "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
         "Resource": models},
        {"Effect": "Allow", "Action": "lambda:InvokeFunction", "Resource": FN_ARN}]}

iam.put_role_policy(RoleName=FLOW_ROLE, PolicyName="flow", PolicyDocument=json.dumps(flow_role_policy(PROMPT_ARNS)))
print(m.masked(json.dumps(flow_role_policy(PROMPT_ARNS)["Statement"][0]["Resource"][:2])))

# %% [markdown]
# Publishing a flow: update the **DRAFT** → `prepare_flow` (validates the graph) → **version** (immutable snapshot) →
# point the **alias** `live` at it. Applications invoke the alias, never a version — that is what makes rollback a
# pointer move. Versions are content-addressed again, so a re-run adopts the existing version.
#
# > 🔑 **Key code**
# > - `lam.add_permission(..., Principal="bedrock.amazonaws.com", SourceArn=<this flow>)` — the Lambda's side of the
# >   trust; `ResourceConflictException` means the statement already exists (re-run).
# > - `prepare_flow` + the wait for `Prepared` — a graph with a dangling input or a bad expression fails *here*,
# >   with the reasons in `validations`, not at invoke time.
# > - `create_flow_version` only when no version carries this definition's hash; `point_alias` = `update_flow_alias`
# >   with `routingConfiguration=[{"flowVersion": …}]` (or `create_flow_alias` the first time).

# %%
def find_flow():
    pages = agent.get_paginator("list_flows").paginate()
    return next((f for page in pages for f in page["flowSummaries"] if f["name"] == FLOW), None)

def publish_flow(definition: dict, note: str) -> str:
    sha = hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()[:12]
    f = find_flow()
    args = dict(name=FLOW, executionRoleArn=FLOW_ROLE_ARN, definition=definition,
                description="MLAIDP M04 support workflow")
    fid = agent.update_flow(flowIdentifier=f["id"], **args)["id"] if f else \
        m.retry(lambda: agent.create_flow(**args, tags=m.TAGS)["id"], when="role|assume|AccessDenied")
    try:
        # ▶ KEY: the Lambda admits Bedrock only on behalf of this flow
        lam.add_permission(FunctionName=FN, StatementId="bedrock-flow", Action="lambda:InvokeFunction",
                           Principal="bedrock.amazonaws.com", SourceArn=f"arn:aws:bedrock:{REGION}:{ACCOUNT}:flow/{fid}")
    except ClientError as e:
        if m.error_code(e) != "ResourceConflictException":
            raise
    # ▶ KEY: validate and build the DRAFT - graph errors surface here
    agent.prepare_flow(flowIdentifier=fid)
    st = m.wait_for(lambda: agent.get_flow(flowIdentifier=fid),
                    until=lambda r: r["status"] in ("Prepared", "Failed") or r["status"],
                    timeout=120, every=2, label="prepare_flow")
    assert st["status"] == "Prepared", st.get("validations")
    for v in agent.list_flow_versions(flowIdentifier=fid)["flowVersionSummaries"]:
        if v["version"] == "DRAFT":
            continue
        if f"[sha {sha}]" in agent.get_flow_version(flowIdentifier=fid, flowVersion=v["version"]).get("description", ""):
            return v["version"]
    # ▶ KEY: an immutable snapshot of the prepared DRAFT
    return agent.create_flow_version(flowIdentifier=fid, description=f"{note} [sha {sha}]")["version"]

# %%
def point_alias(version: str) -> float:
    t0, fid = time.perf_counter(), find_flow()["id"]
    aliases = agent.list_flow_aliases(flowIdentifier=fid)["flowAliasSummaries"]
    alias, routing = next((a for a in aliases if a["name"] == ALIAS), None), [{"flowVersion": version}]
    if alias:
        # ▶ KEY: release and rollback are the same call - move the pointer
        agent.update_flow_alias(flowIdentifier=fid, aliasIdentifier=alias["id"], name=ALIAS, routingConfiguration=routing)
    else:
        agent.create_flow_alias(flowIdentifier=fid, name=ALIAS, routingConfiguration=routing, tags=m.TAGS)
    return time.perf_counter() - t0

V_GOOD = publish_flow(DEFINITION, "release 1: triage v1")
print(f"flow version {V_GOOD}; alias {ALIAS} → {V_GOOD} in {point_alias(V_GOOD):.2f} s")

# %% [markdown]
# > 📊 **How to read this** — `prepare_flow ready after N s` is the validation step; then the version number
# > (1 on a fresh account, the same number again on a re-run with an unchanged definition) and how long the alias
# > update took — the number to remember for the rollback in §4.
#
# ### Invoke with trace (the "log for quality assurance" step)
# `InvokeFlow` with `enableTrace=True` streams one event per node input, output, condition result and dependency call,
# each with a timestamp. The function returns the output, the route the condition node took, and the trace.
#
# > 💡 **Concept — the response is an event stream.** `InvokeFlow` returns `responseStream`: `flowOutputEvent`
# > (the document an Output node produced, with the node's name), `flowTraceEvent` (only with `enableTrace=True`) and
# > a final `flowCompletionEvent`. Trace events come in kinds — `nodeInputTrace`, `nodeOutputTrace`,
# > `conditionNodeResultTrace` (which conditions were satisfied) and `nodeActionTrace` (the call a node made to
# > another service: the `Converse` request and response *with token usage*, or the Lambda invocation).
# > Errors such as an IAM delay can also arrive *inside* the stream, so the retry wraps the whole call.
#
# > 🔑 **Key code** — `invoke_flow(flowIdentifier, flowAliasIdentifier, enableTrace=True, inputs=[{nodeName: "Start",
# > nodeOutputName: "document", content: {document: {...}}}])`; the loop sorts events by kind; the last lines sum
# > the `usage` of every prompt node's `Converse` call into tokens and dollars per request.

# %%
def invoke(message: str, customer_id: str = "C-1001") -> dict:
    fid = find_flow()["id"]
    alias = next(a for a in agent.list_flow_aliases(flowIdentifier=fid)["flowAliasSummaries"] if a["name"] == ALIAS)
    return m.retry(lambda: _invoke(fid, alias, message, customer_id), when="not authorized|dependencyFailed")

def _invoke(fid: str, alias: dict, message: str, customer_id: str) -> dict:
    t0 = time.perf_counter()        # errors such as an IAM delay arrive inside the event stream - retry the whole call
    # ▶ KEY: invoke the alias (never the DRAFT) with the per-node trace switched on
    resp = agent_rt.invoke_flow(flowIdentifier=fid, flowAliasIdentifier=alias["id"], enableTrace=True,
                                inputs=[{"nodeName": "Start", "nodeOutputName": "document",
                                         "content": {"document": {"message": message, "customer_id": customer_id}}}])
    out = {"output": None, "node": None, "route": None, "trace": [],
           "version": alias["routingConfiguration"][0]["flowVersion"]}
    for ev in resp["responseStream"]:
        ms = (time.perf_counter() - t0) * 1000
        if "flowOutputEvent" in ev:
            out["output"], out["node"] = ev["flowOutputEvent"]["content"]["document"], ev["flowOutputEvent"]["nodeName"]
        elif "flowTraceEvent" in ev:
            t = ev["flowTraceEvent"]["trace"]
            kind, body = next(iter(t.items()))
            if kind == "conditionNodeResultTrace":
                # ▶ KEY: which condition fired = the route this request took
                out["route"] = ", ".join(c["conditionName"] for c in body.get("satisfiedConditions", []))
            out["trace"].append({"ms": ms, "node": body.get("nodeName"), "event": kind.replace("Trace", ""),
                                 "body": body})
    # every prompt node's Converse call (request, response, usage) is in a nodeActionTrace -> tokens and $ per request
    # ▶ KEY: cost per request straight from the trace - no extra instrumentation
    usage = [t["body"]["operationResponse"]["usage"] for t in out["trace"]
             if t["event"] == "nodeAction" and t["body"].get("operationName") == "Converse"]
    out["tokens"] = sum(u["inputTokens"] + u["outputTokens"] for u in usage)
    out["usd"] = sum(m.usd(m.NOVA_LITE, u) for u in usage)          # every prompt's default variant runs Nova Lite
    out["ms"] = (time.perf_counter() - t0) * 1000
    out["json"] = m.parse_json(out["output"]) if isinstance(out["output"], str) else None
    return out

first = invoke(GOLDEN[0][0], "C-1001")
print(first["node"], "·", first["route"], f"· {first['ms']:.0f} ms ·", first["output"][:200])

# %% [markdown]
# > 📊 **How to read this** — Output node · condition that fired · end-to-end time · the start of the specialist's
# > JSON. The first invoke of a fresh flow is slower than later ones (IAM and Lambda warm-up), so judge latency
# > from the trace below and from §4, not from this one call.
#
# The per-node timeline: what each node received and produced, and when.

# %%
def show_trace(res: dict, width: int = 110):
    rows = []
    for t in res["trace"]:
        b = t["body"]
        if t["event"] in ("nodeInput", "nodeOutput"):
            val = "; ".join(f'{f.get("nodeInputName") or f.get("nodeOutputName")}={json.dumps(f["content"]["document"])}'
                            for f in b["fields"])
        elif t["event"] == "conditionNodeResult":
            val = "→ " + res["route"]
        elif t["event"] == "nodeAction" and b["operationName"] == "Converse":
            u, req, lat = b["operationResponse"]["usage"], b["operationRequest"], b["operationResponse"]["metrics"]
            val = (f'Converse modelId=…{req["modelId"].split("/")[-1]} · '
                   f'{u["inputTokens"]} in / {u["outputTokens"]} out tokens · {lat["latencyMs"]} ms')
        else:
            fn = b.get("operationRequest", {}).get("functionName", "").split(":")[-1]
            val = f'{b.get("operationName", "")} {fn}'.strip()
        rows.append([f'{t["ms"]:.0f}', t["node"], t["event"], val[:width]])
    m.table(rows, ["ms", "Node", "Event", "Value"])

show_trace(first)
print(f"{first['tokens']} tokens · ${first['usd'] * 1000:.3f} per 1,000 requests")

# %% [markdown]
# > 📊 **How to read this** — one row per trace event, in time order (**ms** since the call started). Follow the
# > **Node** column down the flow: Start → Triage → DecideAndCRM → Route → one specialist → its Output. The two
# > `nodeAction … Converse` rows are the model calls — the modelId ends with the *prompt id:version* (the flow calls
# > Converse with the version ARN), and they show tokens and model latency; compare their ms with the gaps between
# > rows to see where the time goes. The `conditionNodeResult` row is the routing decision; the `nodeAction` of the
# > Lambda is the CRM call.
#
# ### Structured is not the same as correct (slide 33)
# The three inputs from the slide-35 tip. A validator in code checks the specialist's JSON against the
# **business rules** the prompt asked for — the step that makes this production-ready.
#
# > 💡 **Concept — syntax, schema, semantics.** Valid JSON is the first check, the right keys and enum values the
# > second — but the one that protects the business is the third: does the content obey the rules? A reply that
# > says `"action": "REFUND", "refund_usd": 240` is perfectly structured and breaks the 100 USD limit. A validator
# > in code after the model is cheap, deterministic, and the place to block, escalate or log.
#
# > 🔑 **Key code** — `business_check` runs the three levels in order: parsed JSON, required keys and the allowed
# > `action` values, then the rule "refund above 100 USD must be `ESCALATE`".

# %%
def business_check(res: dict) -> str:
    if res["node"] == "ClarifyOut":
        return "clarification question – nothing to refund"
    j = res["json"]
    if not isinstance(j, dict):
        return "✗ not JSON"
    problems = [f"missing {k}" for k in ("reply", "action", "refund_usd", "confidence") if k not in j]
    if j.get("action") not in ("REFUND", "ESCALATE", "INFO", "TICKET"):
        problems.append(f"unknown action {j.get('action')}")
    # ▶ KEY: the business rule, checked in code - structure alone does not catch this
    if isinstance(j.get("refund_usd"), (int, float)) and j["refund_usd"] > 100 and j.get("action") == "REFUND":
        problems.append(f"refund {j['refund_usd']} USD > 100 USD without ESCALATE")
    return "✓ OK" if not problems else "✗ " + "; ".join(problems)

DEMO_INPUTS = [("simple billing", GOLDEN[0][0], "C-1001"), ("angry, three requests", GOLDEN[1][0], "C-1002"),
               ("no order number", GOLDEN[2][0], "C-1001")]
with ThreadPoolExecutor(3) as pool:
    demo = list(pool.map(lambda x: invoke(x[1], x[2]), DEMO_INPUTS))
rows = []
for (name, _, _), r in zip(DEMO_INPUTS, demo):
    j = r["json"] or {}
    rows.append([name, r["route"], r["node"], f'{r["ms"]:.0f}', j.get("action", "–"), j.get("refund_usd", "–"),
                 str(j.get("reply", r["output"]))[:90], business_check(r)])
m.table(rows, ["Input", "Route", "Output node", "ms", "Action", "Refund", "Reply", "Business rules"])

# %% [markdown]
# > 📊 **How to read this** — one row per slide-35 input. **Route** and **Output node** show the path; **Action** and
# > **Refund** are the specialist's decision. Read the last column first: ✓ means the reply passed all three checks,
# > ✗ names the rule that broke. For the angry message, compare **Refund** with the duplicate charge in the message
# > and check that a refund above 100 USD comes with `ESCALATE` and a supervisor in the reply. The no-order-number
# > input must end in **ClarifyOut** with a question, not a refund.
#
# > **Think about it:** Where is state passed between steps in this flow — and what happens to the customer if the
# > CRM Lambda times out?
#
# <details><summary>More questions</summary>
#
# - The routing rules run in a Lambda, not in the triage prompt. What did §1 measure that justifies that?
# - Which node would you unit-test first, and which one needs a golden set?
# </details>
#
# > ✅ **Takeaway §3** — A Bedrock Flow is a reviewable JSON graph that pins prompt versions, runs rules in a Lambda
# > and records every node in the trace; a code validator after the model turns "structured" into "correct".

# %% [markdown]
# ## 4 · Release, regress, roll back (slides 40–42)
# **Baseline:** the whole golden set through alias `live`. Route accuracy is read from the trace (which condition fired),
# business rules from the validator.
#
# ![Release, gate, roll back: prompt versions, flow versions, the alias pointer](../assets/diagrams/m04_release_alias.svg)
#
# > 💡 **Concept — what is immutable and what moves.** Prompt versions and flow versions never change once created;
# > a flow version pins exact prompt versions. The only mutable thing in production is the **alias** — a named
# > pointer (`routingConfiguration`) to one flow version. A release creates new versions and moves the pointer; a
# > rollback moves it back. Nothing is redeployed, and the bad version stays for the post-mortem.
#
# > 🔑 **Key code** — `golden_through_flow` sends all 18 messages through the **alias** (exactly what production
# > runs) and maps the Output node back to a label; `summary` turns the rows into the release record: version,
# > routes right, rule violations, p50 latency, $ / 1k requests, and which messages were misrouted.

# %%
CUSTOMER = lambda i: "C-1001" if i % 2 == 0 else "C-1002"

def golden_through_flow() -> list[dict]:
    with ThreadPoolExecutor(4) as pool:
        res = list(pool.map(lambda i: invoke(GOLDEN[i][0], CUSTOMER(i)), range(len(GOLDEN))))
    # ▶ KEY: the route is read from which Output node answered
    to_label = {"BillingOut": "BILLING", "TechnicalOut": "TECHNICAL", "GeneralOut": "GENERAL", "ClarifyOut": "CLARIFY"}
    return [{"item": i, "routed": to_label.get(r["node"]), "ok": to_label.get(r["node"]) == GOLDEN[i][1],
             "rules": business_check(r), "ms": r["ms"], "usd": r["usd"], "version": r["version"]}
            for i, r in enumerate(res)]

def summary(rows: list[dict]) -> dict:
    k = sum(r["ok"] for r in rows)
    return {"flow version": rows[0]["version"], "routes right": f"{k}/{len(rows)}", "acc": k / len(rows),
            "rule violations": " ".join(str(r["item"]) for r in rows if r["rules"].startswith("✗")) or "none",
            "p50 ms": median(r["ms"] for r in rows), "$ / 1k requests": 1000 * sum(r["usd"] for r in rows) / len(rows),
            "misrouted": " ".join(f'{r["item"]}→{r["routed"]}' for r in rows if not r["ok"])}

BASELINE = golden_through_flow()
m.table([summary(BASELINE)], fmt={"acc": "{:.0%}", "p50 ms": "{:.0f}", "$ / 1k requests": "${:.3f}"})
for r in BASELINE:
    if r["rules"].startswith("✗"):
        print(f'  #{r["item"]} {r["rules"]}')

# %% [markdown]
# > 📊 **How to read this** — the release record of the version `live` points to. **routes right** is the route
# > accuracy over the 18 golden messages; **rule violations** lists item numbers whose specialist reply failed the
# > validator (printed below the table with the reason). A violation in the *baseline* is not a routing error — the
# > message went to the right team, but the reply broke a rule: look at which message it is and what the CRM
# > briefing told the specialist. These two columns are what the gate compares next.
# **The change request:** "The triage prompt is too long — trim it, we pay for every input token." The engineer keeps
# the label names and the JSON keys, drops the definitions. It reads fine — and how many tokens does it save? The next
# cell counts them.
# The release goes the way most teams do it: new prompt version → new flow version → alias `live` moved.
#
# > 🔑 **Key code** — the token count calls Converse on both rendered templates with the same message and reads
# > `usage.inputTokens`. The release reuses every helper: `publish_prompt` (version 2 of the triage prompt), the
# > IAM policy for the new ARN, `publish_flow` (a new definition → new hash → new flow version) and `point_alias`.

# %%
TRIAGE_V2 = """Triage this support message. Topics: BILLING, TECHNICAL, GENERAL.
Return only JSON: {"topics": [...], "disputes_payment": true|false, "order_id": "ORD-12345 or null", "sentiment": "POSITIVE|NEUTRAL|NEGATIVE"}

Message: {{message}}"""
tokens = {}
for name, tpl in [("v1", TRIAGE_V1), ("v2", TRIAGE_V2)]:
    msg = [{"role": "user", "content": [{"text": render(tpl, message=GOLDEN[0][0])}]}]
    r = bedrock.converse(modelId=m.NOVA_LITE, messages=msg, inferenceConfig={"maxTokens": 200, "temperature": 0})
    tokens[name] = r["usage"]["inputTokens"]
print(f"input tokens per triage call: v1 {tokens['v1']} → v2 {tokens['v2']} ({1 - tokens['v2'] / tokens['v1']:.0%} fewer)")

V2_VARIANTS = [variant("lite", m.NOVA_LITE, TRIAGE_V2, ["message"], 200),
               variant("micro", m.NOVA_MICRO, TRIAGE_V2, ["message"], 200)]
CANDIDATE_ARNS = {**PROMPT_ARNS, "triage": publish_prompt("triage", V2_VARIANTS, "lite", "v2: trimmed to save tokens")}
iam.put_role_policy(RoleName=FLOW_ROLE, PolicyName="flow", PolicyDocument=json.dumps(flow_role_policy(CANDIDATE_ARNS)))
# ▶ KEY: the release - new flow version, then the alias moves (before any test: the mistake §4 is about)
V_CANDIDATE = publish_flow(flow_definition(CANDIDATE_ARNS), "release 2: triage v2 (trimmed)")
print(f"triage prompt version {CANDIDATE_ARNS['triage'].split(':')[-1]} → flow version {V_CANDIDATE}; "
      f"alias {ALIAS} → {V_CANDIDATE} in {point_alias(V_CANDIDATE):.2f} s")

# %% [markdown]
# > 📊 **How to read this** — first line: input tokens per triage call before and after the trim — the saving the
# > change request promised, for the triage step only. Last line: the new prompt version, the new flow version, and
# > the alias already moved to it. Production now runs the trimmed prompt — untested.
#
# ### The regression gate
# Same golden set, now through the release. The gate is a **non-regression rule**: no golden message that the live
# version routes correctly may break, and no new business-rule violations. (A significance test is for sampled traffic;
# a curated golden message that breaks is a stop on its own.)
#
# > 💡 **Concept — a gate, not a vote.** A golden message is a requirement someone signed off, so the gate compares
# > *message by message*: anything the baseline routed right and the candidate routes wrong is a regression, however
# > good the average looks. That differs from an A/B test on live traffic, where you sample and need statistics.
# > The gate also reports the cost change, because "cheaper" was the reason for the release — and is now weighed
# > against what broke.
#
# > 🔑 **Key code** — `broke` = items where the baseline was right and the candidate is wrong; `new_violations` =
# > the change in rule violations; `GATE_PASSED` needs no broken message and no increase in violations.

# %%
CANDIDATE = golden_through_flow()
m.table([summary(BASELINE), summary(CANDIDATE)], fmt={"acc": "{:.0%}", "p50 ms": "{:.0f}", "$ / 1k requests": "${:.3f}"})
# ▶ KEY: per-message non-regression - right before, wrong now
broke = [i for i, (b, c) in enumerate(zip(BASELINE, CANDIDATE)) if b["ok"] and not c["ok"]]
new_violations = sum(c["rules"].startswith("✗") for c in CANDIDATE) - sum(b["rules"].startswith("✗") for b in BASELINE)
base_usd, cand_usd = (sum(r["usd"] for r in rows) / len(rows) for rows in (BASELINE, CANDIDATE))
print(f"cost per 1,000 requests: ${1000 * base_usd:.3f} → ${1000 * cand_usd:.3f} "
      f"({1 - cand_usd / base_usd:.0%} cheaper)")
for i in broke:
    print(f"  broke #{i}: {GOLDEN[i][0][:70]!r}  expected {GOLDEN[i][1]}, now {CANDIDATE[i]['routed']}")
# ▶ KEY: the release decision
GATE_PASSED = not broke and new_violations <= 0
plural = lambda n, word: f"{n} {word}{'' if n == 1 else 's'}"
print("✓ gate passed – keep the release" if GATE_PASSED else
      f"✗ gate failed: {plural(len(broke), 'golden message')} broke, "
      f"{plural(max(new_violations, 0), 'new rule violation')} – roll back")

# %% [markdown]
# > 📊 **How to read this** — the table puts the two release records side by side: compare **routes right** and
# > **misrouted** first (which messages moved to which route), then **$ / 1k requests**. The cost line shows the
# > saving per *request* — smaller than the triage-token saving, because the specialist prompts are unchanged. Each
# > `broke #` line names a golden message and where it went now; read it next to the definitions the trim removed.
# > The last line is the gate's verdict.
#
# ### Roll back = move the pointer
# The bad version stays in the catalog for the post-mortem; production goes back to the last good version.
#
# > 🔑 **Key code** — `point_alias(V_GOOD)` is the whole rollback. The check that follows does not trust it: it
# > invokes the alias with the first message that broke and confirms the version that answered and the route taken.

# %%
if not GATE_PASSED:
    # ▶ KEY: rollback = one alias update
    print(f"alias {ALIAS} → {V_GOOD} in {point_alias(V_GOOD):.2f} s (rollback)")
probe = broke[0] if broke else 0
check = invoke(GOLDEN[probe][0], CUSTOMER(probe))
fid = find_flow()["id"]
versions = sorted(agent.list_flow_versions(flowIdentifier=fid)["flowVersionSummaries"],
                  key=lambda v: int(v["version"]) if v["version"].isdigit() else 0)
m.table([[v["version"],
          agent.get_flow_version(flowIdentifier=fid, flowVersion=v["version"]).get("description", "")
          if v["version"] != "DRAFT" else "(working copy)",
          "← live" if v["version"] == check["version"] else ""] for v in versions],
        ["Flow version", "Description", ""])
routed_ok = check["node"] == SPECIALISTS.get(GOLDEN[probe][1], "Clarify") + "Out"
if GATE_PASSED:      # nothing to roll back - but Part 2 expects live -> the good version: run point_alias(V_GOOD)
    print(f"– gate passed this time: alias {ALIAS} stays on version {check['version']}. "
          f"Before Part 2, run point_alias(V_GOOD) to put {ALIAS} back on version {V_GOOD}.")
elif check["version"] == V_GOOD and routed_ok:
    print(f"✓ alias {ALIAS} → version {check['version']}; message #{probe} is routed to {check['node']} again")
else:
    print(f"✗ alias → {check['version']}, message #{probe} → {check['node']}")

# %% [markdown]
# > 📊 **How to read this** — first line: how long the pointer move took. The table lists every flow version with
# > its description (and the content hash); `← live` marks the version that answered the probe. Both versions are
# > still there — only the pointer moved. The last line confirms (✓) or refutes (✗) that the broken message is routed
# > correctly again; if the gate passed this time, it tells you how to put `live` back before Part 2.
# > **Think about it:** The bad release reached `live` before the gate ran. Change the order: what would a `staging`
# > alias and this gate look like in your CI/CD pipeline (slide 42)?
#
# <details><summary>More questions</summary>
#
# - If the trimmed prompt saves about half the triage tokens, at what traffic does that saving pay for the golden-set
#   run on every change?
# - If messages broke, which words in the v1 definitions were doing the work?
# </details>
#
# > ✅ **Takeaway §4** — Every prompt edit is a release: run the golden set through it, block any message that
# > breaks, and keep production on an alias so that rollback is a pointer move, not a redeployment.
#
# ## Summary — what you built
# | Piece | AWS API / pattern | Production next step |
# |---|---|---|
# | Technique scorecard | `Converse`, forced `toolChoice`, Wilson CI, McNemar | a larger golden set and Amazon Bedrock Evaluations (Module 9) |
# | Governed prompts | Prompt Management: `CreatePrompt`, variants, `CreatePromptVersion`, Converse with a version ARN | IAM split: who edits DRAFTs vs who publishes versions; approval in the pipeline |
# | Support workflow | Bedrock Flows: Prompt / Lambda / Condition nodes, `PrepareFlow`, versions, alias | Guardrails on prompt nodes (Module 6), invocation logging and dashboards (Module 8) |
# | Least-privilege wiring | execution role scoped to 5 prompts, 1 model, 1 Lambda; Lambda `SourceArn` permission | infrastructure as code (CloudFormation / CDK) for the roles and the flow |
# | Release gate + rollback | golden set through the alias, per-message non-regression, `UpdateFlowAlias` | a `staging` alias and this gate in CI/CD before `live` moves |

# %% [markdown]
# ## Cleanup — don't skip this
# Deletes the Lambda first (so no late invocation recreates its log group), then the flow (alias, versions), the five
# prompts (all versions) and both roles, and finally the Lambda's log group — after a pause, because log delivery can
# lag behind the delete and recreate the group. **Run it only after Part 2 (console)** if you show the flow, the
# prompts or the alias there. The cells only need §0 Setup (re-run it first after a kernel restart).

# %% tags=["cleanup"]
logs = m.client("logs")
try:
    lam.delete_function(FunctionName=FN)                               # first: nothing can invoke it any more
    print("  deleted lambda", FN)
except ClientError as e:
    if m.error_code(e) != "ResourceNotFoundException":
        raise
flows = [f for page in agent.get_paginator("list_flows").paginate() for f in page["flowSummaries"] if f["name"] == FLOW]
for f in flows:
    for a in agent.list_flow_aliases(flowIdentifier=f["id"])["flowAliasSummaries"]:
        if not a["id"].startswith("TSTALIAS"):                         # the built-in test alias goes with the flow
            agent.delete_flow_alias(flowIdentifier=f["id"], aliasIdentifier=a["id"])
    for v in agent.list_flow_versions(flowIdentifier=f["id"])["flowVersionSummaries"]:
        if v["version"] != "DRAFT":
            agent.delete_flow_version(flowIdentifier=f["id"], flowVersion=v["version"])
    agent.delete_flow(flowIdentifier=f["id"])
    print("  deleted flow", FLOW)
OURS = {f"{P}-{short}" for short in ["triage", "billing", "technical", "general", "clarifier"]}
for page in agent.get_paginator("list_prompts").paginate():
    for p in page["promptSummaries"]:
        if p["name"] in OURS:
            agent.delete_prompt(promptIdentifier=p["id"])              # deletes the draft and every version
            print("  deleted prompt", p["name"])
m.delete_role(FLOW_ROLE); m.delete_role(FN_ROLE)

# %% tags=["cleanup"]
LOG_GROUP = f"/aws/lambda/{FN}"
def our_log_groups() -> list[str]:
    return [g["logGroupName"] for g in logs.describe_log_groups(logGroupNamePrefix=f"/aws/lambda/{P}")["logGroups"]]

for attempt in range(3):          # late log delivery can recreate the group after the function is gone
    time.sleep(5)
    for g in our_log_groups():
        logs.delete_log_group(logGroupName=g)
        print("  deleted log group", g)
    time.sleep(5)
    if not our_log_groups():
        break

# %% tags=["cleanup"]
left = [f["name"] for page in agent.get_paginator("list_flows").paginate() for f in page["flowSummaries"]
        if f["name"].startswith(P)]
left += [p["name"] for page in agent.get_paginator("list_prompts").paginate() for p in page["promptSummaries"]
         if p["name"].startswith(P)]
left += [f["FunctionName"] for page in lam.get_paginator("list_functions").paginate() for f in page["Functions"]
         if f["FunctionName"].startswith(P)]
left += [r["RoleName"] for page in iam.get_paginator("list_roles").paginate() for r in page["Roles"]
         if r["RoleName"].startswith(P)]
left += our_log_groups()
print("✓ nothing left" if not left else f"✗ still there: {left}")
