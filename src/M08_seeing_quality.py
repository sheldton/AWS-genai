# %% [markdown]
# # Seeing quality, not just uptime — observability for a GenAI application
#
# **Course** MLAIDP · **Module 8 — Monitoring and Observability for Generative AI**
# **When:** after slide 16 (*Implementing your foundation model monitoring strategy*), before slide 17 (*Business impact
# and value measurement*) — the end of lesson 1; §2 comes back in lesson 3 (slide 33, preview)
# · **Kernel** MLAIDP (Python 3.12) · your AWS credentials (`AWS_PROFILE` or the default chain), us-east-1
# · **Runtime** about 2–5 minutes of cell time (130–310 s measured; mostly two waits: log records ≈ 15 s, up to
# 90 s when a record is lost, and the alarm 40–180 s) · **Cost per run** about $0.02
#
# **What you'll learn**
# - Turn on **model invocation logging** in code — an *account-level* setting: save what was there, never overwrite
#   someone else's, restore it exactly — with a least-privilege role and a **data protection policy** that masks PII.
# - Tag every call with **`requestMetadata`** (tenant, feature, prompt version) and answer *who spends the tokens* with
#   **Logs Insights queries run from code**: cost per tenant, p95 latency and tokens/s per feature, truncated answers,
#   the most expensive prompts — and reconcile the log with what the client measured.
# - Make quality a **metric**: grounding scores from a Guardrails contextual grounding check, published as **EMF**
#   log lines, a **metric-math alarm** with a minimum-sample guard, and a three-layer **dashboard defined as JSON**.
# - Capture the **OpenTelemetry spans of a Strands agent** in-process (span tree, tokens and time per step) and check
#   them against the invocation log.
#
# > **Resources:** log groups `/aws/bedrock/mlaidp2-m08-invocations` and `/mlaidp2-m08/app`, IAM role
# > `mlaidp2-m08-bedrock-logging`, guardrail `mlaidp2-m08-grounding`, alarm `mlaidp2-m08-ungrounded-rate`, dashboard
# > `mlaidp2-m08-genai`, and the account's **model invocation logging configuration** (only if none exists).
# > §0 Prep takes about 30 s. The cleanup section deletes everything and restores the logging configuration.

# %% [markdown]
# ## §0 Setup

# %%
import sys, json, re, time, calendar
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "../common")
import mlaidp as m
from botocore.exceptions import ClientError

RID = m.setup_banner("M08")
bedrock, runtime = m.client("bedrock"), m.client("bedrock-runtime")
logs, cw, iam = m.client("logs"), m.client("cloudwatch"), m.client("iam")
ACCOUNT, REGION = m.account(), m.REGION

OUR_LOG = "/aws/bedrock/mlaidp2-m08-invocations"          # the log group this notebook creates (and may delete)
INVOCATION_LOG = OUR_LOG               # the group the queries read; §0 Prep points it elsewhere if logging is foreign
APP_LOG, APP_STREAM = "/mlaidp2-m08/app", "quality"          # where the application writes its EMF lines
LOG_ROLE, GUARDRAIL = "mlaidp2-m08-bedrock-logging", "mlaidp2-m08-grounding"
ALARM, DASHBOARD, NAMESPACE = "mlaidp2-m08-ungrounded-rate", "mlaidp2-m08-genai", "MLAIDP2/M08"
SAVED = Path("../notebooks/.m08_logging_before.json")       # the account's logging config before this notebook
POLICY = (m.ASSETS / "m03_docs" / "home_policy_wording_HP-2026.txt").read_text()
MANUAL = (m.ASSETS / "m03_docs" / "claims_handling_manual_agents.txt").read_text()

# %% [markdown]
# ## §0 Prep — model invocation logging is an account-level switch
# There is **one** logging configuration per account and Region. Turning it on logs *every* Bedrock call — other teams'
# prompts included; turning it off breaks someone else's audit trail. So: read it first, keep a copy of what was there
# (once — a re-run must not save our own config as "the original"), and only write it if nobody else owns it.

# %%
def logging_owner() -> tuple[str, dict | None]:
    cfg = bedrock.get_model_invocation_logging_configuration().get("loggingConfig")
    if not cfg:
        return "none", None
    group = (cfg.get("cloudWatchConfig") or {}).get("logGroupName")
    return ("ours" if group == OUR_LOG else "foreign"), cfg

owner, current = logging_owner()
if not SAVED.exists():                       # first run only: remember the state we must restore in cleanup
    SAVED.write_text(json.dumps({"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "owner": owner, "config": current}))
print("logging configuration now:", owner, "·", m.masked(json.dumps(current)) if current else "(none)")
print("saved original state     :", json.loads(SAVED.read_text())["owner"], "→ restored by the cleanup section")

# %% [markdown]
# The destination: a log group with a **retention** (prompts are data — keep them as long as you need, not forever), a
# **data protection policy** that masks e-mail addresses for everyone without `logs:Unmask`, and a role Bedrock assumes
# that may write only to that one stream. The trust policy pins the caller to *this* account (confused-deputy guard).

# %%
try:                                    # always OUR_LOG: never change retention or policies of a group we didn't create
    logs.create_log_group(logGroupName=OUR_LOG, tags=m.TAGS)
except logs.exceptions.ResourceAlreadyExistsException:
    pass
logs.put_retention_policy(logGroupName=OUR_LOG, retentionInDays=3)
EMAIL = "arn:aws:dataprotection::aws:data-identifier/EmailAddress"
logs.put_data_protection_policy(logGroupIdentifier=OUR_LOG, policyDocument=json.dumps({
    "Name": "mask-pii", "Version": "2021-06-01", "Statement": [
        {"Sid": "audit", "DataIdentifier": [EMAIL], "Operation": {"Audit": {"FindingsDestination": {}}}},
        {"Sid": "mask", "DataIdentifier": [EMAIL], "Operation": {"Deidentify": {"MaskConfig": {}}}}]}))

TRUST = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "bedrock.amazonaws.com"},
         "Action": "sts:AssumeRole", "Condition": {"StringEquals": {"aws:SourceAccount": ACCOUNT},
                                                   "ArnLike": {"aws:SourceArn": f"arn:aws:bedrock:{REGION}:{ACCOUNT}:*"}}}]}
WRITE = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
         "Resource": f"arn:aws:logs:{REGION}:{ACCOUNT}:log-group:{OUR_LOG}:log-stream:aws/bedrock/modelinvocations"}]}
try:
    role_arn = iam.create_role(RoleName=LOG_ROLE, AssumeRolePolicyDocument=json.dumps(TRUST),
                               Tags=[{"Key": k, "Value": v} for k, v in m.TAGS.items()])["Role"]["Arn"]
except iam.exceptions.EntityAlreadyExistsException:
    role_arn = iam.get_role(RoleName=LOG_ROLE)["Role"]["Arn"]
iam.put_role_policy(RoleName=LOG_ROLE, PolicyName="write-invocation-log", PolicyDocument=json.dumps(WRITE))
print(m.masked(role_arn))

# %% [markdown]
# Now the switch itself. A new role needs a few seconds before Bedrock may assume it — the call fails with a
# `ValidationException` until then, so retry. If someone else's configuration is in place, we **leave it alone** and read
# their CloudWatch log group instead (our queries filter on our own `requestMetadata`, so their records do not mix in).

# %%
LOGGING_CONFIG = {"cloudWatchConfig": {"logGroupName": OUR_LOG, "roleArn": role_arn},
                  "textDataDeliveryEnabled": True, "imageDataDeliveryEnabled": False,
                  "embeddingDataDeliveryEnabled": False, "videoDataDeliveryEnabled": False}
owner, current = logging_owner()                        # read again right before writing: others share this account
if owner == "none":
    t0 = time.time()
    m.retry(lambda: bedrock.put_model_invocation_logging_configuration(loggingConfig=LOGGING_CONFIG),
            when="ValidationException|AccessDenied", tries=20, delay=3)
    print(f"✓ model invocation logging ON → {OUR_LOG} (role usable after {time.time() - t0:.0f} s)")
elif owner == "ours":
    print(f"✓ adopted: logging already goes to {OUR_LOG}")
else:
    foreign = (current.get("cloudWatchConfig") or {}).get("logGroupName")
    if not foreign:
        raise RuntimeError("Another configuration logs to S3 only - not overwriting it. Ask its owner, then re-run.")
    INVOCATION_LOG = foreign                            # read-only from here on: we only run queries against it
    m.note(f"Someone else's logging configuration is active - left unchanged; reading <b>{foreign}</b>.", "#b45309")

# %% [markdown]
# Don't assume the pipeline works — **prove it**. A small helper runs a **Logs Insights** query from code
# (`start_query` → poll `get_query_results`); a canary call then has to appear in the log before the demo starts.
# A fresh configuration takes a few seconds to reach every endpoint: calls in that window are *not* logged.

# %%
def insights(query: str, *, group: str | None = None, minutes: int = 60) -> list[dict]:
    qid = logs.start_query(logGroupNames=[group or INVOCATION_LOG], startTime=int(time.time() - minutes * 60),
                           endTime=int(time.time()) + 60, queryString=query)["queryId"]
    while (r := logs.get_query_results(queryId=qid))["status"] in ("Scheduled", "Running"):
        time.sleep(1)
    return [{f["field"]: f["value"] for f in row if f["field"] != "@ptr"} for row in r["results"]]

def wait_for_records(where: str, expected: int, timeout: int = 240, group: str | None = None) -> float:
    t0 = time.time()
    while True:
        rows = insights(f"filter {where} | stats count(*) as n", group=group)
        seen = int(rows[0]["n"]) if rows else 0
        if seen >= expected or time.time() - t0 > timeout:
            print(f"{seen}/{expected} records visible after {time.time() - t0:.0f} s")
            return seen
        time.sleep(5)

# %%
CANARY, t0, sent = f"{RID}-canary-{time.strftime('%H%M%S')}", time.time(), 0
while True:
    runtime.converse(modelId=m.NOVA_MICRO, messages=[{"role": "user", "content": [{"text": "ping"}]}],
                     inferenceConfig={"maxTokens": 1}, requestMetadata={"run": CANARY})
    sent += 1
    time.sleep(5)
    rows = insights(f'filter requestMetadata.run = "{CANARY}" | stats count(*) as n')
    if rows and int(rows[0]["n"]) > 0:
        break
    if time.time() - t0 > 240:
        raise TimeoutError("no invocation record after 4 minutes - check the role and the logging configuration")
print(f"✓ invocation records arrive: first canary visible after {time.time() - t0:.0f} s ({sent} canary calls sent)")

# %% [markdown]
# Two more pieces for §2: a guardrail with only a **contextual grounding check** in *detect* mode (`action: NONE` —
# score everything, block nothing yet: slide 33's advice, preview), and the application's own log group for EMF lines.

# %%
GROUNDING_THRESHOLD = 0.75
gr = next((g for g in bedrock.list_guardrails(maxResults=100)["guardrails"] if g["name"] == GUARDRAIL), None)
guardrail_id = gr["id"] if gr else bedrock.create_guardrail(
    name=GUARDRAIL, description="MLAIDP M08: contextual grounding check, detect mode",
    blockedInputMessaging="Blocked.", blockedOutputsMessaging="I could not verify this answer against the policy.",
    contextualGroundingPolicyConfig={"filtersConfig": [
        {"type": "GROUNDING", "threshold": GROUNDING_THRESHOLD, "action": "NONE", "enabled": True},
        {"type": "RELEVANCE", "threshold": 0.5, "action": "NONE", "enabled": True}]},
    tags=[{"key": k, "value": v} for k, v in m.TAGS.items()])["guardrailId"]
m.wait_for(lambda: bedrock.get_guardrail(guardrailIdentifier=guardrail_id)["status"],
           until=lambda s: s == "READY" or s, every=2, label=f"guardrail {guardrail_id}")
for create in (lambda: logs.create_log_group(logGroupName=APP_LOG, tags=m.TAGS),
               lambda: logs.create_log_stream(logGroupName=APP_LOG, logStreamName=APP_STREAM)):
    try:
        create()
    except logs.exceptions.ResourceAlreadyExistsException:
        pass
logs.put_retention_policy(logGroupName=APP_LOG, retentionInDays=3)
print("app log group", APP_LOG, "ready")

# %% [markdown]
# ## 1 · Who spends the tokens?
# AnyCompany's assistant API serves three broker partners (**tenants**) with three **features**. The AWS/Bedrock
# metrics show one line per *model* — for the whole account. Attribution needs context the model call does not have, so
# the application attaches it: `requestMetadata` is written into the invocation log record (it is not sent to the model).
#
# | Feature | Model | What it sends |
# |---|---|---|
# | `faq` | Nova Micro | a short question |
# | `email-reply` | Nova Lite | a customer e-mail, `maxTokens` 80 |
# | `claims-assist` | Nova Lite | the full policy wording + claims manual (≈ 2,000 tokens) with every question |

# %%
FEATURES = {
    "faq": dict(model=m.NOVA_MICRO, max_tokens=120,
                system="You are AnyCompany Insurance's FAQ bot. Answer in one sentence."),
    "email-reply": dict(model=m.NOVA_LITE, max_tokens=80,
                        system="Write a complete, friendly reply to the customer's e-mail."),
    "claims-assist": dict(model=m.NOVA_LITE, max_tokens=150,
                          system="Answer the agent's question in two sentences from these documents.\n\n"
                                 + POLICY + "\n\n" + MANUAL),
}
FAQS = ["Is hail damage to my roof covered?", "Do you cover a stolen bicycle?", "Is my garden fence covered in a storm?",
        "Does home insurance cover a broken window?"]
EMAILS = ["Hi, I am {n} (reach me at {n}@example.com). My premium went up 20 percent this year, why? "
          "Also, can I pay monthly?",
          "Dear AnyCompany, this is {n}, {n}@example.com. A tree fell on our shed in last night's storm. What do I do now?"]
CLAIMS = ["Claim is at R4 for 35 days. What should I do?", "Customer wants to bin wet carpets today. What do I tell them?",
          "Burst pipe vs river water - which one is covered?",
          "Customer asks for 800 USD for drying equipment. Can I approve it?"]
MIX = {"northwind": {"faq": 8, "email-reply": 4}, "contoso": {"faq": 4, "email-reply": 4, "claims-assist": 2},
       "fabrikam": {"faq": 2, "claims-assist": 8}}

def converse_tagged(feature: str, prompt: str, metadata: dict) -> dict:
    f = FEATURES[feature]
    r = runtime.converse(modelId=f["model"], system=[{"text": f["system"]}],
                         messages=[{"role": "user", "content": [{"text": prompt}]}],
                         inferenceConfig={"maxTokens": f["max_tokens"], "temperature": 0.3},
                         requestMetadata={"feature": feature, **metadata})     # string values, ≤ 16 pairs
    return {"feature": feature, "tenant": metadata.get("tenant"), "requestId": r["ResponseMetadata"]["RequestId"],
            "usage": r["usage"], "stop": r["stopReason"], "ms": r["metrics"]["latencyMs"], "text": m.text_of(r)}

# %%
RUN = f"{RID}-{time.strftime('%H%M%S')}"          # one tag per traffic run: re-runs never double-count
names = ["dana", "li", "omar", "sofia"]
jobs = []
for tenant, features in MIX.items():
    for feature, n in features.items():
        texts = ({"faq": FAQS, "claims-assist": CLAIMS}.get(feature)
                 or [e.format(n=names[i % 4]) for i, e in enumerate(EMAILS * 2)])
        jobs += [(feature, texts[i % len(texts)], {"tenant": tenant, "run": RUN}) for i in range(n)]
t0 = time.time()
with ThreadPoolExecutor(4) as pool:
    CALLS = list(pool.map(lambda j: converse_tagged(*j), jobs))
print(f"{len(CALLS)} calls in {time.time() - t0:.0f} s, run tag {RUN}")
print("input tokens per tenant (client side):",
      {t: sum(c["usage"]["inputTokens"] for c in CALLS if c["tenant"] == t) for t in MIX})

# %%
# the records of this run usually arrive 10-20 s after the calls (talk through slide 6 meanwhile); a record that is
# still missing after 90 s did not arrive at all in testing - the next cell names it
seen = wait_for_records(f'requestMetadata.run = "{RUN}"', len(CALLS), timeout=90)

# %% [markdown]
# ### Is the log complete?
# Before any number from the log is believed, reconcile it with the client **by request id** (Bedrock's
# `x-amzn-RequestId` is the record's `requestId`): which calls have no record, and do the token counts agree?

# %%
logged = {r["requestId"]: r for r in insights(f'''filter requestMetadata.run = "{RUN}"
| fields requestId, input.inputTokenCount as tokens_in | limit 1000''')}
missing = [c for c in CALLS if c["requestId"] not in logged]
same_tokens = all(int(logged[c["requestId"]]["tokens_in"]) == c["usage"]["inputTokens"]
                  for c in CALLS if c["requestId"] in logged)
print(f"✓ every call has its invocation record ({len(logged)}/{len(CALLS)})" if not missing else
      f"⚠ {len(missing)} of {len(CALLS)} calls have no invocation record yet: " +
      ", ".join(f"{c['requestId'][:8]}… ({c['tenant']}/{c['feature']})" for c in missing))
print("✓ token counts in the log equal the client's, call by call" if same_tokens
      else "✗ token counts differ between log and client")

# %% [markdown]
# ### Cost per tenant
# Logs Insights sums tokens per tenant **and model**; the price join happens in Python (prices differ per model, and a
# query should not hard-code them).

# %%
Q_TENANT = f'''filter requestMetadata.run = "{RUN}"
| stats count(*) as calls, sum(input.inputTokenCount) as tokens_in, sum(output.outputTokenCount) as tokens_out
  by requestMetadata.tenant as tenant, modelId'''
print(Q_TENANT)
rows = insights(Q_TENANT)
cost = {}
for r in rows:
    c = cost.setdefault(r["tenant"], {"tenant": r["tenant"], "calls": 0, "tokens in": 0, "tokens out": 0, "USD": 0.0})
    c["calls"] += int(r["calls"]); c["tokens in"] += int(r["tokens_in"]); c["tokens out"] += int(r["tokens_out"])
    c["USD"] += m.usd(r["modelId"], {"inputTokens": int(r["tokens_in"]), "outputTokens": int(r["tokens_out"])})
total = sum(c["USD"] for c in cost.values())
TENANTS = sorted(cost.values(), key=lambda c: -c["USD"])
for c in TENANTS:
    c["share"], c["USD / 1k calls"] = c["USD"] / total, 1000 * c["USD"] / c["calls"]
m.table(TENANTS, fmt={"USD": "${:.5f}", "share": "{:.0%}", "USD / 1k calls": "${:.3f}"})
print(f"top tenant: {TENANTS[0]['tenant']} - {TENANTS[0]['calls']} of {len(CALLS)} calls, "
      f"{TENANTS[0]['share']:.0%} of the cost")

# %% [markdown]
# ### Latency per feature — and why a long answer does not mean a slow model
# The Converse record carries the server-side `latencyMs`. Percentiles per feature, plus **output tokens per second**:
# the number that tells *slow model* apart from *long answer*.

# %%
Q_LATENCY = f'''filter requestMetadata.run = "{RUN}"
| stats count(*) as calls, pct(output.outputBodyJson.metrics.latencyMs, 50) as p50_ms,
        pct(output.outputBodyJson.metrics.latencyMs, 95) as p95_ms, avg(output.outputTokenCount) as avg_out,
        sum(output.outputTokenCount) * 1000 / sum(output.outputBodyJson.metrics.latencyMs) as out_tok_per_s
  by requestMetadata.feature as feature
| sort p95_ms desc'''
print(Q_LATENCY)
LAT = insights(Q_LATENCY)
m.table([{k: (float(v) if k != "feature" else v) for k, v in r.items()} for r in LAT],
        fmt={"calls": "{:.0f}", "p50_ms": "{:.0f}", "p95_ms": "{:.0f}", "avg_out": "{:.0f}", "out_tok_per_s": "{:.0f}"})

# %% [markdown]
# ### Model errors that return HTTP 200
# Slide 15's *model errors*: an answer cut off at `maxTokens` is a success for every infrastructure metric. The stop
# reason is in the log — group by it.

# %%
Q_STOP = f'''filter requestMetadata.run = "{RUN}"
| stats count(*) as calls by requestMetadata.feature as feature, output.outputBodyJson.stopReason as stop_reason'''
STOPS = insights(Q_STOP)
pivot = {}
for r in STOPS:
    pivot.setdefault(r["feature"], {"feature": r["feature"]})[r["stop_reason"]] = int(r["calls"])
rows = [{"end_turn": 0, "max_tokens": 0, **p} for p in pivot.values()]
for r in rows:
    r["truncated"] = r["max_tokens"] / (r["end_turn"] + r["max_tokens"])
m.table(rows, fmt={"truncated": "{:.0%}"})
worst = max(rows, key=lambda r: r["truncated"])
print(f"✓ found it: {worst['feature']} truncates {worst['truncated']:.0%} of its answers" if worst["truncated"] > 0
      else "no truncated answers in this run")

# %% [markdown]
# ### The most expensive prompts — and what the log shows of them
# The top prompts by input tokens, with the first characters of the user text — a short question: the tokens are the
# documents in the system prompt. The e-mail address in the
# `email-reply` prompts is masked by the data protection policy: Insights shows asterisks unless the reader has
# `logs:Unmask` and asks for `unmask(@message)`.

# %%
Q_TOP = f'''filter requestMetadata.run = "{RUN}"
| sort input.inputTokenCount desc
| limit 3
| fields requestMetadata.tenant as tenant, requestMetadata.feature as feature, input.inputTokenCount as tokens_in,
         substr(input.inputBodyJson.messages.0.content.0.text, 0, 60) as prompt_start'''
m.table(insights(Q_TOP))
masked_rows = insights(f'''filter requestMetadata.run = "{RUN}" and requestMetadata.feature = "email-reply"
| limit 2 | fields substr(input.inputBodyJson.messages.0.content.0.text, 0, 75) as prompt_as_stored''')
m.table(masked_rows)
leaked = any("@example.com" in r["prompt_as_stored"] for r in masked_rows)
print("⚠ no email-reply record has arrived yet - masking not checked; run the cell again in a minute" if not masked_rows
      else "✗ an e-mail address is visible in the log" if leaked else "✓ e-mail addresses are masked in the log")

# %% [markdown]
# > **Think about it:** `fabrikam` makes about a third of the calls — compare that with its share of the cost. What do
# > you change first — the tenant's price, the feature's prompt (retrieval instead of the whole manual, Module 3), or a
# > cache (Module 7)?
#
# <details><summary>More questions</summary>
#
# - Why is `tenant` fine as `requestMetadata` but a bad CloudWatch *metric dimension* (slide 7)?
# - The invocation log holds every prompt of every team in this account. Who should be able to read it?
# </details>

# %% [markdown]
# ## 2 · Is it right? Quality as a metric
# The refund desk's bot answers policy questions. Version **v1** sends the policy wording with the question. Version
# **v2** is a release in which the retrieval step silently broke: same prompt, no documents. Both return HTTP 200 —
# would any infrastructure dashboard turn red?
#
# Each answer is scored by the guardrail's **contextual grounding check** (`ApplyGuardrail` with the policy as
# `grounding_source`, the question as `query`). Because the questions have known answers, a deterministic **fact check**
# runs next to it — that is how we find out how good the detector is.

# %%
QUESTIONS = [  # (question, regex that a correct answer must match)
    ("A pipe burst in my kitchen. What is my deductible?", r"\b750\b"),
    ("My house is unlivable after a fire. How many hotel nights do you pay for, and up to what price per night?",
     r"\b30\b.*\b150\b|\b150\b.*\b30\b"),
    ("Burglars broke in and took jewelry worth 4,000 USD. How much of the jewelry is covered?", r"1,?500"),
    ("Can I throw away my soaked carpet before the adjuster comes?", r"(?i)\b(no|not|keep|retain|hold on to)\b"),
    ("How many days do I have to report storm damage?", r"\b30\b"),
    ("River water flooded my basement. Am I covered?", r"(?i)add-on"),
]
VERSIONS = {
    "v1": "You are AnyCompany Insurance's refund desk. Answer in at most two sentences, only from this policy "
          "wording:\n\n" + POLICY,
    "v2": "You are AnyCompany Insurance's refund desk. Answer in at most two sentences. Be specific: give amounts, "
          "limits and deadlines.",
}

# %%
def grounding(answer: str, question: str) -> dict:
    r = runtime.apply_guardrail(guardrailIdentifier=guardrail_id, guardrailVersion="DRAFT", source="OUTPUT",
                                content=[
        {"text": {"text": POLICY, "qualifiers": ["grounding_source"]}},
        {"text": {"text": question, "qualifiers": ["query"]}},
        {"text": {"text": answer, "qualifiers": ["guard_content"]}}])
    f = {x["type"]: x for a in r["assessments"] for x in a.get("contextualGroundingPolicy", {}).get("filters", [])}
    return {"grounding": f["GROUNDING"]["score"], "relevance": f["RELEVANCE"]["score"],
            "flagged": f["GROUNDING"]["detected"]}

def answer_and_score(job) -> dict:
    version, (question, fact) = job
    r = runtime.converse(modelId=m.NOVA_LITE, system=[{"text": VERSIONS[version]}],
                         messages=[{"role": "user", "content": [{"text": question}]}],
                         inferenceConfig={"maxTokens": 150, "temperature": 0},
                         requestMetadata={"feature": "refund-bot", "prompt_version": version, "run": RUN})
    text = m.text_of(r)
    return {"version": version, "question": question[:38], "answer": text[:90],
            "fact ok": bool(re.search(fact, text, re.S)),
            "requestId": r["ResponseMetadata"]["RequestId"], **grounding(text, question)}

with ThreadPoolExecutor(4) as pool:
    ANSWERS = list(pool.map(answer_and_score, [(v, q) for v in VERSIONS for q in QUESTIONS]))
m.table([{k: v for k, v in a.items() if k != "requestId"} for a in ANSWERS],
        fmt={"grounding": "{:.2f}", "relevance": "{:.2f}"})

# %% [markdown]
# How good is the detector? Compare the grounding flag with the fact check: a wrong answer the check did **not** flag is
# a miss; a correct answer it flagged is a false alarm.

# %%
summary = []
for v in VERSIONS:
    rs = [a for a in ANSWERS if a["version"] == v]
    summary.append({"version": v, "answers": len(rs), "facts correct": sum(a["fact ok"] for a in rs),
                    "mean grounding": sum(a["grounding"] for a in rs) / len(rs), "flagged": sum(a["flagged"] for a in rs)})
m.table(summary, fmt={"mean grounding": "{:.2f}"})
wrong = [a for a in ANSWERS if not a["fact ok"]]
missed = [a for a in wrong if not a["flagged"]]
false_alarms = [a for a in ANSWERS if a["fact ok"] and a["flagged"]]
print(f"wrong answers: {len(wrong)} · flagged by the grounding check: {len(wrong) - len(missed)} · missed: {len(missed)}"
      f" · false alarms: {len(false_alarms)}")
for a in missed:
    print(f"  missed ({a['grounding']:.2f}): {a['question']} → {a['answer']}")
v1, v2 = summary
print("✓ the broken release is visible: v2 mean grounding", f"{v2['mean grounding']:.2f} < v1 {v1['mean grounding']:.2f}"
      if v2["mean grounding"] < v1["mean grounding"] else "✗ v2 does not score lower than v1")

# %% [markdown]
# ### Publish the scores — Embedded Metric Format
# One **EMF** log line per answer: CloudWatch extracts the metrics (no `PutMetricData` call), and the line itself stays
# in the log with the `requestId` — the key into the invocation log record of that exact call. Two dimension sets:
# per feature (for the alarm) and per feature + prompt version (for the diagnosis).

# %%
def emf_line(a: dict) -> dict:
    ts = int(time.time() * 1000)
    return {"timestamp": ts, "message": json.dumps({
        "_aws": {"Timestamp": ts, "CloudWatchMetrics": [{"Namespace": NAMESPACE,
                 "Dimensions": [["Feature"], ["Feature", "PromptVersion"]],
                 "Metrics": [{"Name": "GroundingScore", "Unit": "None"}, {"Name": "Ungrounded", "Unit": "Count"},
                             {"Name": "Answers", "Unit": "Count"}]}]},
        "Feature": "refund-bot", "PromptVersion": a["version"],
        "GroundingScore": round(a["grounding"], 3), "Ungrounded": int(a["grounding"] < GROUNDING_THRESHOLD), "Answers": 1,
        "requestId": a["requestId"], "question": a["question"], "run": RUN})}   # properties: in the log, not metrics

print(json.dumps(json.loads(emf_line(ANSWERS[-1])["message"]), indent=1)[:900])

# %% [markdown]
# ### An alarm on a *rate*, with a minimum sample
# "Average grounding < 0.75" would page on one bad answer at 3 a.m. The alarm uses **metric math**: the share of
# ungrounded answers per minute, but only when there are at least 5 answers in that minute (`IF`) — slide 12's alert
# fatigue, handled in the expression. The alarm is re-created on every run so it starts from a clean state.

# %%
def stat(metric_id: str, name: str) -> dict:
    return {"Id": metric_id, "ReturnData": False, "MetricStat": {"Period": 60, "Stat": "Sum", "Metric": {
        "Namespace": NAMESPACE, "MetricName": name, "Dimensions": [{"Name": "Feature", "Value": "refund-bot"}]}}}

ALARM_METRICS = [stat("ungrounded", "Ungrounded"), stat("answers", "Answers"),
                 {"Id": "rate", "Label": "ungrounded answers (%)", "ReturnData": True,
                  "Expression": "IF(answers >= 5, 100 * ungrounded / answers, 0)"}]
cw.delete_alarms(AlarmNames=[ALARM])
cw.put_metric_alarm(AlarmName=ALARM, AlarmDescription="refund-bot: more than 20 % of answers not grounded in the policy",
                    Metrics=ALARM_METRICS, ComparisonOperator="GreaterThanThreshold", Threshold=20,
                    EvaluationPeriods=1, DatapointsToAlarm=1, TreatMissingData="notBreaching",
                    Tags=[{"Key": k, "Value": v} for k, v in m.TAGS.items()])
print(json.dumps(ALARM_METRICS[2]), "> 20 for 1 of 1 minutes")
print("state:", cw.describe_alarms(AlarmNames=[ALARM])["MetricAlarms"][0]["StateValue"])

# %% [markdown]
# Write the twelve EMF lines and wait for the alarm (up to 3 minutes). An alarm evaluates the **latest datapoint in a
# look-back window of several minutes** — after a re-run it may still be in ALARM on the previous run's datapoint, so
# the wait accepts only an evaluation of *this* run's minute (and resets a stale ALARM once with `SetAlarmState`). If
# CloudWatch is slow, the cell prints ⚠ after 3 minutes and the demo carries on — nothing below needs the alarm.

# %%
t0 = time.time()
EVENTS = [emf_line(a) for a in ANSWERS]
this_minute = EVENTS[0]["timestamp"] // 60_000 * 60_000
logs.put_log_events(logGroupName=APP_LOG, logStreamName=APP_STREAM, logEvents=EVENTS)
reset = []

def alarm_on_this_run() -> str:
    a = cw.describe_alarms(AlarmNames=[ALARM])["MetricAlarms"][0]
    points = json.loads(a.get("StateReasonData") or "{}").get("evaluatedDatapoints", [])
    fresh = any(calendar.timegm(time.strptime(p["timestamp"][:19], "%Y-%m-%dT%H:%M:%S")) >= this_minute / 1000
                for p in points)
    if a["StateValue"] == "ALARM" and not fresh:
        if not reset:
            cw.set_alarm_state(AlarmName=ALARM, StateValue="OK", StateReason="re-run: evaluate the new datapoint")
            reset.append(True); print("  alarm was in ALARM on an earlier run's datapoint - reset to OK once")
        return "ALARM on an older datapoint"
    return a["StateValue"]

try:
    m.wait_for(alarm_on_this_run, until=lambda s: s == "ALARM" or s, timeout=180, every=5, label="alarm")
    print(cw.describe_alarms(AlarmNames=[ALARM])["MetricAlarms"][0]["StateReason"])
    print(f"✓ quality alarm fired on this run's scores, {time.time() - t0:.0f} s after they were written")
except TimeoutError:                     # a slow evaluation must not stop the demo: the logs below don't need the alarm
    print(f"⚠ the alarm has not evaluated this run's minute after {time.time() - t0:.0f} s - CloudWatch is slow today. "
          "Carry on; check the alarm's History in a few minutes (Part 2 step 5).")

# %% [markdown]
# Metrics page someone; **logs** tell them why. The EMF lines are ordinary log events, so the on-call engineer can list
# the ungrounded answers — and follow `requestId` into the invocation log to see the exact prompt the model received.

# %%
Q_UNGROUNDED = f'''filter run = "{RUN}" and Ungrounded = 1
| stats count(*) as ungrounded, avg(GroundingScore) as mean_score by PromptVersion'''
wait_for_records(f'run = "{RUN}"', len(ANSWERS), group=APP_LOG)
m.table(insights(Q_UNGROUNDED, group=APP_LOG))
ungrounded = sorted((a for a in ANSWERS if a["grounding"] < GROUNDING_THRESHOLD), key=lambda a: a["grounding"])
ids = json.dumps([a["requestId"] for a in ungrounded])     # every ungrounded answer: any one record is enough
rec = insights(f'''filter requestId in {ids} | fields requestId, requestMetadata.prompt_version as version,
  strlen(input.inputBodyJson.system.0.text) as system_prompt_chars, input.inputTokenCount as tokens_in | limit 1''')
print("invocation record of one ungrounded answer:", rec)
print("✓ root cause visible: the v2 call carried no policy text" if rec and int(rec[0]["system_prompt_chars"]) < 500
      else "⚠ no invocation record of an ungrounded answer yet (log delivery lags) - run the cell again in a minute")

# %% [markdown]
# > **Think about it:** The grounding check missed a wrong answer (if it did in this run — read the "missed" line).
# > What would you add before you let this check **block** answers instead of only scoring them?
#
# <details><summary>More questions</summary>
#
# - Why alarm on the *rate* with a minimum sample, not on the minimum score?
# - If the alarm said "v2 is broken" within a minute or two, how would you roll back automatically (slide 9: AppConfig,
#   Module 1)?
# </details>

# %% [markdown]
# ## 3 · One dashboard, three layers — defined as code
# Slide 10's layers: an **executive** row (is quality OK? the alarm), an **operational** row (cost per tenant from the
# invocation log, grounding per prompt version), a **diagnostic** row (latency, time to first token, throttles per model
# from AWS/Bedrock). The body is JSON, so it lives in version control and is deployed like code.

# %%
def metric_widget(x, y, w, title, metrics, **props):
    return {"type": "metric", "x": x, "y": y, "width": w, "height": 6, "properties": {
        "title": title, "metrics": metrics, "region": REGION, "period": 60, "view": "timeSeries", **props}}

RATE = [[{"expression": "IF(a >= 5, 100 * u / a, 0)", "label": "ungrounded answers (%)", "id": "rate"}],
        [NAMESPACE, "Ungrounded", "Feature", "refund-bot", {"id": "u", "stat": "Sum", "visible": False}],
        [NAMESPACE, "Answers", "Feature", "refund-bot", {"id": "a", "stat": "Sum", "visible": False}]]
BY_MODEL = lambda name, stat: [["AWS/Bedrock", name, "ModelId", mid, {"stat": stat, "label": m.model_name(mid)}]
                               for mid in (m.NOVA_MICRO, m.NOVA_LITE, m.NOVA_PRO)]
BODY = {"widgets": [
    {"type": "text", "x": 0, "y": 0, "width": 24, "height": 1, "properties": {
        "markdown": "## AnyCompany assistant - is it **right**, is it **affordable**, is it **fast**?"}},
    metric_widget(0, 1, 12, "Are refund answers grounded? (alarm > 20 %)", RATE,
                  annotations={"horizontal": [{"value": 20, "label": "alarm"}]}, yAxis={"left": {"min": 0, "max": 100}}),
    {"type": "alarm", "x": 12, "y": 1, "width": 12, "height": 6, "properties": {"title": "Quality alarms",
     "alarms": [f"arn:aws:cloudwatch:{REGION}:{ACCOUNT}:alarm:{ALARM}"]}},
    {"type": "log", "x": 0, "y": 7, "width": 12, "height": 6, "properties": {
        "title": "Who spends the tokens? (invocation log)", "region": REGION, "view": "table",
        "query": f"SOURCE '{INVOCATION_LOG}' | filter requestMetadata.tenant like /./ | stats count(*) as calls, "
                 "sum(input.inputTokenCount) as tokens_in by requestMetadata.tenant, modelId | sort tokens_in desc"}},
    metric_widget(12, 7, 12, "Grounding score by prompt version (avg)",
                  [[NAMESPACE, "GroundingScore", "Feature", "refund-bot", "PromptVersion", v,
                    {"stat": "Average", "label": v}] for v in VERSIONS],
                  view="bar", yAxis={"left": {"min": 0, "max": 1}}),
    metric_widget(0, 13, 8, "Latency p95 (ms) per model", BY_MODEL("InvocationLatency", "p95")),
    metric_widget(8, 13, 8, "Time to first token p95 (ms)", BY_MODEL("TimeToFirstToken", "p95")),
    metric_widget(16, 13, 8, "Throttles per model", BY_MODEL("InvocationThrottles", "Sum")),
]}
problems = cw.put_dashboard(DashboardName=DASHBOARD, DashboardBody=json.dumps(BODY)).get("DashboardValidationMessages", [])
print("✓ dashboard valid" if not problems else f"✗ {problems}", f"· {len(BODY['widgets'])} widgets")
print(f"https://{REGION}.console.aws.amazon.com/cloudwatch/home?region={REGION}#dashboards/dashboard/{DASHBOARD}")

# %% [markdown]
# Widget JSON also renders server-side with `GetMetricWidgetImage` — the picture CloudWatch would draw, for reports,
# chat-ops messages and runbooks. Here: the ungrounded rate **per prompt version** over the last 10 minutes, one bar each
# (metric math per version, `setPeriodToTimeRange` aggregates the whole window).

# %%
from IPython.display import Image, display
per_version = []
for i, v in enumerate(VERSIONS):
    dims = ["Feature", "refund-bot", "PromptVersion", v]
    per_version += [[{"expression": f"100 * u{i} / a{i}", "label": f"prompt {v}", "id": f"r{i}"}],
                    [NAMESPACE, "Ungrounded", *dims, {"id": f"u{i}", "stat": "Sum", "visible": False}],
                    [NAMESPACE, "Answers", *dims, {"id": f"a{i}", "stat": "Sum", "visible": False}]]
widget = {"metrics": per_version, "view": "bar", "setPeriodToTimeRange": True, "start": "-PT10M",
          "width": 900, "height": 300,
          "title": "Ungrounded answers (%) by prompt version - the release without retrieval is far above the alarm line",
          "yAxis": {"left": {"min": 0, "max": 100}}, "annotations": {"horizontal": [{"value": 20, "label": "alarm"}]}}
display(Image(cw.get_metric_widget_image(MetricWidget=json.dumps(widget))["MetricWidgetImage"]))

# %% [markdown]
# > **Think about it:** Which widget would you show the CFO, which one the on-call engineer — and which one would
# > you delete because nobody makes a decision with it?

# %% [markdown]
# ## 4 · Traces: what an agent did, step by step
# An agent turns one question into several model calls and tool calls. **Strands** emits **OpenTelemetry** spans
# (GenAI semantic conventions: `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, …). In production you export them
# with OTLP to CloudWatch (generative AI observability) or your own collector; here an **in-memory exporter** captures
# them in the notebook. The agent's calls also carry `requestMetadata`, so every `chat` span should have an invocation
# log record.

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

if not isinstance(trace.get_tracer_provider(), TracerProvider):      # the global provider can be set once per process
    SPANS = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(SPANS))
    trace.set_tracer_provider(provider)

from strands import Agent, tool
from strands.models import BedrockModel

@tool
def lookup_policy(section: str) -> str:
    """Return the text of one section of the home policy wording, e.g. '6' for temporary accommodation."""
    part = re.search(rf"Section {re.escape(section.strip())} - .*?(?=\nSection \d|\Z)", POLICY, re.S)
    return part.group(0) if part else "No such section."

@tool
def nights_cost(nights: int, price_per_night: float) -> float:
    """Total hotel cost the policy pays: at most 30 nights and 150 USD per night."""
    return min(nights, 30) * min(price_per_night, 150.0)

# %%
SESSION = f"{RUN}-agent"
SPANS.clear()
AGENT_TAGS = {"feature": "claims-agent", "session": SESSION, "run": RUN}      # → the invocation log record
agent = Agent(model=BedrockModel(model_id=m.NOVA_PRO, temperature=0, max_tokens=400, boto_session=m.session(),
                                 additional_args={"requestMetadata": AGENT_TAGS}),
              tools=[lookup_policy, nights_cost], callback_handler=None, trace_attributes={"session.id": SESSION},
              system_prompt="You are AnyCompany's claims assistant. Look up the policy section before you answer. "
                            "Use the tools.")
result = agent("Our house burned down. We need a hotel for 45 nights at 180 USD per night. "
               "How much will AnyCompany pay for it?")
answer = re.sub(r"<thinking>.*?</thinking>|</?response>", "", str(result), flags=re.S).strip()
print(answer[:400])
expected = 30 * 150
print(f"✓ the agent's answer contains the expected amount {expected:,}" if re.search(r"4,?500", str(result))
      else f"✗ expected {expected:,} in the answer")

# %% [markdown]
# The span tree: one `invoke_agent` span, an `execute_event_loop_cycle` per reasoning step, inside each a `chat`
# span (the model call, with its tokens) and the `execute_tool` spans. Where did the time go — model or tools?

# %%
spans = SPANS.get_finished_spans()
children = {}
for s in spans:
    children.setdefault(s.parent.span_id if s.parent else None, []).append(s)

from opentelemetry.trace import StatusCode

def show(span, depth=0):
    a, ms = span.attributes, (span.end_time - span.start_time) / 1e6
    detail = (f"in {a.get('gen_ai.usage.input_tokens', 0):>5} · out {a.get('gen_ai.usage.output_tokens', 0):>4}"
              if span.name == "chat" or span.name.startswith("invoke_agent") else a.get("gen_ai.tool.name", ""))
    if span.status.status_code == StatusCode.ERROR:
        detail += f"  ERROR {(span.status.description or '')[:40]}"
    print(f"{'   ' * depth}{span.name[:32]:<{40 - 3 * depth}} {ms:8.0f} ms   {detail}")
    for c in sorted(children.get(span.context.span_id, []), key=lambda c: c.start_time):
        show(c, depth + 1)

for root in children[None]:
    show(root)
chat = sorted((s for s in spans if s.name == "chat"), key=lambda s: s.start_time)
model_ms = sum((s.end_time - s.start_time) / 1e6 for s in chat)
tool_ms = sum((s.end_time - s.start_time) / 1e6 for s in spans if s.name.startswith("execute_tool"))
print(f"\nmodel calls: {len(chat)} · model time {model_ms:.0f} ms · tool time {tool_ms:.1f} ms")

# %% [markdown]
# ### Reconcile the trace with the invocation log
# Spans are produced by the client library; the invocation log by the service. The log also records **failed
# attempts** (`errorCode`, e.g. a `ThrottlingException` the SDK retried) — a trace shows those only as a slow `chat`
# span. Pair the successful calls, count the failed ones; when tokens disagree, the log is the one to bill from.

# %%
ok_chat = [s for s in chat if s.status.status_code != StatusCode.ERROR]
wait_for_records(f'requestMetadata.session = "{SESSION}" and output.outputTokenCount >= 0', len(ok_chat), timeout=90)
LOGGED = insights(f'''filter requestMetadata.session = "{SESSION}"
| fields @timestamp, errorCode, input.inputTokenCount as tokens_in, output.outputTokenCount as tokens_out,
         output.outputBodyJson.stopReason as stop_reason, inferenceRegion | sort @timestamp asc''')
ok_log = sorted((r for r in LOGGED if "errorCode" not in r), key=lambda r: (r["@timestamp"], int(r["tokens_in"])))
failed = [r["errorCode"] for r in LOGGED if "errorCode" in r]
# pair by input tokens (each cycle re-sends a longer conversation), then leftovers in time order
left, match = list(ok_log), {}
for i, s in enumerate(ok_chat):
    r = next((r for r in left if int(r["tokens_in"]) == s.attributes.get("gen_ai.usage.input_tokens")), None)
    if r:
        match[i] = r; left.remove(r)
for i in [i for i in range(len(ok_chat)) if i not in match][:len(left)]:
    match[i] = left.pop(0)
pairs = [{"call": i + 1, "span tokens in": s.attributes.get("gen_ai.usage.input_tokens", 0),
          "log tokens in": int(match[i]["tokens_in"]), "span tokens out": s.attributes.get("gen_ai.usage.output_tokens", 0),
          "log tokens out": int(match[i]["tokens_out"]), "stop reason": match[i]["stop_reason"],
          "served in": match[i].get("inferenceRegion", "–")} for i, s in enumerate(ok_chat) if i in match]
m.table(pairs)
print(f"✓ every successful chat span has its invocation record ({len(ok_chat)} spans, {len(ok_log)} records)"
      if len(pairs) == len(ok_chat) else f"⚠ {len(ok_chat) - len(pairs)} of {len(ok_chat)} chat spans have no invocation "
      "record yet (log delivery lags or drops records - see §1)")
print(f"failed attempts in the log: {len(failed)} {sorted(set(failed))} - retried by the SDK, invisible in the answer"
      if failed else "failed attempts in the log: none this run")
gap = sum(p["log tokens in"] + p["log tokens out"] - p["span tokens in"] - p["span tokens out"] for p in pairs)
print("✓ tokens agree call by call" if gap == 0
      else f"⚠ the spans miss {gap} tokens that the service billed - bill from the log")

# %% [markdown]
# > **Think about it:** Input tokens grow with every cycle. Why — and what does that mean for the cost of an agent
# > that needs ten steps instead of three?
#
# <details><summary>More questions</summary>
#
# - Which attribute would you add to every span so a support ticket can be traced to one agent session?
# - The invocation log already has the prompts. What do traces add that the log cannot?
# </details>

# %% [markdown]
# ## Cleanup — don't skip this
# Restores the account's model invocation logging configuration to what §0 saved (it deletes the configuration only if
# it is ours and nothing was there before), then deletes the dashboard, alarm, guardrail, both log groups and the role.
# Run it **only after Part 2** (the console part uses the log group, the dashboard and the alarm). In a fresh kernel, run
# the **§0 Setup** cell first — the cleanup needs nothing else. The custom metrics in `MLAIDP2/M08` cannot be deleted;
# they expire on their own.

# %% tags=["cleanup"]
def logging_group() -> str | None:          # self-contained: works in a fresh kernel after §0 Setup only
    cfg = bedrock.get_model_invocation_logging_configuration().get("loggingConfig")
    return (cfg.get("cloudWatchConfig") or {}).get("logGroupName", "(S3 only)") if cfg else None

group_now = logging_group()
owner = "none" if group_now is None else "ours" if group_now == OUR_LOG else "foreign"
before = json.loads(SAVED.read_text()) if SAVED.exists() else {"owner": "none", "config": None}
if owner == "ours":
    if before["config"] and before["owner"] != "ours":
        bedrock.put_model_invocation_logging_configuration(loggingConfig=before["config"])
        print("logging configuration restored to the saved original")
    else:
        bedrock.delete_model_invocation_logging_configuration()
        print("logging configuration deleted (there was none before)")
else:
    print(f"logging configuration is {owner} - not ours, left unchanged")
group_after = logging_group()
after = "none" if group_after is None else "ours" if group_after == OUR_LOG else "foreign"
restored = after == before["owner"] or owner != "ours"
print(f"✓ logging is '{after}' again (saved original: '{before['owner']}')" if restored
      else f"✗ logging is '{after}', the original was '{before['owner']}'")
SAVED.unlink(missing_ok=True)

# %% tags=["cleanup"]
if cw.list_dashboards(DashboardNamePrefix=DASHBOARD)["DashboardEntries"]:
    cw.delete_dashboards(DashboardNames=[DASHBOARD])
cw.delete_alarms(AlarmNames=[ALARM])
for g in bedrock.list_guardrails(maxResults=100)["guardrails"]:
    if g["name"] == GUARDRAIL:
        bedrock.delete_guardrail(guardrailIdentifier=g["id"])
for group in (OUR_LOG, APP_LOG):                 # never INVOCATION_LOG: it may be someone else's group
    try:
        logs.delete_log_group(logGroupName=group)
    except logs.exceptions.ResourceNotFoundException:
        pass
m.delete_role(LOG_ROLE)
left = ([d["DashboardName"] for d in cw.list_dashboards(DashboardNamePrefix="mlaidp2-m08")["DashboardEntries"]]
        + [a["AlarmName"] for a in cw.describe_alarms(AlarmNamePrefix="mlaidp2-m08")["MetricAlarms"]]
        + [g["name"] for g in bedrock.list_guardrails(maxResults=100)["guardrails"] if g["name"].startswith("mlaidp2-m08")]
        + [g["logGroupName"] for p in ("/aws/bedrock/mlaidp2-m08", "/mlaidp2-m08")
           for g in logs.describe_log_groups(logGroupNamePrefix=p)["logGroups"]]
        + [r["RoleName"] for page in iam.get_paginator("list_roles").paginate() for r in page["Roles"]
           if r["RoleName"].startswith("mlaidp2-m08")])
print("✓ nothing left" if not left else f"✗ still there: {left}")
