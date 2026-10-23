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
#
# ![M08 at a glance: tagged calls feed the invocation log (§1), grounding scores become a metric and an alarm (§2), a dashboard shows both (§3), agent spans are checked against the log (§4)](../assets/diagrams/m08_overview.svg)
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
#
# > 💡 **Concept — model invocation logging.** `PutModelInvocationLoggingConfiguration` (control plane, `bedrock`) sets
# > **one** configuration per account and Region. From then on Amazon Bedrock writes one JSON record per
# > `Converse` / `InvokeModel` call (streaming variants included): model id, request id, the caller's IAM identity,
# > token counts, the `requestMetadata` you attached, and — for the data types you switch on (text, image, embedding,
# > video) — the full request and response bodies. Destinations: a CloudWatch Logs group (written through a role
# > Bedrock assumes), an S3 bucket, or both. It is off by default, and there is no per-application switch: it is
# > everyone's calls or nobody's.
#
# > 🔑 **Key code** — `get_model_invocation_logging_configuration()` answers "who owns the switch?" (no config / ours /
# > foreign, decided by the log group name). The `if not SAVED.exists()` guard writes the "before" snapshot **once**:
# > a second run would otherwise save *our* configuration as the original and the cleanup would never turn it off.

# %%
def logging_owner() -> tuple[str, dict | None]:
    # ▶ KEY: read the account-level switch before touching it
    cfg = bedrock.get_model_invocation_logging_configuration().get("loggingConfig")
    if not cfg:
        return "none", None
    group = (cfg.get("cloudWatchConfig") or {}).get("logGroupName")
    return ("ours" if group == OUR_LOG else "foreign"), cfg

owner, current = logging_owner()
# ▶ KEY: snapshot the original state exactly once - the cleanup restores this, not "off"
if not SAVED.exists():                       # first run only: remember the state we must restore in cleanup
    SAVED.write_text(json.dumps({"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "owner": owner, "config": current}))
print("logging configuration now:", owner, "·", m.masked(json.dumps(current)) if current else "(none)")
print("saved original state     :", json.loads(SAVED.read_text())["owner"], "→ restored by the cleanup section")

# %% [markdown]
# The destination: a log group with a **retention** (prompts are data — keep them as long as you need, not forever), a
# **data protection policy** that masks e-mail addresses for everyone without `logs:Unmask`, and a role Bedrock assumes
# that may write only to that one stream. The trust policy pins the caller to *this* account (confused-deputy guard).
#
# > 💡 **Concept — CloudWatch Logs data protection.** A data protection policy on a log group names **managed data
# > identifiers** (here `EmailAddress`; there are identifiers for names, card numbers, keys, …) and what to do with a
# > match: **Audit** (report findings) and **Deidentify** (mask). Masking happens at ingestion, for every reader:
# > console, `GetLogEvents`, Logs Insights all show asterisks. Only a principal with `logs:Unmask` can see the original,
# > and only by asking for it (`unmask(@message)`). The prompts of an LLM application are user data — treat the
# > invocation log like any other store of customer data.
#
# > 🔑 **Key code**
# > - `put_retention_policy(..., retentionInDays=3)` — without it, a log group keeps every prompt forever.
# > - `put_data_protection_policy(...)` — the `audit` and `mask` statements for the `EmailAddress` identifier.
# > - The trust policy's `aws:SourceAccount` / `aws:SourceArn` conditions — only Bedrock acting **for this account** may
# >   assume the role (confused-deputy guard); the `WRITE` policy names one log stream, nothing else.

# %%
try:                                    # always OUR_LOG: never change retention or policies of a group we didn't create
    logs.create_log_group(logGroupName=OUR_LOG, tags=m.TAGS)
except logs.exceptions.ResourceAlreadyExistsException:
    pass
# ▶ KEY: prompts are data - keep them 3 days, not forever
logs.put_retention_policy(logGroupName=OUR_LOG, retentionInDays=3)
EMAIL = "arn:aws:dataprotection::aws:data-identifier/EmailAddress"
# ▶ KEY: mask e-mail addresses for every reader without logs:Unmask
logs.put_data_protection_policy(logGroupIdentifier=OUR_LOG, policyDocument=json.dumps({
    "Name": "mask-pii", "Version": "2021-06-01", "Statement": [
        {"Sid": "audit", "DataIdentifier": [EMAIL], "Operation": {"Audit": {"FindingsDestination": {}}}},
        {"Sid": "mask", "DataIdentifier": [EMAIL], "Operation": {"Deidentify": {"MaskConfig": {}}}}]}))

# ▶ KEY: confused-deputy guard - only Bedrock acting for THIS account may assume the role
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
#
# > 🔑 **Key code** — `put_model_invocation_logging_configuration(loggingConfig=...)` is the account-level write; it
# > runs **only** in the `owner == "none"` branch. `m.retry(..., when="ValidationException|AccessDenied")` absorbs IAM
# > propagation: Bedrock validates that it can assume the role at the moment of the call, and a role created seconds
# > ago is not usable yet. `textDataDeliveryEnabled: True` logs prompt and answer text; images and embeddings stay off.

# %%
LOGGING_CONFIG = {"cloudWatchConfig": {"logGroupName": OUR_LOG, "roleArn": role_arn},
                  "textDataDeliveryEnabled": True, "imageDataDeliveryEnabled": False,
                  "embeddingDataDeliveryEnabled": False, "videoDataDeliveryEnabled": False}
owner, current = logging_owner()                        # read again right before writing: others share this account
if owner == "none":
    t0 = time.time()
    # ▶ KEY: the account-level write - only when nobody owns the switch; retried until the new role is assumable
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
#
# > 💡 **Concept — Logs Insights is asynchronous.** `StartQuery` takes log group(s), a time window in epoch seconds
# > and a query string (`filter`, `fields`, `stats … by …`, `sort`, `limit`; JSON fields are addressed with dots, e.g.
# > `requestMetadata.tenant`). It returns a `queryId`; `GetQueryResults` reports `Scheduled` → `Running` →
# > `Complete` and the rows, each a list of `{field, value}` pairs — **values are strings**, so convert before you do
# > arithmetic. A query is billed by the amount of log data it scans, which is why every query here filters on our run tag
# > and covers only the last hour.
#
# > 🔑 **Key code** — `start_query(...)` then the `while … in ("Scheduled", "Running")` poll; the list comprehension
# > turns each row into a dict and drops the internal `@ptr` field. `wait_for_records` is the same query in a loop: it
# > counts records until the expected number has arrived (or the timeout says how many never did).

# %%
def insights(query: str, *, group: str | None = None, minutes: int = 60) -> list[dict]:
    # ▶ KEY: start the query - it runs asynchronously and returns only an id
    qid = logs.start_query(logGroupNames=[group or INVOCATION_LOG], startTime=int(time.time() - minutes * 60),
                           endTime=int(time.time()) + 60, queryString=query)["queryId"]
    # ▶ KEY: poll until the status leaves Scheduled/Running; every value comes back as a string
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

# %% [markdown]
# The canary: a one-token call tagged with a unique `run` value, repeated every 5 s until its record is visible.
#
# > ⏱️ This cell prints nothing until the first canary record arrives — usually 10–15 s, at most 4 minutes.

# %%
CANARY, t0, sent = f"{RID}-canary-{time.strftime('%H%M%S')}", time.time(), 0
while True:
    # ▶ KEY: a cheap tagged call (1 output token) whose record must appear in the log
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
#
# > 💡 **Concept — contextual grounding check.** A guardrail policy that compares a model answer with a **source**
# > (the reference text) and the user's **query**, and returns two scores between 0 and 1: **grounding** (is every claim
# > supported by the source?) and **relevance** (does the answer address the query?). Below the filter's `threshold`
# > the filter counts as *detected*. With `action: "BLOCK"` the guardrail would replace the answer with
# > `blockedOutputsMessaging`; with `action: "NONE"` it only reports the score — the way to measure a detector on real
# > traffic before you let it block anything.
#
# > 🔑 **Key code** — `contextualGroundingPolicyConfig.filtersConfig`: `GROUNDING` with `threshold 0.75` and
# > `RELEVANCE` with `0.5`, both `"action": "NONE"` (detect mode). The guardrail is adopted by name if it exists, and
# > the cell waits until its status is `READY`.

# %%
GROUNDING_THRESHOLD = 0.75
gr = next((g for g in bedrock.list_guardrails(maxResults=100)["guardrails"] if g["name"] == GUARDRAIL), None)
guardrail_id = gr["id"] if gr else bedrock.create_guardrail(
    name=GUARDRAIL, description="MLAIDP M08: contextual grounding check, detect mode",
    blockedInputMessaging="Blocked.", blockedOutputsMessaging="I could not verify this answer against the policy.",
    # ▶ KEY: detect mode - score every answer, block none (action NONE)
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
#
# ![The invocation logging pipeline: tagged Converse calls, the record Bedrock writes, the queries that read it](../assets/diagrams/m08_logging_pipeline.svg)
#
# > 💡 **Concept — `requestMetadata`.** An optional map of string key/value pairs (up to 16) on `Converse` /
# > `ConverseStream`. The model never sees it; Bedrock copies it into the invocation log record of that call. That
# > makes it the place for **high-cardinality context** — tenant, feature, prompt version, session, a run tag — that
# > would be ruinous as a CloudWatch metric dimension (every distinct value is a separate, billed metric). Metrics
# > answer "how much, overall"; the log with metadata answers "who, which feature, which prompt".
#
# > 🔑 **Key code** — in `converse_tagged`, `requestMetadata={"feature": feature, **metadata}` is the whole attribution
# > mechanism. The function also keeps `r["ResponseMetadata"]["RequestId"]` — the same id the log record carries as
# > `requestId`, so client and log can be joined call by call.

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
                         # ▶ KEY: attribution travels with the call into the log record - the model never sees it
                         requestMetadata={"feature": feature, **metadata})     # string values, ≤ 16 pairs
    # ▶ KEY: keep the request id - it is the join key into the invocation log
    return {"feature": feature, "tenant": metadata.get("tenant"), "requestId": r["ResponseMetadata"]["RequestId"],
            "usage": r["usage"], "stop": r["stopReason"], "ms": r["metrics"]["latencyMs"], "text": m.text_of(r)}

# %%
# ▶ KEY: one tag per traffic run - every query below filters on it, so re-runs never double-count
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

# %% [markdown]
# > ⏱️ The next cell prints nothing while it waits for the records — usually ≈ 15 s, up to 90 s when a record is lost.

# %%
# the records of this run usually arrive 10-20 s after the calls (talk through slide 6 meanwhile); a record that is
# still missing after 90 s did not arrive at all in testing - the next cell names it
seen = wait_for_records(f'requestMetadata.run = "{RUN}"', len(CALLS), timeout=90)

# %% [markdown]
# ### Is the log complete?
# Before any number from the log is believed, reconcile it with the client **by request id** (Bedrock's
# `x-amzn-RequestId` is the record's `requestId`): which calls have no record, and do the token counts agree?
#
# > 💡 **Concept — a log is evidence only after reconciliation.** Log delivery is asynchronous and best-effort: a record
# > can arrive late or, rarely, not at all. Before you bill a tenant or alarm on a log-derived number, check it against
# > an independent source — here the client's own list of request ids and the `usage` block of every response. Two
# > questions: is every call there (completeness), and do the numbers match (accuracy)?
#
# > 🔑 **Key code** — the dict comprehension keys the log rows by `requestId`; `missing` is a set difference against the
# > client's calls; `same_tokens` compares `input.inputTokenCount` with the client's `usage["inputTokens"]` call by call.

# %%
# ▶ KEY: the log keyed by request id - the same id the client got back
logged = {r["requestId"]: r for r in insights(f'''filter requestMetadata.run = "{RUN}"
| fields requestId, input.inputTokenCount as tokens_in | limit 1000''')}
# ▶ KEY: completeness - which client calls have no record
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
#
# > 🔑 **Key code** — `stats … by requestMetadata.tenant as tenant, modelId`: group by **both**, because the price
# > depends on the model. `m.usd(r["modelId"], …)` applies the price list per row, then Python adds up per tenant.

# %%
# ▶ KEY: aggregate in the log, group by tenant AND model (the price depends on the model)
Q_TENANT = f'''filter requestMetadata.run = "{RUN}"
| stats count(*) as calls, sum(input.inputTokenCount) as tokens_in, sum(output.outputTokenCount) as tokens_out
  by requestMetadata.tenant as tenant, modelId'''
print(Q_TENANT)
rows = insights(Q_TENANT)
cost = {}
for r in rows:
    c = cost.setdefault(r["tenant"], {"tenant": r["tenant"], "calls": 0, "tokens in": 0, "tokens out": 0, "USD": 0.0})
    c["calls"] += int(r["calls"]); c["tokens in"] += int(r["tokens_in"]); c["tokens out"] += int(r["tokens_out"])
    # ▶ KEY: the price join happens in code, per model
    c["USD"] += m.usd(r["modelId"], {"inputTokens": int(r["tokens_in"]), "outputTokens": int(r["tokens_out"])})
total = sum(c["USD"] for c in cost.values())
TENANTS = sorted(cost.values(), key=lambda c: -c["USD"])
for c in TENANTS:
    c["share"], c["USD / 1k calls"] = c["USD"] / total, 1000 * c["USD"] / c["calls"]
m.table(TENANTS, fmt={"USD": "${:.5f}", "share": "{:.0%}", "USD / 1k calls": "${:.3f}"})
print(f"top tenant: {TENANTS[0]['tenant']} - {TENANTS[0]['calls']} of {len(CALLS)} calls, "
      f"{TENANTS[0]['share']:.0%} of the cost")

# %% [markdown]
# > 📊 **How to read this** — one row per tenant, most expensive first. Compare **calls** with **share** (of the cost):
# > a tenant whose share of the cost is far above its share of the calls has expensive *requests*, not many of them.
# > Then look at **tokens in** vs **tokens out** — input-heavy means big prompts (documents in the system prompt),
# > output-heavy means long answers. **USD / 1k calls** is the unit price per tenant, the number to compare with what
# > the tenant pays you.

# %% [markdown]
# ### Latency per feature — and why a long answer does not mean a slow model
# The Converse record carries the server-side `latencyMs`. Percentiles per feature, plus **output tokens per second**:
# the number that tells *slow model* apart from *long answer*.
#
# > 💡 **Concept — latency = fixed cost + tokens ÷ speed.** A call's latency is roughly time to first token plus
# > output tokens divided by the generation speed. A feature that writes long answers is slow even on a fast model;
# > a feature with short answers and high latency points at the model, the prompt size or throttling. Dividing output
# > tokens by latency separates the two. p95, not the average, is what the unluckiest users of a feature feel.
#
# > 🔑 **Key code** — `pct(output.outputBodyJson.metrics.latencyMs, 95)` reads the server-side latency the Converse
# > response itself reports; `sum(output tokens) * 1000 / sum(latencyMs)` is output tokens per second per feature.

# %%
# ▶ KEY: percentiles of the server-side latency + output tokens per second, per feature
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
# > 📊 **How to read this** — sorted by p95, slowest first. Read **avg_out** next to **p50_ms**: if the slowest
# > feature also has the most output tokens and the *highest* tokens per second, the model is fast and the answers are
# > long — shorten the answer, don't change the model. A low **out_tok_per_s** with short answers would point at the
# > model or the input size instead. With 8–14 calls per feature, p95 is close to the maximum — indicative only.

# %% [markdown]
# ### Model errors that return HTTP 200
# Slide 15's *model errors*: an answer cut off at `maxTokens` is a success for every infrastructure metric. The stop
# reason is in the log — group by it.
#
# > 💡 **Concept — `stopReason`.** Every Converse response says why generation ended: `end_turn` (the model finished),
# > `max_tokens` (it hit `maxTokens` — the answer is cut off mid-sentence), `tool_use` (it wants a tool), or
# > `guardrail_intervened` / `content_filtered`. HTTP status, error metrics and latency all look healthy for a
# > truncated answer; only the stop reason gives it away.
#
# > 🔑 **Key code** — `stats count(*) … by feature, output.outputBodyJson.stopReason`, then a small pivot in Python:
# > `truncated = max_tokens / (end_turn + max_tokens)` per feature.

# %%
# ▶ KEY: count stop reasons per feature - truncation hides here, not in any error metric
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
# > 📊 **How to read this** — one row per feature: how many answers ended normally (`end_turn`) and how many were cut
# > off (`max_tokens`). Any **truncated** share above 0 % is a user-visible defect; a share near 100 % means the
# > `maxTokens` setting is wrong for that feature's task, not that the model misbehaved.

# %% [markdown]
# ### The most expensive prompts — and what the log shows of them
# The top prompts by input tokens, with the first characters of the user text — a short question: the tokens are the
# documents in the system prompt. The e-mail address in the
# `email-reply` prompts is masked by the data protection policy: Insights shows asterisks unless the reader has
# `logs:Unmask` and asks for `unmask(@message)`.
#
# > 🔑 **Key code** — `sort input.inputTokenCount desc | limit 3` finds the most expensive prompts;
# > `substr(input.inputBodyJson.messages.0.content.0.text, 0, 60)` reads into the logged request body (array elements
# > are addressed by index). The second query reads the same field of the `email-reply` records to check the masking.

# %%
# ▶ KEY: the three biggest prompts, with the start of the user text from the logged request body
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
# > 📊 **How to read this** — first table: the biggest prompts by input tokens. Compare **tokens_in** with the length of
# > **prompt_start** — when the user's text is one short line and the record still counts thousands of tokens, the
# > cost sits in the system prompt (here: documents sent with every question). Second table: the logged `email-reply`
# > prompts — an address should appear as a run of asterisks. A readable address would mean the data protection
# > policy is missing or the reader holds `logs:Unmask`.

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
#
# > ✅ **Takeaway §1** — Attribution is something the application attaches (`requestMetadata`), and the invocation log
# > plus Logs Insights turns it into cost, latency and truncation per tenant and feature — after you have reconciled the
# > log with what the client sent.

# %% [markdown]
# ## 2 · Is it right? Quality as a metric
# The refund desk's bot answers policy questions. Version **v1** sends the policy wording with the question. Version
# **v2** is a release in which the retrieval step silently broke: same prompt, no documents. Both return HTTP 200 —
# would any infrastructure dashboard turn red?
#
# Each answer is scored by the guardrail's **contextual grounding check** (`ApplyGuardrail` with the policy as
# `grounding_source`, the question as `query`). Because the questions have known answers, a deterministic **fact check**
# runs next to it — that is how we find out how good the detector is.
#
# > 💡 **Concept — `ApplyGuardrail` and measuring a detector.** `ApplyGuardrail` (runtime API) evaluates a guardrail
# > on any text without invoking a model, so you can score answers that came from anywhere. Each content block carries
# > a **qualifier** that tells the grounding check its role: `grounding_source` (the reference), `query` (the
# > question), `guard_content` (the answer to judge). A detector has two failure modes: a **miss** (a wrong answer
# > scores above the threshold) and a **false alarm** (a right answer scores below it). You can only count them on
# > questions whose correct answer you know — hence the regex fact check next to every score.

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

# %% [markdown]
# > 🔑 **Key code**
# > - `apply_guardrail(..., guardrailVersion="DRAFT", source="OUTPUT", content=[...])` — score an answer without a model
# >   call; the three `qualifiers` assign source, query and answer.
# > - The scores come back in `assessments[].contextualGroundingPolicy.filters[]` — one entry per filter type, with
# >   `score`, `threshold` and `detected`.
# > - `requestMetadata` in `answer_and_score` carries `prompt_version` — the link from a bad score back to the release.

# %%
def grounding(answer: str, question: str) -> dict:
    # ▶ KEY: a guardrail check on any text - no model call; qualifiers say which block is source, query, answer
    r = runtime.apply_guardrail(guardrailIdentifier=guardrail_id, guardrailVersion="DRAFT", source="OUTPUT",
                                content=[
        {"text": {"text": POLICY, "qualifiers": ["grounding_source"]}},
        {"text": {"text": question, "qualifiers": ["query"]}},
        {"text": {"text": answer, "qualifiers": ["guard_content"]}}])
    # ▶ KEY: one filter entry per type (GROUNDING, RELEVANCE) with score and detected
    f = {x["type"]: x for a in r["assessments"] for x in a.get("contextualGroundingPolicy", {}).get("filters", [])}
    return {"grounding": f["GROUNDING"]["score"], "relevance": f["RELEVANCE"]["score"],
            "flagged": f["GROUNDING"]["detected"]}

def answer_and_score(job) -> dict:
    version, (question, fact) = job
    r = runtime.converse(modelId=m.NOVA_LITE, system=[{"text": VERSIONS[version]}],
                         messages=[{"role": "user", "content": [{"text": question}]}],
                         inferenceConfig={"maxTokens": 150, "temperature": 0},
                         # ▶ KEY: the prompt version goes into the log record - the diagnosis needs it later
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
# > 📊 **How to read this** — twelve rows, v1 first. **fact ok** is the ground truth (the regex), **grounding** the
# > detector's score (0–1; below 0.75 → **flagged**). Look for agreement first: v1 rows should be fact ok with high
# > grounding; v2 rows (no policy in the prompt) should be wrong with low grounding. Then look for the exceptions —
# > a *False* fact with grounding above 0.75 is a miss. **relevance** stays high in both versions: a wrong answer
# > can still be on topic, which is exactly why relevance alone is not a quality signal.
#
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
# ▶ KEY: the detector's two failure modes, counted against the fact check
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
# > 📊 **How to read this** — the table compares the versions: **facts correct** (truth) against **mean grounding** and
# > **flagged** (detector). The printed line counts the detector's errors: *missed* wrong answers are the dangerous kind
# > (a wrong answer that looks fine), *false alarms* cost reviewer time. Scores of missed answers usually sit just
# > above the threshold — that is where a threshold decision is made, with these numbers in hand.
#
# ### Publish the scores — Embedded Metric Format
# One **EMF** log line per answer: CloudWatch extracts the metrics (no `PutMetricData` call), and the line itself stays
# in the log with the `requestId` — the key into the invocation log record of that exact call. Two dimension sets:
# per feature (for the alarm) and per feature + prompt version (for the diagnosis).
#
# ![From a grounding score to an alarm, and from the alarm back to the prompt](../assets/diagrams/m08_quality_loop.svg)
#
# > 💡 **Concept — Embedded Metric Format (EMF).** A structured JSON log line whose `_aws.CloudWatchMetrics` block
# > tells CloudWatch Logs which top-level fields are **metrics** (with units), which are **dimensions**, and in which
# > namespace. Logs extracts the metric values at ingestion; every other field (here `requestId`, `question`, `run`)
# > is just a log property — searchable with Logs Insights, but not a metric, so high-cardinality values cost nothing
# > extra. One write gives you both: a number to alarm on and the evidence to investigate it.
#
# > 🔑 **Key code** — the `_aws` block: `Namespace`, `Dimensions` as a **list of dimension sets** (each set becomes its
# > own metric series), `Metrics` with names and units; the metric values and dimension values are ordinary top-level
# > keys of the same JSON object.

# %%
def emf_line(a: dict) -> dict:
    ts = int(time.time() * 1000)
    return {"timestamp": ts, "message": json.dumps({
        # ▶ KEY: the EMF directive - which keys are metrics, which are dimensions, in which namespace
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
#
# > 💡 **Concept — a metric-math alarm.** `PutMetricAlarm` with `Metrics=[…]` instead of a single metric: each entry is
# > either a `MetricStat` (a metric, a statistic, a period) or an `Expression` over the other entries' ids; exactly
# > one entry has `ReturnData: True` — that is the series the alarm evaluates. `IF(answers >= 5, …, 0)` makes a quiet
# > minute count as healthy, and `TreatMissingData="notBreaching"` does the same for a minute with no data at all.
# > The alarm compares each 1-minute value with `Threshold` and changes state after `DatapointsToAlarm` of
# > `EvaluationPeriods` breach.
#
# > 🔑 **Key code** — `ALARM_METRICS`: two hidden `Sum` series (`ReturnData: False`) and the `rate` expression that
# > is returned; `put_metric_alarm(..., Threshold=20, EvaluationPeriods=1, DatapointsToAlarm=1)`.

# %%
def stat(metric_id: str, name: str) -> dict:
    return {"Id": metric_id, "ReturnData": False, "MetricStat": {"Period": 60, "Stat": "Sum", "Metric": {
        "Namespace": NAMESPACE, "MetricName": name, "Dimensions": [{"Name": "Feature", "Value": "refund-bot"}]}}}

ALARM_METRICS = [stat("ungrounded", "Ungrounded"), stat("answers", "Answers"),
                 # ▶ KEY: the alarm evaluates a RATE, and only when there are at least 5 answers in the minute
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
#
# > 🔑 **Key code** — `put_log_events(...)` is the only write: the metrics come out of those log lines. To know *which*
# > minute the alarm judged, `alarm_on_this_run` reads `StateReasonData` → `evaluatedDatapoints[].timestamp` and
# > accepts only a datapoint from this run's minute; `set_alarm_state` resets a stale ALARM once.
#
# > ⏱️ This cell waits for CloudWatch — usually 25–120 s, at most 3 minutes, with only a progress line meanwhile.

# %%
t0 = time.time()
EVENTS = [emf_line(a) for a in ANSWERS]
this_minute = EVENTS[0]["timestamp"] // 60_000 * 60_000
# ▶ KEY: twelve log lines - CloudWatch extracts the metrics from them, no PutMetricData
logs.put_log_events(logGroupName=APP_LOG, logStreamName=APP_STREAM, logEvents=EVENTS)
reset = []

def alarm_on_this_run() -> str:
    a = cw.describe_alarms(AlarmNames=[ALARM])["MetricAlarms"][0]
    # ▶ KEY: which datapoints did the alarm actually evaluate? Only this run's minute counts
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
#
# > 🔑 **Key code** — the first query reads the **app** log group (EMF lines are queryable like any JSON log); the
# > second uses `filter requestId in [...]` on the **invocation** log and measures `strlen(input.inputBodyJson.system.0.text)`
# > — the length of the system prompt the model actually received.

# %%
Q_UNGROUNDED = f'''filter run = "{RUN}" and Ungrounded = 1
| stats count(*) as ungrounded, avg(GroundingScore) as mean_score by PromptVersion'''
wait_for_records(f'run = "{RUN}"', len(ANSWERS), group=APP_LOG)
m.table(insights(Q_UNGROUNDED, group=APP_LOG))
ungrounded = sorted((a for a in ANSWERS if a["grounding"] < GROUNDING_THRESHOLD), key=lambda a: a["grounding"])
ids = json.dumps([a["requestId"] for a in ungrounded])     # every ungrounded answer: any one record is enough
# ▶ KEY: from the metric's evidence (requestId) to the exact prompt in the invocation log
rec = insights(f'''filter requestId in {ids} | fields requestId, requestMetadata.prompt_version as version,
  strlen(input.inputBodyJson.system.0.text) as system_prompt_chars, input.inputTokenCount as tokens_in | limit 1''')
print("invocation record of one ungrounded answer:", rec)
print("✓ root cause visible: the v2 call carried no policy text" if rec and int(rec[0]["system_prompt_chars"]) < 500
      else "⚠ no invocation record of an ungrounded answer yet (log delivery lags) - run the cell again in a minute")

# %% [markdown]
# > 📊 **How to read this** — the table (from the EMF lines) says *which prompt version* produced the ungrounded
# > answers and how low they scored. The printed record (from the invocation log) says *why*: compare
# > **system_prompt_chars** with the length of a v1 system prompt, which carries the whole policy wording (thousands of
# > characters). A system prompt of a few hundred characters or less means the documents never reached the model.

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
#
# > ✅ **Takeaway §2** — Quality can be a metric like latency: score every answer, publish the score as EMF, alarm on a
# > rate with a minimum sample — and keep the `requestId` in the log line so the page leads straight to the prompt.
# > Measure the detector itself before you trust (or block on) it.

# %% [markdown]
# ## 3 · One dashboard, three layers — defined as code
# Slide 10's layers: an **executive** row (is quality OK? the alarm), an **operational** row (cost per tenant from the
# invocation log, grounding per prompt version), a **diagnostic** row (latency, time to first token, throttles per model
# from AWS/Bedrock). The body is JSON, so it lives in version control and is deployed like code.
#
# > 💡 **Concept — a CloudWatch dashboard body.** `PutDashboard` takes a JSON document with a list of `widgets`, each
# > placed on a grid **24 units wide** (`x`, `y`, `width`, `height`). Widget types used here: `text` (Markdown),
# > `metric` (metric series and metric-math expressions, the same syntax as the alarm), `alarm` (alarm status by ARN)
# > and `log` (a Logs Insights query, with `SOURCE '<log group>'` in the query string). The **AWS/Bedrock** namespace
# > publishes per-`ModelId` metrics such as `Invocations`, `InvocationLatency`, `InvocationThrottles`, token counts and
# > `TimeToFirstToken` — account-wide, with no tenant or feature.
#
# > 🔑 **Key code** — `BODY["widgets"]`: one row per layer (`y` = 0/1, 7, 13); the `RATE` list reuses the alarm's
# > metric math; `put_dashboard(...)` returns `DashboardValidationMessages` — an empty list means the body is valid.

# %%
def metric_widget(x, y, w, title, metrics, **props):
    return {"type": "metric", "x": x, "y": y, "width": w, "height": 6, "properties": {
        "title": title, "metrics": metrics, "region": REGION, "period": 60, "view": "timeSeries", **props}}

# ▶ KEY: the same metric math as the alarm - the dashboard shows exactly what pages
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
# ▶ KEY: deploy the JSON; an empty DashboardValidationMessages list = valid body
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
# ▶ KEY: CloudWatch renders the widget server-side and returns PNG bytes
display(Image(cw.get_metric_widget_image(MetricWidget=json.dumps(widget))["MetricWidgetImage"]))

# %% [markdown]
# > 📊 **How to read this** — one bar per prompt version: the share of its answers in the last 10 minutes that scored
# > below the grounding threshold, against the 20 % alarm line. A healthy version sits near 0; a version far above the
# > line is the one to roll back. The window is the whole 10 minutes (`setPeriodToTimeRange`), so a re-run within that
# > time adds its answers to the same bars.

# %% [markdown]
# > **Think about it:** Which widget would you show the CFO, which one the on-call engineer — and which one would
# > you delete because nobody makes a decision with it?
#
# > ✅ **Takeaway §3** — A dashboard is code: a JSON body in version control, validated on deploy, with one row per
# > audience — quality first, then cost, then the per-model diagnostics.

# %% [markdown]
# ## 4 · Traces: what an agent did, step by step
# An agent turns one question into several model calls and tool calls. **Strands** emits **OpenTelemetry** spans
# (GenAI semantic conventions: `gen_ai.usage.input_tokens`, `gen_ai.tool.name`, …). In production you export them
# with OTLP to CloudWatch (generative AI observability) or your own collector; here an **in-memory exporter** captures
# them in the notebook. The agent's calls also carry `requestMetadata`, so every `chat` span should have an invocation
# log record.
#
# ![One agent request as a span tree, reconciled with the invocation log](../assets/diagrams/m08_agent_trace.svg)
#
# > 💡 **Concept — OpenTelemetry in four words.** A **trace** is one request end to end; it is made of **spans**
# > (a named, timed operation with a parent, a status and **attributes**). A **TracerProvider** creates spans; a
# > **span processor** hands finished spans to an **exporter** (OTLP to a collector or to CloudWatch in production, an
# > in-memory list here). Strands instruments itself: you only install a provider. The global provider can be set
# > **once per process** — a second `set_tracer_provider` is ignored, hence the guard.
#
# > 🔑 **Key code** — `TracerProvider()` + `SimpleSpanProcessor(InMemorySpanExporter())` + `trace.set_tracer_provider`
# > (in production: `BatchSpanProcessor` with an OTLP exporter). In the agent cell, `additional_args={"requestMetadata":
# > AGENT_TAGS}` makes every model call of the agent carry our tags into the invocation log, and
# > `trace_attributes={"session.id": …}` stamps the session onto the spans — the same id on both sides.

# %%
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

if not isinstance(trace.get_tracer_provider(), TracerProvider):      # the global provider can be set once per process
    # ▶ KEY: provider + processor + exporter - Strands' spans land in this in-memory list
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
                                 # ▶ KEY: every model call of the agent carries our tags into the invocation log
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
#
# > 🔑 **Key code** — the tree is rebuilt from `span.parent.span_id` (a root span has no parent); `show` recurses
# > through children in start-time order and prints `gen_ai.usage.input_tokens` / `output_tokens` for `chat` spans and
# > `gen_ai.tool.name` for tool spans. Durations are `end_time − start_time` in nanoseconds.

# %%
spans = SPANS.get_finished_spans()
children = {}
for s in spans:
    # ▶ KEY: rebuild the tree from parent span ids
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
# > 📊 **How to read this** — indentation is nesting: the agent span contains one cycle per reasoning step, each cycle a
# > `chat` span and, if the model asked for one, an `execute_tool` span. Read the **ms** column top-down to see where
# > the time went (compare the summary line's model time with tool time), and the **in** column of successive `chat`
# > spans: each cycle re-sends the whole conversation so far, so input tokens grow step by step.
#
# ### Reconcile the trace with the invocation log
# Spans are produced by the client library; the invocation log by the service. The log also records **failed
# attempts** (`errorCode`, e.g. a `ThrottlingException` the SDK retried) — a trace shows those only as a slow `chat`
# span. Pair the successful calls, count the failed ones; when tokens disagree, the log is the one to bill from.
#
# > 🔑 **Key code** — the log query filters on `requestMetadata.session` (the agent's tag); records with an
# > `errorCode` are failed attempts. Pairing: first by **equal input tokens** (every cycle's prompt is longer than the
# > last, so the count identifies the call), then any leftovers in time order. `gap` sums the token differences.
#
# > ⏱️ This cell first waits for the agent's records — usually ≈ 15 s, up to 90 s — before it prints anything.

# %%
ok_chat = [s for s in chat if s.status.status_code != StatusCode.ERROR]
wait_for_records(f'requestMetadata.session = "{SESSION}" and output.outputTokenCount >= 0', len(ok_chat), timeout=90)
LOGGED = insights(f'''filter requestMetadata.session = "{SESSION}"
| fields @timestamp, errorCode, input.inputTokenCount as tokens_in, output.outputTokenCount as tokens_out,
         output.outputBodyJson.stopReason as stop_reason, inferenceRegion | sort @timestamp asc''')
ok_log = sorted((r for r in LOGGED if "errorCode" not in r), key=lambda r: (r["@timestamp"], int(r["tokens_in"])))
# ▶ KEY: the service logs failed attempts too - the client-side trace only sees a slow span
failed = [r["errorCode"] for r in LOGGED if "errorCode" in r]
# ▶ KEY: pair by input tokens (each cycle re-sends a longer conversation), then leftovers in time order
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
# > 📊 **How to read this** — one row per successful model call of the agent: the span's token counts next to the log
# > record's. Equal pairs mean the client library and the service agree; any difference is resolved in favour of the
# > log (it is what the service billed). **stop reason** shows the agent's rhythm — `tool_use` for steps that call a
# > tool, `end_turn` for the final answer; **served in** is the Region a cross-Region (`us.`) profile routed the call to.

# %% [markdown]
# > **Think about it:** Input tokens grow with every cycle. Why — and what does that mean for the cost of an agent
# > that needs ten steps instead of three?
#
# <details><summary>More questions</summary>
#
# - Which attribute would you add to every span so a support ticket can be traced to one agent session?
# - The invocation log already has the prompts. What do traces add that the log cannot?
# </details>
#
# > ✅ **Takeaway §4** — Traces explain *how* an agent spent its time and tokens step by step; the invocation log is
# > what the service billed and also shows the retried failures. Tag both with the same session id and check that they
# > agree.
#
# ## Summary — what you built
# | Piece | AWS API / pattern | Production next step |
# |---|---|---|
# | Invocation logging, safely | `Get/PutModelInvocationLoggingConfiguration`, save/restore, least-privilege role, data protection policy | own the setting centrally (one team, infrastructure as code); S3 destination for long-term audit |
# | Attribution | `requestMetadata` on `Converse` + Logs Insights `StartQuery` | saved queries per team; a scheduled cost-per-tenant report |
# | Log reconciliation | request id join, token comparison | a scheduled check that alarms on missing records |
# | Quality metric | `ApplyGuardrail` grounding score → EMF line | score a sample of production traffic; calibrate the threshold on labelled answers (Module 9) |
# | Quality alarm | metric-math `PutMetricAlarm` with a minimum sample | alarm action → SNS / EventBridge → roll back the prompt version (AppConfig, Module 1) |
# | Dashboard | `PutDashboard` JSON, `GetMetricWidgetImage` | dashboard body in version control, deployed with the application |
# | Agent traces | Strands + OpenTelemetry spans, reconciled with the log | OTLP export to CloudWatch generative AI observability or your collector |

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
