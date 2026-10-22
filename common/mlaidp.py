"""Plumbing shared by the MLAIDP notebooks - and nothing else.

Every AWS call a demo is about is written out in the notebook itself. This file only holds what would be noise on a
screen: the boto3 session (profile / Region), client retry settings, the price table, a run id, a few table and
chart helpers, and a resource sweep for cleanup.
"""
from __future__ import annotations

import functools
import json
import os
import re
import time
import uuid
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

PROFILE = os.environ.get("AWS_PROFILE") or None
REGION = os.environ.get("AWS_REGION", "us-east-1")
PREFIX = "mlaidp2"
HOME = Path(__file__).resolve().parent.parent
ASSETS = HOME / "assets"
TAGS = {"Project": f"{PREFIX}-demo", "Owner": "student"}


# ------------------------------------------------------------------ session and clients
@functools.lru_cache(maxsize=None)
def session() -> boto3.Session:
    return boto3.Session(profile_name=PROFILE, region_name=REGION)


@functools.lru_cache(maxsize=None)
def client(name: str, retries: int = 8, read_timeout: int = 120):
    """A client with adaptive retries: the on-demand quotas of a new account are small."""
    return session().client(name, config=Config(retries={"max_attempts": retries, "mode": "adaptive"},
                                                 read_timeout=read_timeout, max_pool_connections=32))


def bare_client(name: str, read_timeout: int = 60):
    """A client WITHOUT retries - for cells that show what a failure looks like."""
    return session().client(name, config=Config(retries={"total_max_attempts": 1, "mode": "standard"},
                                                read_timeout=read_timeout, connect_timeout=5))


@functools.lru_cache(maxsize=None)
def account() -> str:
    return session().client("sts").get_caller_identity()["Account"]


def masked(text: str) -> str:
    """Hide the account id in anything printed on screen."""
    return str(text).replace(account(), "123456789012")


def run_id(module: str) -> str:
    """One id per module, reused on re-run (stored next to the notebooks) so cleanup-by-name always works."""
    f = HOME / "notebooks" / f".run_id_{module.lower()}"
    if f.exists():
        return f.read_text().strip()
    rid = time.strftime("%m%d") + uuid.uuid4().hex[:4]
    f.write_text(rid)
    return rid


def error_code(e: ClientError) -> str:
    return e.response["Error"]["Code"]


def retry(fn, *, when: str, tries: int = 15, delay: float = 4.0):
    """Call fn() until it stops raising a ClientError matching `when` (IAM propagation, eventual consistency)."""
    for i in range(tries):
        try:
            return fn()
        except ClientError as e:
            if i == tries - 1 or not re.search(when, error_code(e) + " " + str(e), re.I):
                raise
            time.sleep(delay)


def wait_for(fn, *, until, timeout: int = 600, every: float = 5.0, label: str = ""):
    """Poll fn() until until(result) is true. Prints one line per state change."""
    t0, last = time.time(), None
    while True:
        r = fn()
        state = until(r)
        if state is True:
            print(f"  {label} ready after {int(time.time() - t0)} s")
            return r
        if state != last and state not in (False, None):
            print(f"  {label} {state} ({int(time.time() - t0)} s)")
            last = state
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{label} not ready after {timeout} s")
        time.sleep(every)


# ------------------------------------------------------------------ models and prices
# Models that answered in the test account (us-east-1) in October 2026. Claude needs a use-case form in that
# account, so no demo depends on it.
NOVA_MICRO = "us.amazon.nova-micro-v1:0"
NOVA_LITE = "us.amazon.nova-lite-v1:0"
NOVA2_LITE = "us.amazon.nova-2-lite-v1:0"
NOVA_PRO = "us.amazon.nova-pro-v1:0"
LLAMA_8B = "us.meta.llama3-1-8b-instruct-v1:0"
LLAMA_70B = "us.meta.llama3-3-70b-instruct-v1:0"
MISTRAL_L3 = "mistral.mistral-large-3-675b-instruct"
GPT_OSS_120B = "openai.gpt-oss-120b-1:0"
GPT_OSS_20B = "openai.gpt-oss-20b-1:0"
QWEN3_32B = "qwen.qwen3-32b-v1:0"
TITAN_EMBED_V2 = "amazon.titan-embed-text-v2:0"
COHERE_RERANK = f"arn:aws:bedrock:{REGION}::foundation-model/cohere.rerank-v3-5:0"

# USD per 1,000 tokens (AWS Price List API, AmazonBedrock, us-east-1, read 2026-10-19). cache = cache-read price.
PRICES = {
    NOVA_MICRO:   {"name": "Nova Micro",      "in": 0.000035, "out": 0.00014, "cache": 0.00000875},
    NOVA_LITE:    {"name": "Nova Lite",       "in": 0.00006,  "out": 0.00024, "cache": 0.000015},
    NOVA2_LITE:   {"name": "Nova 2 Lite",     "in": 0.00033,  "out": 0.00275, "cache": 0.0000825,
                   "flex": (0.000165, 0.001375), "priority": (0.0005775, 0.0048125)},
    NOVA_PRO:     {"name": "Nova Pro",        "in": 0.0008,   "out": 0.0032,  "cache": 0.0002,
                   "flex": (0.0004, 0.0016), "priority": (0.0014, 0.0056)},
    LLAMA_8B:     {"name": "Llama 3.1 8B",    "in": 0.00022,  "out": 0.00022},
    LLAMA_70B:    {"name": "Llama 3.3 70B",   "in": 0.00072,  "out": 0.00072},
    MISTRAL_L3:   {"name": "Mistral Large 3", "in": 0.0005,   "out": 0.0015},
    GPT_OSS_120B: {"name": "gpt-oss-120b",    "in": 0.00015,  "out": 0.0006},
    GPT_OSS_20B:  {"name": "gpt-oss-20b",     "in": 0.00007,  "out": 0.0003},
    QWEN3_32B:    {"name": "Qwen3 32B",       "in": 0.00015,  "out": 0.0006},
    TITAN_EMBED_V2: {"name": "Titan Text Embeddings V2", "in": 0.00002, "out": 0.0},
}
RERANK_USD_PER_QUERY = 0.002


def model_name(model_id: str) -> str:
    key = model_id.split("/")[-1]
    for k, v in PRICES.items():
        if k == key or k.endswith(key) or key.endswith(k.split(".", 1)[-1]):
            return v["name"]
    return key


def usd(model_id: str, usage: dict, tier: str = "default") -> float:
    """Dollars for one Converse call from its `usage` block (cache reads at the cache price)."""
    key = model_id.split("/")[-1]
    p = PRICES.get(key) or next((v for k, v in PRICES.items() if key.endswith(k.split(".", 1)[-1])), None)
    if not p:
        return 0.0
    pin, pout = p[tier] if tier in ("flex", "priority") and tier in p else (p["in"], p["out"])
    return (usage.get("inputTokens", 0) * pin + (usage.get("cacheReadInputTokens") or 0) * p.get("cache", pin)
            + usage.get("outputTokens", 0) * pout) / 1000


def text_of(converse_response: dict) -> str:
    """The text of a Converse response, without <thinking> blocks or reasoning content."""
    parts = converse_response["output"]["message"]["content"]
    t = "".join(c.get("text", "") for c in parts if "text" in c)
    return re.sub(r"<thinking>.*?</thinking>\s*", "", t, flags=re.S).strip()


def parse_json(text: str):
    """First JSON object/array in a model answer (models like to wrap JSON in ``` fences)."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    m = re.search(r"\{.*\}|\[.*\]", t, re.S)
    try:
        return json.loads(m.group(0) if m else t)
    except (json.JSONDecodeError, AttributeError):
        return None


# ------------------------------------------------------------------ display
def table(rows, columns=None, *, caption: str | None = None, fmt: dict | None = None) -> None:
    """Show rows (list of lists/dicts or a DataFrame) as a compact table (returns nothing, so it never renders twice)."""
    import pandas as pd
    from IPython.display import display
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    sty = df.style.hide(axis="index").format(fmt or {}, na_rep="–")
    sty = sty.set_table_attributes('class="tex2jax_ignore"').set_properties(**{"text-align": "left", "font-size": "11pt"})
    if caption:
        sty = sty.set_caption(caption)
    display(sty)


def note(html: str, color: str = "#1d4ed8"):
    from IPython.display import HTML, display
    display(HTML(f'<div class="tex2jax_ignore" style="border-left:4px solid {color};padding:6px 12px;'
                 f'margin:6px 0;background:{color}11">{html}</div>'))


def setup_banner(module: str):
    import sys
    rid = run_id(module)
    print(f"account {account()[:4]}…  ·  {REGION}  ·  credentials {'AWS_PROFILE' if PROFILE else 'default chain'}  ·  run id {rid}")
    if str(HOME / ".venv") not in sys.executable:
        note("<b>Wrong kernel:</b> select <b>MLAIDP (Python 3.12)</b> and run this cell again.", "#b91c1c")
    return rid


# ------------------------------------------------------------------ cleanup helpers
def delete_role(role: str) -> None:
    iam = client("iam")
    try:
        for p in iam.list_role_policies(RoleName=role)["PolicyNames"]:
            iam.delete_role_policy(RoleName=role, PolicyName=p)
        for p in iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
            iam.detach_role_policy(RoleName=role, PolicyArn=p["PolicyArn"])
        iam.delete_role(RoleName=role)
        print(f"  deleted role {role}")
    except ClientError as e:
        if error_code(e) != "NoSuchEntity":
            print(masked(f"  role {role}: {e}"))


def delete_lambda(fn: str) -> None:
    try:
        client("lambda").delete_function(FunctionName=fn)
        print(f"  deleted lambda {fn}")
    except ClientError as e:
        if error_code(e) != "ResourceNotFoundException":
            print(masked(f"  lambda {fn}: {e}"))
    try:
        client("logs").delete_log_group(logGroupName=f"/aws/lambda/{fn}")
    except ClientError:
        pass


def empty_and_delete_bucket(name: str) -> None:
    s3 = client("s3")
    try:
        for page in s3.get_paginator("list_object_versions").paginate(Bucket=name):
            objs = [{"Key": o["Key"], "VersionId": o["VersionId"]}
                    for o in page.get("Versions", []) + page.get("DeleteMarkers", [])]
            for i in range(0, len(objs), 1000):
                s3.delete_objects(Bucket=name, Delete={"Objects": objs[i:i + 1000]})
        s3.delete_bucket(Bucket=name)
        print(masked(f"  deleted bucket {name}"))
    except ClientError as e:
        if error_code(e) != "NoSuchBucket":
            print(masked(f"  bucket {name}: {e}"))


def lambda_zip(source: str) -> bytes:
    """Zip one inline source file as index.py (handler index.handler)."""
    import io
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("index.py", source)
    return buf.getvalue()
