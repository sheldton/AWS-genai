#!/usr/bin/env python3
"""List every resource the kit may leave behind (name contains mlaidp2), across all services the demos use.

    .venv/bin/python tools/sweep.py          list only (never deletes - each notebook's cleanup section deletes)

Also reports whether Bedrock model invocation logging (account-level, M08) is configured.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import mlaidp as m  # noqa: E402

K = "mlaidp2"


def names(fn):
    try:
        return [x for x in fn() if K in x.lower().replace("_", "")]
    except Exception as e:  # a service not available in the Region must not stop the sweep
        return [f"(error: {type(e).__name__}: {str(e)[:80]})"]


def pages(client, op, key, field, **kw):
    out = []
    for p in m.client(client).get_paginator(op).paginate(**kw):
        out += [x[field] for x in p.get(key, [])]
    return out


def nexttoken(client, op, key, field, **kw):
    c, out, tok = m.client(client), [], None
    while True:
        r = getattr(c, op)(**kw, **({"nextToken": tok} if tok else {}))
        out += [x[field] for x in r.get(key, [])]
        tok = r.get("nextToken")
        if not tok:
            return out


CHECKS = {
    "IAM roles": lambda: pages("iam", "list_roles", "Roles", "RoleName"),
    "Lambda functions": lambda: pages("lambda", "list_functions", "Functions", "FunctionName"),
    "S3 buckets": lambda: [b["Name"] for b in m.client("s3").list_buckets()["Buckets"]],
    "Log groups": lambda: pages("logs", "describe_log_groups", "logGroups", "logGroupName"),
    "CloudFormation stacks": lambda: [s["StackName"] for s in m.client("cloudformation").list_stacks()["StackSummaries"]
                                      if s["StackStatus"] != "DELETE_COMPLETE"],
    "Guardrails": lambda: nexttoken("bedrock", "list_guardrails", "guardrails", "name"),
    "Flows": lambda: nexttoken("bedrock-agent", "list_flows", "flowSummaries", "name"),
    "Prompts": lambda: nexttoken("bedrock-agent", "list_prompts", "promptSummaries", "name"),
    "Knowledge bases": lambda: nexttoken("bedrock-agent", "list_knowledge_bases", "knowledgeBaseSummaries", "name"),
    "Evaluation jobs": lambda: nexttoken("bedrock", "list_evaluation_jobs", "jobSummaries", "jobName"),
    "Application inference profiles": lambda: nexttoken("bedrock", "list_inference_profiles", "inferenceProfileSummaries",
                                                        "inferenceProfileName", typeEquals="APPLICATION"),
    "Prompt routers": lambda: nexttoken("bedrock", "list_prompt_routers", "promptRouterSummaries", "promptRouterName", type="custom"),
    "AgentCore gateways": lambda: nexttoken("bedrock-agentcore-control", "list_gateways", "items", "name"),
    "AgentCore memories": lambda: nexttoken("bedrock-agentcore-control", "list_memories", "memories", "id"),
    "AgentCore policy engines": lambda: nexttoken("bedrock-agentcore-control", "list_policy_engines", "policyEngines", "name"),
    "S3 vector buckets": lambda: nexttoken("s3vectors", "list_vector_buckets", "vectorBuckets", "vectorBucketName"),
    "AppConfig applications": lambda: [a["Name"] for a in m.client("appconfig").list_applications()["Items"]],
    "AppConfig strategies": lambda: [a["Name"] for a in m.client("appconfig").list_deployment_strategies()["Items"]],
    "CloudWatch dashboards": lambda: [d["DashboardName"] for d in m.client("cloudwatch").list_dashboards()["DashboardEntries"]],
    "CloudWatch alarms": lambda: pages("cloudwatch", "describe_alarms", "MetricAlarms", "AlarmName"),
    "SQS queues": lambda: m.client("sqs").list_queues().get("QueueUrls", []),
    "DynamoDB tables": lambda: m.client("dynamodb").list_tables()["TableNames"],
    "API Gateway REST APIs": lambda: [a["name"] for a in m.client("apigateway").get_rest_apis(limit=500)["items"]],
    "Transcribe jobs": lambda: [j["TranscriptionJobName"] for j in m.client("transcribe").list_transcription_jobs(MaxResults=100)["TranscriptionJobSummaries"]],
}

left = 0
for label, fn in CHECKS.items():
    found = names(fn)
    left += len(found)
    print(f"{'✗' if found else '✓'} {label:32s} {', '.join(m.masked(x) for x in found) if found else 'none'}")
log = m.client("bedrock").get_model_invocation_logging_configuration().get("loggingConfig")
print(f"{'!' if log else '✓'} {'Model invocation logging':32s} {'CONFIGURED: ' + m.masked(str(log))[:120] if log else 'not configured'}")
print(f"\n{'nothing left' if not left else f'{left} leftover(s)'}")
sys.exit(1 if left else 0)
