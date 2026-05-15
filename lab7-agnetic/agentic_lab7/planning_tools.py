"""Planning-agent utilities for the Lab 7 agentic workflow."""

import json

import requests


def agent_catalog_for_prompt(model_templates):
    return {
        name: {
            "defaults": template["defaults"],
            "allowed_hyperparameters": sorted(template["allowed"]),
            "uses_scaler": template["use_scaler"],
        }
        for name, template in model_templates.items()
    }


def clean_hyperparameters(model_family, hyperparameters, model_templates):
    template = model_templates[model_family]
    params = template.get("defaults", {}).copy()
    allowed = set(template.get("allowed", set()))
    for key, value in (hyperparameters or {}).items():
        if key in allowed:
            params[key] = value
    return params


def request_strategy_from_nvidia(
    prompt,
    history,
    dataset_summary,
    baseline_summary,
    model_templates,
    api_key,
    model_name,
):
    if not api_key.strip():
        raise RuntimeError("Set NVIDIA_API_KEY in agentic_lab7/api_key.py before running the planning agent.")

    system_prompt = """
You are a planning agent for a collider signal-versus-background classifier.
Return only one JSON strategy object. Do not write Python code.
Do not return a JSON schema or a placeholder such as {"type": "object"}.
Choose only from the allowed model families and feature names.
The classifier infrastructure is fixed; you only choose feature_columns, model_family, and hyperparameters.
""".strip()

    schema_placeholder = {"type": "object"}
    example_strategy = {
        "strategy_name": "physics_mass_random_forest",
        "rationale": "Use invariant masses, angular separations, and b-tag multiplicity because they reflect the four-b signal topology.",
        "feature_columns": ["ht", "m_4j", "m_jj_12", "min_deltaR_jj", "n_btags"],
        "model_family": "random_forest",
        "hyperparameters": {"n_estimators": 180, "min_samples_leaf": 2, "max_features": "sqrt"},
    }

    user_prompt = f"""
Task: classify exotic-Higgs signal events from QCD heavy-flavour background events.
Signal label is 1. Background label is 0.

Available feature names:
{json.dumps(dataset_summary["feature_names"], indent=2)}

Allowed classification agents:
{json.dumps(agent_catalog_for_prompt(model_templates), indent=2, default=str)}

Dataset summary:
{json.dumps({k: v for k, v in dataset_summary.items() if k != "feature_names"}, indent=2)}

Baseline summary:
{json.dumps(baseline_summary, indent=2)}

Previous round summaries:
{json.dumps(history, indent=2)}

Planning instruction:
{prompt}

Return one actual strategy JSON object, not a JSON schema.
Do not return this schema placeholder: {json.dumps(schema_placeholder)}.
Do not describe the schema.

The response must have exactly these keys:
- strategy_name: short string
- rationale: one or two sentences
- feature_columns: list of feature names copied exactly from Available feature names
- model_family: one key copied exactly from Allowed classification agents
- hyperparameters: object using only allowed hyperparameter names for that model family

Example response shape:
{json.dumps(example_strategy, indent=2)}
""".strip()

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 1536,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload, timeout=120)
    if not response.ok:
        raise RuntimeError(f"NVIDIA API error {response.status_code}: {response.text[:1000]}")

    raw_content = response.json()["choices"][0]["message"]["content"]
    plan = json.loads(raw_content)
    return validate_plan(plan, dataset_summary["feature_names"], model_templates, raw_content=raw_content)


def validate_plan(plan, feature_names, model_templates, raw_content=None):
    if plan == {"type": "object"}:
        raise ValueError("Planning agent returned a schema placeholder instead of a strategy. Tighten PLANNING_PROMPT.")
    if "strategy" in plan and isinstance(plan["strategy"], dict):
        plan = plan["strategy"]
    if "plan" in plan and isinstance(plan["plan"], dict):
        plan = plan["plan"]

    required = ["strategy_name", "rationale", "feature_columns", "model_family", "hyperparameters"]
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Planning response missing required keys: {missing}. Raw response: {raw_content or plan}")

    family = plan["model_family"]
    if family not in model_templates:
        raise ValueError(f"Planning response chose unknown model_family {family!r}. Allowed: {list(model_templates)}")

    available = set(feature_names)
    selected = list(dict.fromkeys(col for col in plan["feature_columns"] if col in available))
    if not selected:
        raise ValueError("Planning response did not choose any valid feature columns.")

    return {
        "strategy_name": str(plan["strategy_name"]),
        "rationale": str(plan["rationale"]),
        "feature_columns": selected,
        "model_family": family,
        "hyperparameters": clean_hyperparameters(family, plan.get("hyperparameters", {}), model_templates),
    }


def summarize_round(round_id, plan, classifier_agent, best):
    return {
        "round": round_id,
        "strategy": plan["strategy_name"],
        "model_family": plan["model_family"],
        "features_used": len(plan["feature_columns"]),
        "hyperparameters": json.dumps(plan["hyperparameters"], sort_keys=True),
        "test_auc": classifier_agent.metrics["test"]["auc"],
        "best_threshold": best["threshold"],
        "best_S_over_sqrt_B": best["S_over_sqrt_B"],
        "rationale": plan["rationale"],
    }
