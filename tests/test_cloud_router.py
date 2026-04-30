from router.cloud_router import DEFAULT_POLICY, CloudRouter
from router.schema_v15 import CompactSchemaV15


def _schema(task_type, complexity="light"):
    return CompactSchemaV15(
        task_id="t-1",
        task_type=task_type,
        complexity=complexity,
        embedding_b64="x" * 4,
        embedding_dim=4096,
    )


def test_default_policy_routes_code_to_claude_opus():
    r = CloudRouter()
    decision = r.select(_schema("code"))
    assert decision.cloud_model == "claude-opus-4-7"
    assert decision.task_type == "code"


def test_default_policy_routes_qa_to_gemma():
    r = CloudRouter()
    decision = r.select(_schema("qa"))
    assert decision.cloud_model == "google/gemma-4-31B-it"


def test_unknown_task_type_falls_back_to_other():
    r = CloudRouter()
    decision = r.select(_schema("other"))
    assert decision.cloud_model == DEFAULT_POLICY["other"]


def test_custom_policy_overrides_defaults():
    r = CloudRouter(policy={"qa": "anthropic/claude-haiku-4-5"})
    decision = r.select(_schema("qa"))
    assert decision.cloud_model == "anthropic/claude-haiku-4-5"
    assert r.select(_schema("code")).cloud_model == "claude-opus-4-7"


def test_complexity_override_kicks_in_for_heavy():
    r = CloudRouter(complexity_override={"qa": "claude-opus-4-7"})
    light = r.select(_schema("qa", complexity="light"))
    heavy = r.select(_schema("qa", complexity="heavy"))
    assert light.cloud_model == "google/gemma-4-31B-it"
    assert heavy.cloud_model == "claude-opus-4-7"
    assert "heavy override" in heavy.reason


def test_complexity_override_does_not_apply_to_light():
    r = CloudRouter(complexity_override={"qa": "claude-opus-4-7"})
    decision = r.select(_schema("qa", complexity="light"))
    assert decision.cloud_model == "google/gemma-4-31B-it"


def test_repr_lists_policy_keys():
    r = CloudRouter()
    s = repr(r)
    assert "code" in s
    assert "qa" in s
