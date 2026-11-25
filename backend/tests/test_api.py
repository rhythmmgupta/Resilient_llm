from types import SimpleNamespace
import time
from typing import Any, Dict, List
from unittest.mock import patch

from fastapi.testclient import TestClient
import litellm
from litellm.exceptions import InternalServerError, RateLimitError

from app.config import KeyEntry, Settings
from app.db import Database
from app.llm_gateway import LLMGateway
from app.main import create_app


def _make_settings(
    db_path: str = "sqlite:///:memory:",
    enable_failure_simulation: bool = True,
    keys: int = 2,
) -> Settings:
    key_pool = [
        KeyEntry(key_id=f"key_{i}", api_key=f"secret-key-{i}", model="gemini/gemini-2.0-flash")
        for i in range(1, keys + 1)
    ]
    return Settings(
        key_pool=key_pool,
        database_url=db_path,
        allowed_origins=["http://localhost:3000"],
        enable_failure_simulation=enable_failure_simulation,
        cache_type="local",
        cache_dir="/tmp/test_cache",
        default_system_prompt="You are a helpful test assistant.",
        default_model="gemini/gemini-2.0-flash",
    )


def _mock_response(content: str, prompt_tokens: int = 25, completion_tokens: int = 8) -> Any:
    return litellm.ModelResponse(
        id="chatcmpl-test",
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        model="gemini-2.0-flash",
        usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": prompt_tokens + completion_tokens},
    )


def test_health_endpoint():
    settings = _make_settings()
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_validation_errors_and_no_history_pollution():
    settings = _make_settings()
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    # Empty whitespace message
    res = client.post("/chat", json={"session_id": "s-val-1", "message": "   "})
    assert res.status_code == 422
    assert "detail" in res.json()

    # Over 4000 chars after trimming
    res = client.post("/chat", json={"session_id": "s-val-1", "message": "a" * 4001})
    assert res.status_code == 422
    assert "detail" in res.json()

    # Empty session_id
    res = client.post("/chat", json={"session_id": "", "message": "Hello"})
    assert res.status_code == 422
    assert "detail" in res.json()

    # Malformed JSON
    res = client.post("/chat", content="{bad-json", headers={"Content-Type": "application/json"})
    assert res.status_code == 422
    assert "detail" in res.json()

    # Failed 422 requests must not create session history
    hist = client.get("/sessions/s-val-1/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_multiturn_history_isolation_and_system_prompt():
    settings = _make_settings()
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    recorded_calls: List[List[Dict[str, str]]] = []

    def fake_completion(**kwargs):
        recorded_calls.append(kwargs["messages"])
        last_user = kwargs["messages"][-1]["content"]
        return _mock_response(
            content=f"Reply to: {last_user}",
            prompt_tokens=10 * len(kwargs["messages"]),
            completion_tokens=6,
        )

    with patch("app.llm_gateway.litellm.completion", side_effect=fake_completion):
        r1 = client.post(
            "/chat",
            json={
                "session_id": "s-alpha",
                "message": "  My name is Alice.  ",
                "system_prompt": "Always speak like a pirate.",
            },
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["session_id"] == "s-alpha"
        assert d1["reply"] == "Reply to: My name is Alice."
        assert d1["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert d1["metadata"]["key_id"] == "key_1"
        assert d1["metadata"]["fallback_attempts"] == 0
        assert d1["metadata"]["input_tokens"] == 20
        assert d1["metadata"]["output_tokens"] == 6

        # Second turn in same session remembers history and overridden system_prompt
        r2 = client.post(
            "/chat",
            json={"session_id": "s-alpha", "message": "What is my name?"},
        )
        assert r2.status_code == 200
        assert recorded_calls[1] == [
            {"role": "system", "content": "Always speak like a pirate."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Reply to: My name is Alice."},
            {"role": "user", "content": "What is my name?"},
        ]

        # Separate session stays isolated and uses default system prompt
        r3 = client.post(
            "/chat",
            json={"session_id": "s-beta", "message": "Hello from beta"},
        )
        assert r3.status_code == 200
        assert recorded_calls[2] == [
            {"role": "system", "content": "You are a helpful test assistant."},
            {"role": "user", "content": "Hello from beta"},
        ]

    # History endpoint returns oldest first, without system prompt
    hist_res = client.get("/sessions/s-alpha/messages")
    assert hist_res.status_code == 200
    hist_json = hist_res.json()
    assert hist_json["session_id"] == "s-alpha"
    assert len(hist_json["messages"]) == 4
    assert [m["role"] for m in hist_json["messages"]] == ["user", "assistant", "user", "assistant"]
    assert "metadata" not in hist_json["messages"][0]
    assert hist_json["messages"][1]["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert hist_json["messages"][1]["metadata"]["input_tokens"] == 20
    assert hist_json["messages"][1]["metadata"]["output_tokens"] == 6


def test_litellm_caching_across_sessions():
    settings = _make_settings()
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    orig_completion = litellm.completion

    def completion_with_mock(*args, **kwargs):
        if "cache" not in kwargs or not kwargs["cache"].get("no-cache"):
            cached = litellm.cache.get_cache(*args, **kwargs)
            if cached is None:
                kwargs["mock_response"] = (
                    "The capital of France is Paris."
                    if "France" in kwargs["messages"][-1]["content"]
                    else "The capital of Spain is Madrid."
                )
                time.sleep(0.02)
        return orig_completion(*args, **kwargs)

    with patch("app.llm_gateway.litellm.completion", side_effect=completion_with_mock):
        first = client.post(
            "/chat",
            json={"session_id": "s-cache-1", "message": "What is the capital of France?"},
        )
        assert first.status_code == 200
        f_data = first.json()
        assert f_data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert f_data["metadata"]["key_id"] == "key_1"
        assert f_data["metadata"]["fallback_attempts"] == 0

        # Same first message in a NEW session must hit LLM_CACHE
        second = client.post(
            "/chat",
            json={"session_id": "s-cache-2", "message": "What is the capital of France?"},
        )
        assert second.status_code == 200
        s_data = second.json()
        assert s_data["reply"] == f_data["reply"]
        assert s_data["metadata"]["resolution_source"] == "LLM_CACHE"
        assert s_data["metadata"]["key_id"] is None
        assert s_data["metadata"]["fallback_attempts"] == 0
        assert s_data["metadata"]["input_tokens"] == f_data["metadata"]["input_tokens"]
        assert s_data["metadata"]["output_tokens"] == f_data["metadata"]["output_tokens"]
        assert s_data["metadata"]["latency_ms"] < f_data["metadata"]["latency_ms"]

        # Different message must NOT hit cache
        third = client.post(
            "/chat",
            json={"session_id": "s-cache-3", "message": "What is the capital of Spain?"},
        )
        assert third.status_code == 200
        assert third.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"


def test_failover_on_real_provider_errors_and_simulation_header():
    settings = _make_settings(enable_failure_simulation=True)
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    # 1. Real 429 on key_1 -> fails over to key_2
    def fake_real_failover(**kwargs):
        if kwargs["api_key"] == "secret-key-1":
            raise RateLimitError(message="quota exceeded", llm_provider="gemini", model="gemini/gemini-2.0-flash")
        return _mock_response("Recovered on key 2")

    with patch("app.llm_gateway.litellm.completion", side_effect=fake_real_failover):
        res = client.post("/chat", json={"session_id": "s-fail-real", "message": "Hello"})
        assert res.status_code == 200
        body = res.json()
        assert body["reply"] == "Recovered on key 2"
        assert body["metadata"]["key_id"] == "key_2"
        assert body["metadata"]["fallback_attempts"] == 1
        assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    # 2. X-Simulate-Failure: key_1=429 -> key_2 answers
    with patch("app.llm_gateway.litellm.completion", return_value=_mock_response("Sim failover ok")):
        res = client.post(
            "/chat",
            json={"session_id": "s-fail-sim", "message": "Test 429"},
            headers={"X-Simulate-Failure": "key_1=429"},
        )
        assert res.status_code == 200
        assert res.json()["metadata"]["key_id"] == "key_2"
        assert res.json()["metadata"]["fallback_attempts"] == 1

        # 3. X-Simulate-Failure: key_1=500 -> key_2 answers
        res500 = client.post(
            "/chat",
            json={"session_id": "s-fail-sim", "message": "Test 500"},
            headers={"X-Simulate-Failure": "key_1=500"},
        )
        assert res500.status_code == 200
        assert res500.json()["metadata"]["key_id"] == "key_2"
        assert res500.json()["metadata"]["fallback_attempts"] == 1

        # 4. X-Simulate-Failure: *=429 -> 503 and nothing added to history
        res_all = client.post(
            "/chat",
            json={"session_id": "s-all-down", "message": "Should fail"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert res_all.status_code == 503
        assert res_all.json() == {"detail": "all providers unavailable"}

    hist = client.get("/sessions/s-all-down/messages")
    assert hist.status_code == 404
    assert hist.json() == {"detail": "session not found"}


def test_simulation_header_ignored_when_disabled():
    settings = _make_settings(enable_failure_simulation=False)
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    with patch("app.llm_gateway.litellm.completion", return_value=_mock_response("Normal key 1")):
        res = client.post(
            "/chat",
            json={"session_id": "s-sim-off", "message": "Hello"},
            headers={"X-Simulate-Failure": "*=429"},
        )
        assert res.status_code == 200
        assert res.json()["metadata"]["key_id"] == "key_1"
        assert res.json()["metadata"]["fallback_attempts"] == 0


def test_sqlite_history_survives_restart(tmp_path):
    db_file = tmp_path / "app.db"
    db_url = f"sqlite:///{db_file}"
    settings = _make_settings(db_path=db_url)

    app1 = create_app(settings=settings, db=Database(db_url), gateway=LLMGateway(settings))
    client1 = TestClient(app1)
    with patch("app.llm_gateway.litellm.completion", return_value=_mock_response("Persisted reply")):
        res = client1.post("/chat", json={"session_id": "s-persist", "message": "Remember me"})
        assert res.status_code == 200

    # Simulate container/app restart with a fresh app & Database instance
    app2 = create_app(settings=settings, db=Database(db_url), gateway=LLMGateway(settings))
    client2 = TestClient(app2)
    hist = client2.get("/sessions/s-persist/messages")
    assert hist.status_code == 200
    assert len(hist.json()["messages"]) == 2
    assert hist.json()["messages"][0]["content"] == "Remember me"
    assert hist.json()["messages"][1]["content"] == "Persisted reply"


def test_simulation_bypasses_cache_and_handles_multiple_keys():
    settings = _make_settings(enable_failure_simulation=True, keys=3)
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    orig_completion = litellm.completion

    def completion_with_mock(*args, **kwargs):
        cached = litellm.cache.get_cache(*args, **kwargs)
        if kwargs.get("cache", {}).get("no-cache") or cached is None:
            kwargs["mock_response"] = "Cached or fresh answer"
        return orig_completion(*args, **kwargs)

    with patch("app.llm_gateway.litellm.completion", side_effect=completion_with_mock):
        # Populate cache first
        r1 = client.post("/chat", json={"session_id": "s-sim-cache", "message": "Repeat question"})
        assert r1.status_code == 200
        assert r1.json()["metadata"]["key_id"] == "key_1"

        # Now simulate key_1=502,key_2=503 on the exact same question -> must fail over to key_3
        r2 = client.post(
            "/chat",
            json={"session_id": "s-sim-cache-2", "message": "Repeat question"},
            headers={"X-Simulate-Failure": "key_1=502,key_2=503"},
        )
        assert r2.status_code == 200
        assert r2.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
        assert r2.json()["metadata"]["key_id"] == "key_3"
        assert r2.json()["metadata"]["fallback_attempts"] == 2


def test_cors_preflight_allows_simulate_failure_header():
    settings = _make_settings()
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    res = client.options(
        "/chat",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-simulate-failure",
        },
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_secrets_never_leak_and_invalid_sim_headers_ignored():
    settings = _make_settings(enable_failure_simulation=True)
    app = create_app(settings=settings, db=Database(settings.database_url), gateway=LLMGateway(settings))
    client = TestClient(app)

    with patch("app.llm_gateway.litellm.completion", return_value=_mock_response("Safe answer")):
        # Invalid status (200, abc) must be ignored and not trigger failover
        ok = client.post(
            "/chat",
            json={"session_id": "s-safe", "message": "Hi"},
            headers={"X-Simulate-Failure": "key_1=200,key_2=abc,invalid"},
        )
        assert ok.status_code == 200
        assert ok.json()["metadata"]["key_id"] == "key_1"
        assert ok.json()["metadata"]["fallback_attempts"] == 0
        assert "secret-key-" not in ok.text

        # All-fail 503 must never contain raw API keys
        err = client.post(
            "/chat",
            json={"session_id": "s-safe-fail", "message": "Hi"},
            headers={"X-Simulate-Failure": "*=503"},
        )
        assert err.status_code == 503
        assert err.json() == {"detail": "all providers unavailable"}
        assert "secret-key-" not in err.text


