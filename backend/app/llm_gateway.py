from dataclasses import dataclass
import logging
import os
import time
from typing import Any, Dict, List, Optional

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
import litellm.caching.caching_handler as _caching_handler
from litellm.caching import Cache
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    BudgetExceededError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from app.config import KeyEntry, Settings

logger = logging.getLogger(__name__)

SIMULATABLE_STATUSES = {429, 500, 502, 503}


def _patch_litellm_cache_streaming_check() -> None:
    """Guard against optional proxy audio imports missing in slim environments."""
    orig = _caching_handler._should_defer_streaming_cache_hit_callbacks

    def _safe_should_defer(*, cached_result: Any) -> bool:
        try:
            return orig(cached_result=cached_result)
        except ImportError:
            from litellm.utils import CustomStreamWrapper

            return isinstance(cached_result, CustomStreamWrapper)

    _caching_handler._should_defer_streaming_cache_hit_callbacks = _safe_should_defer


_patch_litellm_cache_streaming_check()


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails to produce a response."""


@dataclass(frozen=True)
class GatewayResult:
    """Normalized response returned by the LiteLLM gateway."""

    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: Optional[str]
    fallback_attempts: int


def parse_simulate_failure_header(header_value: Optional[str]) -> Dict[str, int]:
    """Parse X-Simulate-Failure header into a map of target -> HTTP status code."""
    if not header_value:
        return {}
    rules: Dict[str, int] = {}
    for raw_pair in header_value.split(","):
        pair = raw_pair.strip()
        if not pair or "=" not in pair:
            continue
        target, status_str = pair.split("=", 1)
        target = target.strip()
        status_str = status_str.strip()
        if not target or not status_str.isdigit():
            continue
        status_code = int(status_str)
        if status_code in SIMULATABLE_STATUSES:
            rules[target] = status_code
    return rules


def _build_simulated_exception(status_code: int, model: str, key_id: str) -> Exception:
    provider = model.split("/", 1)[0] if "/" in model else "custom"
    message = f"Simulated {status_code} failure for {key_id}"
    if status_code == 429:
        return RateLimitError(message=message, llm_provider=provider, model=model)
    if status_code == 500:
        return InternalServerError(message=message, llm_provider=provider, model=model)
    if status_code == 502:
        return BadGatewayError(message=message, llm_provider=provider, model=model)
    return ServiceUnavailableError(message=message, llm_provider=provider, model=model)


def _is_failover_exception(exc: Exception) -> bool:
    if isinstance(
        exc,
        (
            RateLimitError,
            InternalServerError,
            BadGatewayError,
            ServiceUnavailableError,
            Timeout,
            APIConnectionError,
            BudgetExceededError,
            AuthenticationError,
            PermissionDeniedError,
        ),
    ):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        if status is None or status == 429 or status >= 500:
            return True
    lowered = str(exc).lower()
    return any(
        token in lowered
        for token in ("429", "rate limit", "quota", "exhausted", "unavailable", "500", "502", "503", "timeout")
    )


def _extract_usage_tokens(response: Any, model: str, messages: List[Dict[str, str]], reply: str) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    if usage is not None:
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            completion_tokens = getattr(usage, "completion_tokens", None)

    if prompt_tokens is None:
        try:
            prompt_tokens = int(litellm.token_counter(model=model, messages=messages))
        except Exception:
            prompt_tokens = 0
    if completion_tokens is None:
        try:
            completion_tokens = int(litellm.token_counter(model=model, text=reply))
        except Exception:
            completion_tokens = 0

    return max(0, int(prompt_tokens)), max(0, int(completion_tokens))


def _extract_reply_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "".join(parts).strip()
    return ""


class LLMGateway:
    """LiteLLM wrapper providing ordered key-pool failover and response caching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.key_pool: List[KeyEntry] = list(settings.key_pool)
        self.cache_enabled = True
        self._init_cache()

    def _init_cache(self) -> None:
        litellm.suppress_debug_info = True
        litellm.set_verbose = False
        if self.settings.cache_type == "disk":
            os.makedirs(self.settings.cache_dir, exist_ok=True)
            litellm.cache = Cache(type="disk", disk_cache_dir=self.settings.cache_dir)
        else:
            litellm.cache = Cache(type="local")
        litellm.enable_cache()

    def complete(
        self,
        messages: List[Dict[str, str]],
        simulate_failure_header: Optional[str] = None,
    ) -> GatewayResult:
        """Execute chat completion with caching and ordered key-pool failover."""
        if not self.key_pool:
            raise AllProvidersUnavailableError("all providers unavailable")

        simulation_rules: Dict[str, int] = {}
        if self.settings.enable_failure_simulation and simulate_failure_header:
            simulation_rules = parse_simulate_failure_header(simulate_failure_header)

        has_active_simulation = "*" in simulation_rules or any(
            entry.key_id in simulation_rules for entry in self.key_pool
        )

        start = time.perf_counter()
        attempts_failed = 0

        for entry in self.key_pool:
            try:
                simulated_status = simulation_rules.get(entry.key_id) or simulation_rules.get("*")
                if simulated_status is not None:
                    raise _build_simulated_exception(simulated_status, entry.model, entry.key_id)

                call_kwargs: Dict[str, Any] = {
                    "model": entry.model,
                    "messages": messages,
                    "api_key": entry.api_key,
                    "num_retries": 0,
                    "timeout": self.settings.request_timeout_seconds,
                }
                if has_active_simulation:
                    # Ensure failover tests exercise live keys rather than returning a cached entry.
                    call_kwargs["cache"] = {"no-cache": True}

                response = litellm.completion(**call_kwargs)
                reply = _extract_reply_text(response)
                if not reply:
                    raise ServiceUnavailableError(
                        message=f"Empty response from {entry.key_id}",
                        llm_provider=entry.model.split("/", 1)[0] if "/" in entry.model else "custom",
                        model=entry.model,
                    )

                elapsed_ms = max(1, int(round((time.perf_counter() - start) * 1000)))
                hidden_params = getattr(response, "_hidden_params", None) or {}
                is_cache_hit = bool(hidden_params.get("cache_hit", False))
                input_tokens, output_tokens = _extract_usage_tokens(response, entry.model, messages, reply)

                if is_cache_hit:
                    return GatewayResult(
                        reply=reply,
                        resolution_source="LLM_CACHE",
                        model=entry.model,
                        latency_ms=elapsed_ms,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        key_id=None,
                        fallback_attempts=0,
                    )

                return GatewayResult(
                    reply=reply,
                    resolution_source="LIVE_INFERENCE",
                    model=entry.model,
                    latency_ms=elapsed_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=entry.key_id,
                    fallback_attempts=attempts_failed,
                )

            except Exception as exc:
                if _is_failover_exception(exc):
                    attempts_failed += 1
                    logger.warning(
                        "Key %s failed on model %s (%s); failing over (attempt %d)",
                        entry.key_id,
                        entry.model,
                        type(exc).__name__,
                        attempts_failed,
                    )
                    continue
                raise

        raise AllProvidersUnavailableError("all providers unavailable")
