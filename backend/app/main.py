from contextlib import asynccontextmanager
import logging
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import Settings, load_settings
from app.db import Database
from app.llm_gateway import AllProvidersUnavailableError, LLMGateway
from app.schemas import (
    ChatMetadata,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    SessionMessagesResponse,
)

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    db: Optional[Database] = None,
    gateway: Optional[LLMGateway] = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    resolved_settings = settings or load_settings()
    resolved_db = db or Database(resolved_settings.database_url)
    resolved_gateway = gateway or LLMGateway(resolved_settings)

    # Ensure schema exists even if TestClient is instantiated without lifespan context.
    resolved_db.init_schema()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_db.init_schema()
        yield

    app = FastAPI(
        title="LLM Chatbot Gateway",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.state.settings = resolved_settings
    app.state.db = resolved_db
    app.state.gateway = resolved_gateway

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved_settings.allowed_origins or ["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": jsonable_encoder(exc.errors())},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "internal server error"},
        )

    def get_db() -> Database:
        return app.state.db

    def get_gateway() -> LLMGateway:
        return app.state.gateway

    def get_settings() -> Settings:
        return app.state.settings

    @app.get("/health", response_model=HealthResponse)
    def health(
        db_inst: Database = Depends(get_db),
        gw_inst: LLMGateway = Depends(get_gateway),
    ) -> HealthResponse:
        try:
            db_ok = db_inst.ping()
        except Exception as exc:
            logger.error("Health check database ping failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            )
        if not db_ok:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="database unavailable",
            )
        if len(gw_inst.key_pool) == 0:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no api keys configured",
            )
        return HealthResponse(
            status="ok",
            keys_configured=len(gw_inst.key_pool),
            cache_enabled=gw_inst.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: Optional[str] = Header(default=None, alias="X-Simulate-Failure"),
        db_inst: Database = Depends(get_db),
        gw_inst: LLMGateway = Depends(get_gateway),
        cfg: Settings = Depends(get_settings),
    ) -> ChatResponse:
        stored_system_prompt = db_inst.get_session_system_prompt(payload.session_id)
        active_system_prompt = (
            payload.system_prompt
            if payload.system_prompt is not None
            else (stored_system_prompt or cfg.default_system_prompt)
        )

        history = db_inst.get_llm_history(payload.session_id)
        llm_messages = [{"role": "system", "content": active_system_prompt}]
        llm_messages.extend(history)
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = await run_in_threadpool(
                gw_inst.complete,
                llm_messages,
                x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="all providers unavailable",
            )

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,  # type: ignore[arg-type]
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        db_inst.save_turn(
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=result.reply,
            metadata=metadata.model_dump(),
            system_prompt=payload.system_prompt,
        )

        logger.info(
            "chat completed session_id=%s source=%s model=%s key_id=%s fallback_attempts=%d latency_ms=%d",
            payload.session_id,
            metadata.resolution_source,
            metadata.model,
            metadata.key_id,
            metadata.fallback_attempts,
            metadata.latency_ms,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=SessionMessagesResponse,
        response_model_exclude_none=True,
    )
    def get_session_messages(
        session_id: str,
        db_inst: Database = Depends(get_db),
    ) -> SessionMessagesResponse:
        messages = db_inst.get_session_messages(session_id)
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="session not found",
            )
        return SessionMessagesResponse(session_id=session_id, messages=messages)  # type: ignore[arg-type]

    return app


app = create_app()
