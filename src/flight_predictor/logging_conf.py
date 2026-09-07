import logging
import sys
import build
import uuid
from contextvars import ContextVar

import structlog

## Correlation ID Context Variable: - ContextVar is safe for concurrent requests; each request has its own value
correlation_id_var: ContextVar[str] = ContextVar(
    "correlation_id", default = "no-request"
)

def get_correlation_id() -> str:
    return correlation_id_var.get()

def set_correlation_id(cid: str | None = None) -> str:
    cid = cid or str(uuid.uuid4())
    correlation_id_var.set(cid)
    return cid

def add_correlation_id(Logger, method, event_dict: dict) -> dict:
    event_dict["correlation_id"] = get_correlation_id()
    return event_dict

def setup_logging(level: str = "INFO", fmt: str= "json") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Shared precessor applied to every log event
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt='iso'),
        add_correlation_id,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info
    ]

    if fmt = "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormater.wrap_for_formatter
        ],
        logger_factory = structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use = True
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Quiet Noisy libraries:
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("xgboost").setLevel(logging.WARNING)

# FastAPI Middleware:
class correlationIDMiddleware:
    """
    ASGI middleware that assigns a unique correlation_id to every request.
    The ID is taken from the X-Correlation-ID header if present,
    or generated fresh as a UUID4.
    """
    def __init__(self, app) -> None:
        self.app = app

    asyn def __call__(self, scope, recieve, send) -> None:
        if scope["type"] == 'http':
            headers = dict(scope.get("headers",[]))
            cid_bytes = headers.get(b"x-correlation-id")
            cid = cid_bytes.decode() if cid_bytes else str(uuid.uuid4())
            set_correlation_id(cid)
        await self.app(scope, recieve, send)