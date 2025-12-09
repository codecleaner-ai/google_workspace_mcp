"""
API Key Middleware (Starlette) for Google Workspace MCP Server

This middleware validates the X-API-Key header for server-level authentication.
It is implemented as a Starlette BaseHTTPMiddleware, which ensures it runs
at the HTTP level before the request is passed to any application logic,
including the FastMCP framework.

This provides the outermost layer of security, guaranteeing that no unauthorized
request can reach the MCP server.
"""

import os
import logging
import secrets
from typing import Callable, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Public endpoints that don't require API key validation
PUBLIC_PATHS = (
    ("/oauth2callback", True),  # Match /oauth2callback and /oauth2callback?code=...
    ("/.well-known/", True),  # Match /.well-known/* (OAuth discovery)
    ("/health", False),  # Match /health exactly
)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Starlette Middleware to validate the X-API-Key header for server-level authentication.

    This middleware provides the first line of defense for the server. It inspects
    every incoming HTTP request and validates the presence and correctness of the
    X-API-Key header before forwarding the request to the main application.

    Security Features:
    - Validates X-API-Key against GOOGLE_MCP_SERVER_API_KEY environment variable.
    - Returns HTTP 401 Unauthorized for missing or invalid API keys.
    - **Fail-closed security**: Any unexpected errors result in access denial.
    - Skips validation for designated public endpoints (/health, /oauth2callback).
    - Can be disabled for development if the API key is not set.
    - Uses constant-time comparison (`secrets.compare_digest()`) to prevent timing attacks.
    """

    def __init__(self, app: ASGIApp):
        """
        Initialize the API key middleware.
        """
        super().__init__(app)
        api_key = os.getenv("GOOGLE_MCP_SERVER_API_KEY")
        if api_key:
            api_key = api_key.strip()

        self.api_key = api_key
        self.enabled = self.api_key is not None

        if self.enabled:
            logger.info("API key protection enabled (Starlette middleware)")
        else:
            logger.warning(
                "API key protection DISABLED - server is unprotected! (Starlette middleware)"
            )

    def _is_public_endpoint(self, path: str) -> bool:
        """Checks if the request path is a public endpoint."""
        return any(
            path.startswith(p) if is_prefix else path == p
            for p, is_prefix in PUBLIC_PATHS
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """
        Dispatch method to process the request.
        """
        if not self.enabled or self._is_public_endpoint(request.url.path):
            return await call_next(request)

        try:
            api_key_header = request.headers.get("X-API-Key")

            if not api_key_header:
                logger.warning(f"Missing X-API-Key header for {request.url.path}")
                return JSONResponse(
                    status_code=401,
                    content={
                        "detail": "Missing X-API-Key header. API key is required."
                    },
                )

            if not secrets.compare_digest(api_key_header, self.api_key):
                logger.warning(f"Invalid API key provided for {request.url.path}")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid API key."},
                )

            logger.info(f"API key validated for {request.url.path}")
            return await call_next(request)

        except Exception as e:
            logger.error(f"Unexpected error in APIKeyMiddleware: {e}", exc_info=True)
            # Fail-closed: Deny access on any unexpected error
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error during authentication."},
            )
