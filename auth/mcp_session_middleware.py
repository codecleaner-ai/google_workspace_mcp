"""
MCP Session Middleware

This middleware intercepts MCP requests and sets the session context
for use by tool functions.
"""

import logging
from typing import Callable, Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from auth.oauth21_session_store import (
    SessionContext,
    SessionContextManager,
    extract_session_from_headers,
    get_oauth21_session_store,
)
# OAuth 2.1 is now handled by FastMCP auth

logger = logging.getLogger(__name__)


class MCPSessionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that extracts session information from requests and makes it
    available to MCP tool functions via context variables.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Any:
        """
        Process request and set session context.

        This method extracts session information from various sources and makes it
        available to MCP tool functions via context variables. It handles both
        MCP requests and OAuth callbacks.
        """

        logger.debug(
            f"MCPSessionMiddleware processing request: {request.method} {request.url.path}"
        )

        # Process MCP paths and OAuth callback path
        # OAuth callback needs session context to link authentication to MCP session
        if (
            not request.url.path.startswith("/mcp")
            and request.url.path != "/oauth2callback"
        ):
            logger.debug(f"Skipping non-MCP/OAuth path: {request.url.path}")
            return await call_next(request)

        # =====================================================================
        # STEP 1: Build Session Context (with error handling)
        # =====================================================================
        # We build the session context in a try-except block. If building fails,
        # we proceed without session context. This ensures call_next is only
        # called once and prevents double-processing of requests.
        session_context = None

        try:
            # Extract session information
            headers = dict(request.headers)
            session_id = extract_session_from_headers(headers)

            # Try to get OAuth 2.1 auth context from FastMCP
            auth_context = None
            user_email = None
            mcp_session_id = None

            # Check for FastMCP auth context
            if hasattr(request.state, "auth"):
                auth_context = request.state.auth
                # Extract user email from auth claims if available
                if hasattr(auth_context, "claims") and auth_context.claims:
                    user_email = auth_context.claims.get("email")

            # Check for FastMCP session ID (from streamable HTTP transport)
            if hasattr(request.state, "session_id"):
                mcp_session_id = request.state.session_id
                logger.debug(f"Found FastMCP session ID: {mcp_session_id}")

            # For OAuth callbacks, also try to extract session ID from OAuth state parameter
            # This links the OAuth callback to the original MCP session that initiated the flow
            if not mcp_session_id and request.url.path == "/oauth2callback":
                state = request.query_params.get("state")
                if state:
                    try:
                        store = get_oauth21_session_store()
                        # Use public method to peek at state without consuming it
                        # This maintains proper encapsulation and avoids accessing private members
                        state_info = store.peek_oauth_state(state)
                        if state_info:
                            mcp_session_id = state_info.get("session_id")
                            if mcp_session_id:
                                logger.debug(
                                    f"Found MCP session ID from OAuth state in middleware: {mcp_session_id}"
                                )
                                # Set in request.state so callback handler can use it
                                if not hasattr(request.state, "session_id"):
                                    request.state.session_id = mcp_session_id
                    except (AttributeError, KeyError, TypeError) as e:
                        # Specific exceptions for OAuth state store access
                        # AttributeError: store doesn't have peek_oauth_state method
                        # KeyError: state_info is not a dict (shouldn't happen with public method)
                        # TypeError: state_info is not a dict (shouldn't happen with public method)
                        logger.debug(
                            f"Could not extract session ID from OAuth state in middleware: {e}"
                        )
                        # Continue - callback handler will also try to extract it
                    except Exception as e:
                        # Catch-all for unexpected errors during state extraction
                        logger.warning(
                            f"Unexpected error extracting session ID from OAuth state: {e}",
                            exc_info=True,
                        )
                        # Continue - callback handler will also try to extract it

            # NOTE: JWT token extraction removed for security reasons
            # Previously, this code attempted to decode JWT tokens without signature
            # verification, which is a critical security vulnerability. An attacker could
            # create a fake token with any payload (e.g., admin privileges) and the server
            # would trust it completely.
            #
            # If JWT token extraction is needed, it should:
            # 1. Use proper signature verification (see auth_info_middleware._verify_jwt_token)
            # 2. Verify the token against the appropriate public key (JWKS)
            # 3. Verify expiration and other claims
            #
            # For now, user_email is extracted from:
            # - FastMCP auth context (verified by FastMCP)
            # - Other authenticated sources (handled by AuthInfoMiddleware)

            # Build session context
            if session_id or auth_context or user_email or mcp_session_id:
                # Create session ID hierarchy: explicit session_id > Google user session > FastMCP session
                effective_session_id = session_id
                if not effective_session_id and user_email:
                    effective_session_id = f"google_{user_email}"
                elif not effective_session_id and mcp_session_id:
                    effective_session_id = mcp_session_id

                session_context = SessionContext(
                    session_id=effective_session_id,
                    user_id=user_email
                    or (auth_context.user_id if auth_context else None),
                    auth_context=auth_context,
                    request=request,
                    metadata={
                        "path": request.url.path,
                        "method": request.method,
                        "user_email": user_email,
                        "mcp_session_id": mcp_session_id,
                    },
                )

                logger.debug(
                    f"MCP request with session: session_id={session_context.session_id}, "
                    f"user_id={session_context.user_id}, path={request.url.path}"
                )

        except Exception as e:
            # Error building session context - log and continue without it
            # This ensures we don't block the request, but we won't have session context
            logger.error(
                f"Error building session context in MCP session middleware: {e}",
                exc_info=True,
            )
            # session_context remains None - request will proceed without session context

        # =====================================================================
        # STEP 2: Process Request with Session Context
        # =====================================================================
        # This is outside the try-except block to ensure call_next is only called once.
        # If an exception occurs during request processing, it will propagate up
        # to the framework's error handler, not be caught here.
        with SessionContextManager(session_context):
            response = await call_next(request)
            return response
