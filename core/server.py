import logging
from typing import List, Optional
from importlib import metadata

import os  # Ensure os is imported if not already

from fastapi.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
from starlette.middleware import Middleware

from fastmcp import FastMCP
from fastmcp.server.auth.providers.google import GoogleProvider

from auth.oauth21_session_store import set_auth_provider, get_oauth21_session_store
from auth.google_auth import handle_auth_callback, start_auth_flow, check_client_secrets
from auth.mcp_session_middleware import MCPSessionMiddleware
from auth.oauth_responses import (
    create_error_response,
    create_success_response,
    create_server_error_response,
)
from auth.auth_info_middleware import AuthInfoMiddleware
from auth.scopes import SCOPES, get_current_scopes  # noqa
from starlette.concurrency import run_in_threadpool
from core.config import (
    USER_GOOGLE_EMAIL,
    get_transport_mode,
    set_transport_mode as _set_transport_mode,
    get_oauth_redirect_uri as get_oauth_redirect_uri_for_current_mode,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_auth_provider: Optional[GoogleProvider] = None
_legacy_callback_registered = False

session_middleware = Middleware(MCPSessionMiddleware)

# ============================================================================
# VERSION CACHING (Performance Optimization)
# ============================================================================
# Cache the package version at module load time (application startup)
# This avoids repeated file I/O operations when the health check endpoint
# is called frequently by load balancers or monitoring systems
#
# Health check endpoints can be called very frequently (e.g., every few seconds),
# so caching the version prevents unnecessary disk I/O on every request

try:
    # First try to get version from environment variable (set in Dockerfile)
    _cached_version = os.getenv("APP_VERSION")

    # If not in env, try to get from package metadata
    if not _cached_version:
        _cached_version = metadata.version("workspace-mcp")
except metadata.PackageNotFoundError:
    # If package metadata not found (e.g., running from source), use "dev"
    _cached_version = "dev"


# ============================================================================
# CREATE FASTMCP SERVER INSTANCE
# ============================================================================
# Initialize the FastMCP server. It will be mounted as a sub-app
# within our main Starlette application.
server = FastMCP(
    name="google_workspace",  # Server name (used in logs and metadata)
    auth=None,  # Auth provider set later in configure_server_for_http()
)


# ============================================================================
# ADD AUTHINFO MIDDLEWARE (FastMCP Middleware)
# ============================================================================
# AuthInfoMiddleware is a FastMCP middleware (not Starlette Middleware)
# It's added via server.add_middleware() to integrate with FastMCP's lifecycle.
#
# NOTE: API Key protection is now handled by a top-level Starlette middleware
# in main.py, which runs before this.
auth_info_middleware = AuthInfoMiddleware()
server.add_middleware(auth_info_middleware)


def set_transport_mode(mode: str):
    """Sets the transport mode for the server."""
    _set_transport_mode(mode)
    logger.info(f"Transport: {mode}")


def _ensure_legacy_callback_route() -> None:
    global _legacy_callback_registered
    if _legacy_callback_registered:
        return
    server.custom_route("/oauth2callback", methods=["GET"])(legacy_oauth2_callback)
    _legacy_callback_registered = True


def _create_external_oauth_provider(config, required_scopes: List[str]):
    """
    Create and configure ExternalOAuthProvider for external OAuth mode.

    This mode is used when:
    - OAuth is handled by frontend/backend (not MCP server)
    - Tokens are passed via X-Mcp-Google-Token header
    - Server operates in stateless mode (Cloud Run compatible)

    Args:
        config: OAuth configuration object
        required_scopes: List of required OAuth scopes

    Returns:
        Configured ExternalOAuthProvider instance
    """
    # Import here to avoid circular dependency
    # ExternalOAuthProvider may import from this module
    from auth.external_oauth_provider import ExternalOAuthProvider

    return ExternalOAuthProvider(
        client_id=config.client_id,  # OAuth client ID (for token validation)
        client_secret=config.client_secret,  # OAuth client secret (for token validation)
        base_url=config.get_oauth_base_url(),  # Base URL for OAuth redirects
        redirect_path=config.redirect_path,  # OAuth callback path
        required_scopes=required_scopes,  # Required OAuth scopes
    )


def _create_standard_oauth_provider(
    config, required_scopes: List[str]
) -> GoogleProvider:
    """
    Create and configure GoogleProvider for standard OAuth mode.

    This mode is used when:
    - MCP server handles OAuth flow internally
    - Users authenticate directly with the MCP server
    - Traditional MCP server deployment (not Cloud Run)

    Args:
        config: OAuth configuration object
        required_scopes: List of required OAuth scopes

    Returns:
        Configured GoogleProvider instance
    """
    return GoogleProvider(
        client_id=config.client_id,  # OAuth client ID
        client_secret=config.client_secret,  # OAuth client secret
        base_url=config.get_oauth_base_url(),  # Base URL for OAuth redirects
        redirect_path=config.redirect_path,  # OAuth callback path
        required_scopes=required_scopes,  # Required OAuth scopes
    )


def _configure_oauth21_provider(config) -> None:
    """
    Configure OAuth 2.1 authentication provider.

    This function handles both external and standard OAuth provider modes.
    It creates the appropriate provider and configures it for use.

    Args:
        config: OAuth configuration object

    Raises:
        Exception: If provider initialization fails
    """
    global _auth_provider

    # Validate that OAuth credentials are configured
    if not config.is_configured():
        logger.warning("OAuth 2.1 enabled but OAuth credentials not configured")
        return

    # Get required OAuth scopes for all enabled tools
    # Scopes define what permissions the OAuth token will have
    required_scopes: List[str] = sorted(get_current_scopes())

    # Determine OAuth provider type and create appropriate provider
    if config.is_external_oauth21_provider():
        # EXTERNAL Provider Mode: OAuth handled externally, tokens passed via header
        provider = _create_external_oauth_provider(config, required_scopes)
        server.auth = None  # Disable protocol-level auth
        logger.info(
            "OAuth 2.1 enabled with EXTERNAL provider mode - protocol-level auth disabled"
        )
        logger.info("Expecting Authorization bearer tokens in tool call headers")
    else:
        # STANDARD Provider Mode: MCP server handles OAuth flow internally
        provider = _create_standard_oauth_provider(config, required_scopes)
        server.auth = provider  # Enable protocol-level auth
        logger.info(
            "OAuth 2.1 enabled using FastMCP GoogleProvider with protocol-level auth"
        )

    # Set auth provider globally for token validation in middleware
    # This is separate from server.auth because:
    # - server.auth controls protocol-level OAuth flow
    # - _auth_provider is used for token validation in middleware
    set_auth_provider(provider)  # Set in oauth21_session_store for middleware use
    _auth_provider = provider  # Store globally for get_auth_provider() function


def _configure_oauth20_provider() -> None:
    """
    Configure OAuth 2.0 authentication provider (legacy mode).

    OAuth 2.0 is the legacy authentication mode used for backward compatibility.
    In this mode:
    - No protocol-level auth (server.auth = None)
    - Uses legacy callback route (/oauth2callback)
    - AuthInfoMiddleware handles token extraction and validation
    """
    global _auth_provider

    logger.info("OAuth 2.0 mode - Server will use legacy authentication.")
    server.auth = None  # Disable protocol-level auth
    _auth_provider = None  # Clear auth provider
    set_auth_provider(None)  # Clear auth provider in session store
    _ensure_legacy_callback_route()  # Register /oauth2callback route


def configure_server_for_http():
    """
    Configures the authentication provider for HTTP transport.

    This function MUST be called BEFORE server.run() to set up OAuth authentication.
    It orchestrates the configuration process by delegating to helper functions.

    Configuration flow:
    1. Check transport mode (only configures for streamable-http)
    2. Load OAuth configuration
    3. Determine OAuth version (2.1 vs 2.0)
    4. Configure appropriate auth provider
    """
    global _auth_provider

    # Only configure OAuth for HTTP transport (streamable-http)
    # For stdio transport, OAuth is handled differently (not via HTTP)
    transport_mode = get_transport_mode()
    if transport_mode != "streamable-http":
        return

    # Load OAuth configuration from centralized config system
    # Import here to avoid circular dependency
    # oauth_config may import from this module
    from auth.oauth_config import get_oauth_config

    config = get_oauth_config()

    # Determine OAuth version and configure appropriate provider
    if config.is_oauth21_enabled():
        # OAuth 2.1 mode (modern, recommended)
        try:
            _configure_oauth21_provider(config)
        except Exception as exc:
            # Log and re-raise exceptions during provider initialization
            # This ensures startup fails fast if OAuth configuration is invalid
            logger.error(
                "Failed to initialize OAuth 2.1 provider: %s", exc, exc_info=True
            )
            raise
    else:
        # OAuth 2.0 mode (legacy, backward compatibility)
        _configure_oauth20_provider()


def get_auth_provider() -> Optional[GoogleProvider]:
    """Gets the global authentication provider instance."""
    return _auth_provider


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================
# This endpoint is intentionally UNPROTECTED (no API key required)
# This is an acceptable security tradeoff for operational visibility
#
# Why Unprotected:
#   - Used by load balancers, monitoring systems, and health checks
#   - Needs to work without authentication for easier setup and debugging
#   - Doesn't expose sensitive data (only returns status, version, transport)
#
# Security Considerations:
#   - Only returns non-sensitive metadata (status, version, transport mode)
#   - Does NOT expose user data, tokens, or internal state
#   - Can be rate-limited at infrastructure level if needed
#
# The APIKeyFastMCPMiddleware automatically skips validation for /health endpoint
@server.custom_route("/health", methods=["GET"])
async def health_check(request: Request):
    """
    Health check endpoint for monitoring and load balancer health checks.

    Returns basic server information without requiring authentication.
    This endpoint is intentionally unprotected for operational visibility.

    Returns:
        JSONResponse with server status, version, and transport mode
    """
    # Use cached version (loaded at application startup)
    # This avoids file I/O on every health check request
    # The version is cached in _cached_version at module load time
    version = _cached_version

    # Return health check response
    # This response is consumed by:
    #   - Load balancers (to determine if server is healthy)
    #   - Monitoring systems (to track server availability)
    #   - Health check scripts (to verify server is running)
    return JSONResponse(
        {
            "status": "healthy",  # Server is operational
            "service": "workspace-mcp",  # Service identifier
            "version": version,  # Package version (or "dev" if running from source)
            "transport": get_transport_mode(),  # Current transport mode (streamable-http, stdio, etc.)
        }
    )


async def legacy_oauth2_callback(request: Request) -> HTMLResponse:
    state = request.query_params.get("state")
    code = request.query_params.get("code")
    error = request.query_params.get("error")

    if error:
        msg = (
            f"Authentication failed: Google returned an error: {error}. State: {state}."
        )
        logger.error(msg)
        return create_error_response(msg)

    if not code:
        msg = "Authentication failed: No authorization code received from Google."
        logger.error(msg)
        return create_error_response(msg)

    try:
        error_message = check_client_secrets()
        if error_message:
            return create_server_error_response(error_message)

        logger.info(f"OAuth callback: Received code (state: {state}).")

        # =====================================================================
        # STEP 1: Extract MCP Session ID from Multiple Sources
        # =====================================================================
        # Priority order:
        # 1. request.state.session_id (set by MCPSessionMiddleware from headers/FastMCP)
        # 2. OAuth state parameter (session_id stored when flow started)
        #
        # The middleware processes /oauth2callback but may not have extracted
        # the session ID from the OAuth state parameter. We extract it here
        # to ensure we can link the OAuth callback to the original MCP session.
        mcp_session_id = None

        # Try to get session ID from request.state (set by middleware)
        if hasattr(request, "state") and hasattr(request.state, "session_id"):
            mcp_session_id = request.state.session_id
            logger.debug(f"Found MCP session ID from request.state: {mcp_session_id}")

        # If not found, try to extract from OAuth state parameter
        # (without consuming it - handle_auth_callback will validate and consume)
        if not mcp_session_id and state:
            try:
                store = get_oauth21_session_store()
                # Use public method to peek at state without consuming it
                # This maintains proper encapsulation and avoids accessing private members
                state_info = store.peek_oauth_state(state)
                if state_info:
                    mcp_session_id = state_info.get("session_id")
                    if mcp_session_id:
                        logger.debug(
                            f"Found MCP session ID from OAuth state: {mcp_session_id}"
                        )
            except Exception as e:
                logger.debug(f"Could not extract session ID from OAuth state: {e}")
                # Continue - handle_auth_callback will handle state validation

        # =====================================================================
        # STEP 2: Exchange Authorization Code for Credentials (Non-Blocking)
        # =====================================================================
        # handle_auth_callback is a synchronous function that performs blocking
        # I/O operations (HTTP requests to Google's OAuth token endpoint and
        # userinfo endpoint). To avoid blocking the event loop, we run it in
        # a thread pool using Starlette's run_in_threadpool utility.
        #
        # This ensures that:
        # - The server can process other requests while waiting for Google's response
        # - The event loop is not blocked by synchronous HTTP calls
        # - Performance remains good under load
        verified_user_id, credentials = await run_in_threadpool(
            handle_auth_callback,
            scopes=get_current_scopes(),
            authorization_response=str(request.url),
            redirect_uri=get_oauth_redirect_uri_for_current_mode(),
            session_id=mcp_session_id,
        )

        logger.info(
            f"OAuth callback: Successfully authenticated user: {verified_user_id}."
        )

        # =====================================================================
        # STEP 3: Store Credentials in Session Store (Critical - Must Succeed)
        # =====================================================================
        # Storing credentials is a critical step. If this fails, the authentication
        # process cannot be considered successful because:
        # 1. The user will be told authentication succeeded
        # 2. But their credentials won't be available for subsequent requests
        # 3. They'll be forced to authenticate again, causing confusion
        # 4. The server state becomes inconsistent with the response sent to client
        #
        # Therefore, if credential storage fails, we MUST return an error response
        # instead of a success response. This ensures the client knows the operation
        # failed and can retry, rather than thinking it succeeded when it didn't.
        try:
            store = get_oauth21_session_store()

            store.store_session(
                user_email=verified_user_id,
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_uri=credentials.token_uri,
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                scopes=credentials.scopes,
                expiry=credentials.expiry,
                session_id=f"google-{state}",
                mcp_session_id=mcp_session_id,
            )
            logger.info(
                f"Stored Google credentials in OAuth 2.1 session store for {verified_user_id}"
            )
        except Exception as e:
            # Log detailed error for debugging (includes full exception traceback)
            # This helps developers diagnose storage issues without exposing details to clients
            logger.error(
                f"Failed to store credentials in OAuth 2.1 session store for {verified_user_id}: {e}",
                exc_info=True,
            )
            # Return error response - authentication cannot be considered successful
            # if credentials cannot be stored, as they won't be available for future requests
            # Return generic error message to client (prevents information disclosure)
            # Never expose internal details like file paths, library names, or stack traces
            return create_server_error_response(
                "An error occurred while saving your authentication credentials. Please try again."
            )

        # =====================================================================
        # STEP 4: Return Success Response
        # =====================================================================
        # Only return success if ALL steps completed successfully:
        # 1. Authorization code exchanged for credentials ✓
        # 2. Credentials stored in session store ✓
        # This ensures the server state is consistent with the response sent to client
        return create_success_response(verified_user_id)
    except Exception as e:
        # Log detailed error for debugging (includes full exception traceback)
        # This helps developers diagnose issues without exposing details to clients
        logger.error(f"Error processing OAuth callback: {str(e)}", exc_info=True)
        # Return generic error message to client (prevents information disclosure)
        # Never expose internal details like file paths, library names, or stack traces
        return create_server_error_response(
            "An error occurred while processing the OAuth callback. Please try again."
        )


@server.tool()
async def start_google_auth(
    service_name: str, user_google_email: str = USER_GOOGLE_EMAIL
) -> str:
    """
    Manually initiate Google OAuth authentication flow.

    NOTE: This tool should typically NOT be called directly. The authentication system
    automatically handles credential checks and prompts for authentication when needed.
    Only use this tool if:
    1. You need to re-authenticate with different credentials
    2. You want to proactively authenticate before using other tools
    3. The automatic authentication flow failed and you need to retry

    In most cases, simply try calling the Google Workspace tool you need - it will
    automatically handle authentication if required.
    """
    if not user_google_email:
        raise ValueError("user_google_email must be provided.")

    # =====================================================================
    # STEP 1: Validate OAuth Client Secrets Configuration
    # =====================================================================
    # Check if OAuth client secrets are properly configured
    # This is required for the OAuth flow to work
    error_message = check_client_secrets()
    if error_message:
        # Raise exception instead of returning error string
        # This allows FastMCP to convert it to an appropriate error response
        # and enables programmatic error handling by clients
        logger.error(f"OAuth client secrets validation failed: {error_message}")
        raise RuntimeError(f"Authentication configuration error: {error_message}")

    # =====================================================================
    # STEP 2: Start OAuth Authentication Flow
    # =====================================================================
    # Initiate the OAuth flow which will redirect the user to Google's
    # authentication page. The flow will complete via the /oauth2callback endpoint.
    try:
        auth_message = await start_auth_flow(
            user_google_email=user_google_email,
            service_name=service_name,
            redirect_uri=get_oauth_redirect_uri_for_current_mode(),
        )
        return auth_message
    except Exception as e:
        # Log detailed error for debugging (includes full exception traceback)
        # This helps developers diagnose issues without exposing details to clients
        logger.error(f"Failed to start Google authentication flow: {e}", exc_info=True)
        # Raise exception instead of returning error string
        # This allows FastMCP to convert it to an appropriate error response
        # and enables programmatic error handling by clients
        # Use generic error message to prevent information disclosure
        # Never expose internal details like file paths, library names, or stack traces
        raise RuntimeError(
            "An unexpected error occurred while starting the authentication flow. Please try again."
        ) from e
