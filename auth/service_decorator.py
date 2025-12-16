import hashlib
import inspect
import json
import logging
import re
import time
from functools import wraps
from socket import timeout as SocketTimeout
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
from fastmcp.server.dependencies import get_access_token, get_context, get_http_headers
from auth.google_auth import get_authenticated_google_service, GoogleAuthenticationError
from auth.oauth21_session_store import (
    get_auth_provider,
    get_oauth21_session_store,
    ensure_session_from_access_token,
    create_credentials_from_token_stateless,
)
from auth.oauth_config import is_oauth21_enabled, get_oauth_config
from core.context import set_fastmcp_session_id
from auth.scopes import (
    GMAIL_READONLY_SCOPE,
    GMAIL_SEND_SCOPE,
    GMAIL_COMPOSE_SCOPE,
    GMAIL_MODIFY_SCOPE,
    GMAIL_LABELS_SCOPE,
    DRIVE_READONLY_SCOPE,
    DRIVE_FILE_SCOPE,
    DOCS_READONLY_SCOPE,
    DOCS_WRITE_SCOPE,
    CALENDAR_READONLY_SCOPE,
    CALENDAR_EVENTS_SCOPE,
    SHEETS_READONLY_SCOPE,
    SHEETS_WRITE_SCOPE,
    CHAT_READONLY_SCOPE,
    CHAT_WRITE_SCOPE,
    CHAT_SPACES_SCOPE,
    FORMS_BODY_SCOPE,
    FORMS_BODY_READONLY_SCOPE,
    FORMS_RESPONSES_READONLY_SCOPE,
    SLIDES_SCOPE,
    SLIDES_READONLY_SCOPE,
    TASKS_SCOPE,
    TASKS_READONLY_SCOPE,
    CUSTOM_SEARCH_SCOPE,
)

logger = logging.getLogger(__name__)


# Authentication helper functions
def _get_auth_context(
    tool_name: str,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get authentication context from FastMCP.

    Returns:
        Tuple of (authenticated_user, auth_method, mcp_session_id)
    """
    try:
        ctx = get_context()
        if not ctx:
            logger.debug(f"[{tool_name}] get_context() returned None")
            return None, None, None

        authenticated_user = ctx.get_state("authenticated_user_email")
        auth_method = ctx.get_state("authenticated_via")
        mcp_session_id = ctx.session_id if hasattr(ctx, "session_id") else None

        # CRITICAL FIX: If authenticated_user is None but stateless_mode is set,
        # try to get user_email as fallback (middleware sets both)
        if not authenticated_user:
            stateless_mode = ctx.get_state("stateless_mode")
            if stateless_mode:
                # Try to get user_email as fallback (middleware sets this in stateless mode)
                authenticated_user = ctx.get_state("user_email")
                if authenticated_user:
                    logger.debug(
                        f"[{tool_name}] Retrieved authenticated_user from user_email fallback (stateless mode)"
                    )
                # Also try to get auth_method from stateless mode
                if not auth_method:
                    auth_method = ctx.get_state("authenticated_via")
                    if not auth_method:
                        # Check if access_token exists (indicates authentication happened)
                        access_token = ctx.get_state("access_token")
                        if access_token:
                            auth_method = "x_google_access_token"
                            logger.debug(
                                f"[{tool_name}] Detected auth_method from access_token presence (stateless mode)"
                            )

        if mcp_session_id:
            set_fastmcp_session_id(mcp_session_id)

        logger.debug(
            f"[{tool_name}] Auth from middleware: {authenticated_user} via {auth_method} (session: {mcp_session_id[:8] if mcp_session_id else 'none'})"
        )
        return authenticated_user, auth_method, mcp_session_id

    except (AttributeError, KeyError, TypeError) as e:
        # Catch specific exceptions that can occur during context retrieval:
        # - AttributeError: ctx doesn't have expected attributes (session_id, get_state)
        # - KeyError: get_state() key doesn't exist (though it should return None)
        # - TypeError: Invalid type for context operations
        logger.debug(f"[{tool_name}] Could not get FastMCP context: {e}")
        return None, None, None
    except Exception as e:
        # Catch-all for truly unexpected errors - log with full context
        # This should rarely happen, but we want to know if it does
        logger.error(
            f"[{tool_name}] Unexpected error getting FastMCP context: {e}",
            exc_info=True,
        )
        return None, None, None


def _detect_oauth_version(
    authenticated_user: Optional[str], mcp_session_id: Optional[str], tool_name: str
) -> bool:
    """
    Detect whether to use OAuth 2.1 based on configuration and context.

    Returns:
        True if OAuth 2.1 should be used, False otherwise
    """
    # CRITICAL: Check stateless_mode FIRST (before global flag check)
    # If X-Google-Access-Token header is present, middleware sets stateless_mode=True
    # Stateless mode REQUIRES OAuth 2.1, so this check takes precedence
    try:
        ctx = get_context()
        if ctx:
            stateless_mode = ctx.get_state("stateless_mode")
            if stateless_mode:
                logger.info(
                    f"[{tool_name}] OAuth 2.1 mode detected via stateless_mode flag (X-Google-Access-Token header present)"
                )
                return True
    except Exception as e:
        logger.debug(
            f"[{tool_name}] Could not check stateless_mode flag in context: {e}"
        )

    # If stateless_mode is not set, check global OAuth 2.1 flag
    if not is_oauth21_enabled():
        return False

    # When OAuth 2.1 is enabled globally, ALWAYS use OAuth 2.1 for authenticated users
    if authenticated_user:
        logger.info(
            f"[{tool_name}] OAuth 2.1 mode: Using OAuth 2.1 for authenticated user '{authenticated_user}'"
        )
        return True

    # Only use version detection for unauthenticated requests
    config = get_oauth_config()
    request_params = {}
    if mcp_session_id:
        request_params["session_id"] = mcp_session_id

    oauth_version = config.detect_oauth_version(request_params)
    use_oauth21 = oauth_version == "oauth21"
    logger.info(
        f"[{tool_name}] OAuth version detected: {oauth_version}, will use OAuth 2.1: {use_oauth21}"
    )
    return use_oauth21


def _update_email_in_args(args: tuple, index: int, new_email: str) -> tuple:
    """Update email at specific index in args tuple."""
    if index < len(args):
        args_list = list(args)
        args_list[index] = new_email
        return tuple(args_list)
    return args


def _override_oauth21_user_email(
    use_oauth21: bool,
    authenticated_user: Optional[str],
    current_user_email: str,
    args: tuple,
    kwargs: dict,
    param_names: List[str],
    tool_name: str,
    service_type: str = "",
) -> Tuple[str, tuple]:
    """
    Override user_google_email with authenticated user when using OAuth 2.1.

    Returns:
        Tuple of (updated_user_email, updated_args)
    """
    if not (
        use_oauth21 and authenticated_user and current_user_email != authenticated_user
    ):
        return current_user_email, args

    service_suffix = f" for service '{service_type}'" if service_type else ""
    logger.info(
        f"[{tool_name}] OAuth 2.1: Overriding user_google_email from '{current_user_email}' to authenticated user '{authenticated_user}'{service_suffix}"
    )

    # Update in kwargs if present
    if "user_google_email" in kwargs:
        kwargs["user_google_email"] = authenticated_user

    # Update in args if user_google_email is passed positionally
    try:
        user_email_index = param_names.index("user_google_email")
        args = _update_email_in_args(args, user_email_index, authenticated_user)
    except ValueError:
        pass  # user_google_email not in positional parameters

    return authenticated_user, args


async def _authenticate_service(
    use_oauth21: bool,
    service_name: str,
    service_version: str,
    tool_name: str,
    user_google_email: str,
    resolved_scopes: List[str],
    mcp_session_id: Optional[str],
    authenticated_user: Optional[str],
) -> Tuple[Any, str]:
    """
    Authenticate and get Google service using appropriate OAuth version.

    Returns:
        Tuple of (service, actual_user_email)
    """
    if use_oauth21:
        logger.debug(f"[{tool_name}] Using OAuth 2.1 flow")
        return await get_authenticated_google_service_oauth21(
            service_name=service_name,
            version=service_version,
            tool_name=tool_name,
            user_google_email=user_google_email,
            required_scopes=resolved_scopes,
            session_id=mcp_session_id,
            auth_token_email=authenticated_user,
            allow_recent_auth=False,
        )
    else:
        logger.debug(f"[{tool_name}] Using legacy OAuth 2.0 flow")
        return await get_authenticated_google_service(
            service_name=service_name,
            version=service_version,
            tool_name=tool_name,
            user_google_email=user_google_email,
            required_scopes=resolved_scopes,
            session_id=mcp_session_id,
        )


async def _authenticate_google_service_for_tool(
    tool_name: str,
    service_type: str,
    scopes: Union[str, List[str]],
    version: Optional[str] = None,
    args: tuple = (),
    kwargs: dict = None,
    wrapper_sig: Optional[inspect.Signature] = None,
    original_sig: Optional[inspect.Signature] = None,
    service_type_display: str = "",
) -> Tuple[Any, str, str]:
    """
    Comprehensive authentication helper for Google services.

    This function centralizes the authentication logic used by both
    require_google_service and require_multiple_services decorators,
    reducing code duplication and ensuring consistent behavior.

    Args:
        tool_name: Name of the tool/function being decorated
        service_type: Type of Google service ("gmail", "drive", etc.)
        scopes: Required scopes (can be scope group names or actual URLs)
        version: Optional service version override
        args: Positional arguments passed to wrapper
        kwargs: Keyword arguments passed to wrapper
        wrapper_sig: Function signature for wrapper (for OAuth 2.0 email extraction)
        original_sig: Original function signature (for multiple services decorator)
        service_type_display: Display name for service type (for logging)

    Returns:
        Tuple of (service, actual_user_email, user_google_email)
        - service: Authenticated Google API service client
        - actual_user_email: Email of the authenticated user (from service)
        - user_google_email: Email used for authentication (may be overridden)

    Raises:
        GoogleAuthenticationError: If authentication fails
        Exception: If service type is unknown
    """
    if kwargs is None:
        kwargs = {}

    # =====================================================================
    # STEP 1: Get authentication context from FastMCP
    # =====================================================================
    authenticated_user, auth_method, mcp_session_id = _get_auth_context(tool_name)

    # =====================================================================
    # STEP 2: Extract user email based on OAuth mode
    # =====================================================================
    if is_oauth21_enabled():
        user_google_email = _extract_oauth21_user_email(authenticated_user, tool_name)
    else:
        # Use original_sig if provided (for multiple services), otherwise wrapper_sig
        sig_to_use = original_sig if original_sig else wrapper_sig
        if not sig_to_use:
            raise Exception(
                f"Function signature required for OAuth 2.0 mode in {tool_name}"
            )
        user_google_email = _extract_oauth20_user_email(args, kwargs, sig_to_use)

    # =====================================================================
    # STEP 3: Get service configuration
    # =====================================================================
    if service_type not in SERVICE_CONFIGS:
        raise Exception(f"Unknown service type: {service_type}")

    config = SERVICE_CONFIGS[service_type]
    service_name = config["service"]
    service_version = version or config["version"]

    # =====================================================================
    # STEP 4: Resolve scopes
    # =====================================================================
    resolved_scopes = _resolve_scopes(scopes)

    # =====================================================================
    # STEP 5: Log authentication status
    # =====================================================================
    logger.debug(
        f"[{tool_name}] Auth: {authenticated_user or 'none'} via {auth_method or 'none'} "
        f"(session: {mcp_session_id[:8] if mcp_session_id else 'none'})"
    )

    # =====================================================================
    # STEP 6: Detect OAuth version
    # =====================================================================
    use_oauth21 = _detect_oauth_version(authenticated_user, mcp_session_id, tool_name)

    # =====================================================================
    # STEP 7: Override user_google_email with authenticated user when appropriate
    # =====================================================================
    # The override logic works based on use_oauth21 (per-request detection),
    # not the global is_oauth21_enabled() flag. This allows the override
    # to work correctly in mixed-mode scenarios where OAuth 2.1 is enabled
    # but some requests may use OAuth 2.0.
    # The _override_oauth21_user_email function only performs the override
    # when use_oauth21 is True, so we call it unconditionally and let the
    # function decide based on the actual OAuth version detected for this request.
    param_names = list((original_sig or wrapper_sig).parameters.keys())
    user_google_email, args = _override_oauth21_user_email(
        use_oauth21,
        authenticated_user,
        user_google_email,
        args,
        kwargs,
        param_names,
        tool_name,
        service_type_display,
    )

    # =====================================================================
    # STEP 8: Authenticate service
    # =====================================================================
    service, actual_user_email = await _authenticate_service(
        use_oauth21,
        service_name,
        service_version,
        tool_name,
        user_google_email,
        resolved_scopes,
        mcp_session_id,
        authenticated_user,
    )

    return service, actual_user_email, user_google_email


async def get_authenticated_google_service_oauth21(
    service_name: str,
    version: str,
    tool_name: str,
    user_google_email: str,
    required_scopes: List[str],
    session_id: Optional[str] = None,
    auth_token_email: Optional[str] = None,
    allow_recent_auth: bool = False,
) -> tuple[Any, str]:
    """
    OAuth 2.1 authentication using the session store with security validation.

    This function handles two authentication modes:
    1. Stateless mode (Cloud Run): Tokens from X-Google-Access-Token header
       - Credentials created per-request WITHOUT storing in session store
       - Each request is independent (true stateless operation)
    2. Session mode (stdio/legacy): Tokens from Authorization: Bearer header
       - Credentials stored in session store for reuse
       - Maintains backward compatibility with existing OAuth flows
    """
    # =====================================================================
    # STEP 1: Get authentication provider and access token from context
    # =====================================================================
    # The auth provider manages token lifecycle and validation
    # The access token is stored in FastMCP context state by the middleware
    provider = get_auth_provider()

    # CRITICAL FIX: Retrieve access token directly from context
    # get_access_token() from FastMCP dependencies may not work correctly,
    # so we retrieve it directly from context where middleware set it
    ctx = get_context()
    access_token = None
    if ctx:
        # Try access_token_obj first (set by middleware for verified tokens)
        access_token = ctx.get_state("access_token_obj")
        if not access_token:
            # Fallback to access_token (set by middleware as AccessTokenData)
            access_token = ctx.get_state("access_token")
        # If still None, try get_access_token() as last resort
        if not access_token:
            access_token = get_access_token()

    # CRITICAL FALLBACK: If context state didn't persist, try getting token from headers directly
    # This handles cases where middleware set the state but context isn't available during tool execution
    if not access_token:
        logger.warning(
            f"[{tool_name}] Access token not found in context - attempting fallback from HTTP headers"
        )
        try:
            headers = get_http_headers()
            logger.debug(
                f"[{tool_name}] Fallback: get_http_headers() returned: {headers is not None}, "
                f"Header keys: {list(headers.keys()) if headers else 'None'}"
            )
            if headers:
                # Try X-Google-Access-Token header first (stateless mode)
                token_str = headers.get("x-google-access-token") or headers.get(
                    "X-Google-Access-Token"
                )
                logger.debug(
                    f"[{tool_name}] Fallback: Token from header: {token_str[:20] + '...' if token_str else 'None'}"
                )
                if token_str and token_str.startswith("ya29."):
                    # Create AccessTokenData object from header token (same as middleware does)
                    from auth.auth_info_middleware import AccessTokenData
                    from auth.oauth_config import get_oauth_config

                    # Get OAuth config for client_id
                    oauth_config = get_oauth_config()
                    client_id = oauth_config.get("client_id", "google")

                    # Get scopes from Google's tokeninfo endpoint
                    # This is needed because we can't extract scopes from the token string directly
                    scopes = []
                    expires_at = int(time.time()) + 3600  # Default 1 hour expiration
                    try:
                        tokeninfo_url = f"https://oauth2.googleapis.com/tokeninfo?access_token={token_str}"
                        request = Request(tokeninfo_url)
                        with urlopen(request, timeout=5) as response:
                            if response.getcode() == 200:
                                tokeninfo = json.loads(response.read().decode())
                                scope_str = tokeninfo.get("scope", "")
                                scopes = scope_str.split() if scope_str else []
                                exp = tokeninfo.get("exp")
                                if exp:
                                    expires_at = int(exp)
                                logger.debug(
                                    f"[{tool_name}] Fallback: Retrieved scopes from tokeninfo: {len(scopes)} scope(s)"
                                )
                            else:
                                logger.warning(
                                    f"[{tool_name}] Fallback: tokeninfo endpoint returned {response.getcode()}, "
                                    f"using empty scopes (will fail scope validation)"
                                )
                    except (
                        URLError,
                        SocketTimeout,
                        json.JSONDecodeError,
                        ValueError,
                        Exception,
                    ) as e:
                        logger.warning(
                            f"[{tool_name}] Fallback: Could not get scopes from tokeninfo endpoint: {e}, "
                            f"using empty scopes (will fail scope validation)"
                        )

                    # Create AccessTokenData (middleware would have done this, but context didn't persist)
                    token_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]
                    session_id = f"google_oauth_{token_hash}"

                    access_token = AccessTokenData(
                        token=token_str,
                        client_id=client_id,
                        scopes=scopes,  # Scopes retrieved from tokeninfo endpoint
                        session_id=session_id,
                        expires_at=expires_at,  # Expiration from tokeninfo or default
                        sub=user_google_email,  # Use requested user email as sub
                        email=user_google_email,  # Use requested user email
                    )
                    logger.info(
                        f"[{tool_name}] FALLBACK SUCCESS: Retrieved access token directly from X-Google-Access-Token header "
                        f"(context state not available, token: {token_str[:20]}..., scopes: {len(scopes)} scope(s))"
                    )
                else:
                    logger.warning(
                        f"[{tool_name}] Fallback: Token found in header but doesn't start with 'ya29.': "
                        f"{token_str[:20] + '...' if token_str else 'None'}"
                    )
            else:
                logger.warning(
                    f"[{tool_name}] Fallback: get_http_headers() returned None or empty"
                )
        except Exception as e:
            logger.error(
                f"[{tool_name}] FALLBACK ERROR: Could not retrieve token from headers: {e}",
                exc_info=True,
            )

    # DEBUG: Log authentication state for troubleshooting
    logger.debug(
        f"[{tool_name}] Authentication check - Provider: {provider is not None}, "
        f"Access token: {access_token is not None}, "
        f"Context: {ctx is not None}, "
        f"User email: {user_google_email}, Session ID: {session_id}"
    )
    if access_token:
        # Get scopes from access_token - handle both None and empty list cases
        token_scopes = getattr(access_token, "scopes", None)
        if token_scopes is None:
            token_scopes = []

        logger.debug(
            f"[{tool_name}] Access token found - Type: {type(access_token).__name__}, "
            f"Has claims: {hasattr(access_token, 'claims')}, "
            f"Scopes: {len(token_scopes)} scope(s) (value: {token_scopes}), "
            f"Token preview: {getattr(access_token, 'token', 'N/A')[:10]}..."
            if hasattr(access_token, "token")
            else "N/A"
        )

        # CRITICAL: If access_token has empty scopes, fetch them from tokeninfo endpoint
        # This handles cases where middleware stored token without scopes (no auth provider to verify)
        # Check for both None and empty list - empty list [] is falsy, but be explicit
        if (token_scopes is None or len(token_scopes) == 0) and hasattr(
            access_token, "token"
        ):
            token_str = access_token.token
            if token_str and token_str.startswith("ya29."):
                logger.warning(
                    f"[{tool_name}] Access token from context has empty scopes - fetching from tokeninfo endpoint"
                )
                try:
                    tokeninfo_url = f"https://oauth2.googleapis.com/tokeninfo?access_token={token_str}"
                    request = Request(tokeninfo_url)
                    with urlopen(request, timeout=5) as response:
                        if response.getcode() == 200:
                            tokeninfo = json.loads(response.read().decode())
                            scope_str = tokeninfo.get("scope", "")
                            scopes = scope_str.split() if scope_str else []
                            if scopes:
                                # Update the access_token object with retrieved scopes
                                access_token.scopes = scopes
                                exp = tokeninfo.get("exp")
                                if exp:
                                    access_token.expires_at = int(exp)
                                logger.info(
                                    f"[{tool_name}] Retrieved {len(scopes)} scope(s) from tokeninfo endpoint "
                                    f"and updated access_token object"
                                )
                            else:
                                logger.warning(
                                    f"[{tool_name}] tokeninfo endpoint returned empty scopes"
                                )
                        else:
                            logger.warning(
                                f"[{tool_name}] tokeninfo endpoint returned {response.getcode()}"
                            )
                except (
                    URLError,
                    SocketTimeout,
                    json.JSONDecodeError,
                    ValueError,
                    Exception,
                ) as e:
                    logger.warning(
                        f"[{tool_name}] Could not get scopes from tokeninfo endpoint: {e}"
                    )
    elif ctx:
        # Log what's actually in context for debugging
        access_token_obj_in_ctx = ctx.get_state("access_token_obj")
        access_token_in_ctx = ctx.get_state("access_token")
        stateless_mode_in_ctx = ctx.get_state("stateless_mode")
        user_email_in_ctx = ctx.get_state("user_email")
        logger.warning(
            f"[{tool_name}] No access token found. Context state: "
            f"access_token_obj={access_token_obj_in_ctx is not None} (type: {type(access_token_obj_in_ctx).__name__ if access_token_obj_in_ctx else 'None'}), "
            f"access_token={access_token_in_ctx is not None} (type: {type(access_token_in_ctx).__name__ if access_token_in_ctx else 'None'}), "
            f"stateless_mode={stateless_mode_in_ctx}, "
            f"user_email={user_email_in_ctx}"
        )
        # If stateless_mode is True but access_token is None, this is a critical issue
        if (
            stateless_mode_in_ctx
            and not access_token_in_ctx
            and not access_token_obj_in_ctx
        ):
            logger.error(
                f"[{tool_name}] CRITICAL: stateless_mode=True but no access_token in context! "
                f"This indicates middleware set stateless_mode but failed to store access_token."
            )

    # =====================================================================
    # STEP 2: Process access token if available (from middleware or header fallback)
    # =====================================================================
    # This branch handles tokens that were extracted by the authentication
    # middleware (from X-Google-Access-Token or Authorization: Bearer headers)
    # OR retrieved directly from headers as a fallback if context state didn't persist
    # NOTE: In stateless mode, we can use access_token even if provider is None
    # (provider is only needed for token verification, which is optional)
    if access_token:
        # Extract user email from token (supports both AccessTokenData and AccessToken objects)
        # AccessTokenData has .email attribute directly
        # AccessToken (verified) has .claims.get("email")
        token_email = None
        if hasattr(access_token, "email"):
            # AccessTokenData object (from middleware when token not verified)
            token_email = access_token.email
        elif getattr(access_token, "claims", None):
            # AccessToken object (from verified token)
            token_email = access_token.claims.get("email")

        # Resolve the actual user email with priority:
        # 1. Token email (from token object - most reliable)
        # 2. Auth token email (from middleware context)
        # 3. Requested user email (from function parameter)
        resolved_email = token_email or auth_token_email or user_google_email
        if not resolved_email:
            raise GoogleAuthenticationError(
                "Authenticated user email could not be determined from access token."
            )

        # =====================================================================
        # STEP 3: Security validation - ensure email consistency
        # =====================================================================
        # CRITICAL SECURITY: Verify that all email sources match
        # This prevents token substitution attacks where a token for one user
        # is used to access another user's resources

        # Check 1: Token email must match auth token email (if both present)
        if auth_token_email and token_email and token_email != auth_token_email:
            raise GoogleAuthenticationError(
                "Access token email does not match authenticated session context."
            )

        # Check 2: Token email must match requested user email (if both present)
        if token_email and user_google_email and token_email != user_google_email:
            raise GoogleAuthenticationError(
                f"Authenticated account {token_email} does not match requested user {user_google_email}."
            )

        # =====================================================================
        # STEP 4: CRITICAL - Check stateless mode flag
        # =====================================================================
        # The middleware sets stateless_mode=True when token comes from
        # X-Google-Access-Token header (Cloud Run compatible)
        # stateless_mode=False when token comes from Authorization: Bearer
        # header (session mode, backward compatible)
        # FALLBACK: If context state not available, check headers directly
        ctx = get_context()
        stateless_mode = ctx.get_state("stateless_mode") if ctx else False

        # FALLBACK: If stateless_mode not in context, check if token came from X-Google-Access-Token header
        # If we got the token from the fallback mechanism, it's definitely stateless mode
        if not stateless_mode and access_token and hasattr(access_token, "token"):
            # Check if we got this token from the fallback (header retrieval)
            # If so, it's definitely stateless mode
            try:
                headers = get_http_headers()
                if headers:
                    # If token is from X-Google-Access-Token header, it's stateless mode
                    token_from_header = headers.get(
                        "x-google-access-token"
                    ) or headers.get("X-Google-Access-Token")
                    if token_from_header and token_from_header == access_token.token:
                        stateless_mode = True
                        logger.info(
                            f"[{tool_name}] FALLBACK: Detected stateless mode from X-Google-Access-Token header "
                            f"(context state not available, token matches header)"
                        )
                    else:
                        logger.debug(
                            f"[{tool_name}] Fallback stateless check: Token from header doesn't match access_token.token"
                        )
                else:
                    logger.debug(
                        f"[{tool_name}] Fallback stateless check: get_http_headers() returned None"
                    )
            except Exception as e:
                logger.debug(
                    f"[{tool_name}] Could not check headers for stateless mode: {e}"
                )

        # DEBUG: Log stateless mode detection
        logger.debug(
            f"[{tool_name}] Stateless mode check - Context: {ctx is not None}, "
            f"Stateless mode: {stateless_mode}, "
            f"Resolved email: {resolved_email}"
        )

        if stateless_mode:
            # =====================================================================
            # STATELESS MODE: Cloud Run compatible (no session storage)
            # =====================================================================
            # In stateless mode, we create credentials WITHOUT storing in
            # session store. This ensures true stateless operation where:
            # - Each request is independent
            # - No server-side state is maintained
            # - Cloud Run instances can scale to zero
            # - Multiple instances can handle requests without shared state

            # Extract token metadata from access_token object
            # Supports both AccessTokenData (has .scopes, .expires_at) and AccessToken objects
            token_scopes = getattr(access_token, "scopes", None)
            token_expires_at = getattr(access_token, "expires_at", None)

            # Extract token string (needed for creating credentials)
            token_str = None
            if hasattr(access_token, "token"):
                # AccessTokenData object
                token_str = access_token.token
            elif hasattr(access_token, "access_token"):
                # AccessToken object (verified)
                token_str = access_token.access_token

            # Create credentials WITHOUT storing in session store
            # This function only creates the Credentials object for this request
            # It does NOT call store.store_session() - critical for stateless operation
            credentials = create_credentials_from_token_stateless(
                access_token=access_token,
                user_email=resolved_email,
                scopes=token_scopes,
                expires_at=token_expires_at,
            )

            if not credentials:
                raise GoogleAuthenticationError(
                    "Unable to build Google credentials from stateless access token."
                )

            logger.debug(
                f"Created stateless credentials for {resolved_email} (NOT stored in session store)"
            )
        else:
            # =====================================================================
            # SESSION MODE: Legacy/stdio mode (with session storage)
            # =====================================================================
            # In session mode, we store credentials in session store for reuse
            # This maintains backward compatibility with:
            # - stdio transport mode (single-user, sequential requests)
            # - Legacy OAuth flows that expect session persistence
            # - Development workflows that rely on session caching

            # This function creates credentials AND stores them in session store
            # The session store allows credentials to be reused across requests
            # for the same user/session
            credentials = ensure_session_from_access_token(
                access_token, resolved_email, session_id
            )
            if not credentials:
                raise GoogleAuthenticationError(
                    "Unable to build Google credentials from authenticated access token."
                )

        # =====================================================================
        # STEP 5: Validate that credentials have required scopes
        # =====================================================================
        # CRITICAL SECURITY: Ensure the token has all scopes required by the tool
        # This prevents privilege escalation where a token with limited scopes
        # is used to access resources requiring additional permissions

        # Get available scopes from credentials or access_token
        scopes_available = set(credentials.scopes or [])
        if not scopes_available and getattr(access_token, "scopes", None):
            scopes_available = set(access_token.scopes)

        # Verify all required scopes are present
        if not all(scope in scopes_available for scope in required_scopes):
            raise GoogleAuthenticationError(
                f"OAuth credentials lack required scopes. Need: {required_scopes}, Have: {sorted(scopes_available)}"
            )

        # =====================================================================
        # STEP 6: Build Google API service client
        # =====================================================================
        # Create the Google API service client using the authenticated credentials
        # This client is used by the tool to make API calls to Google services
        service = build(service_name, version, credentials=credentials)
        logger.info(f"[{tool_name}] Authenticated {service_name} for {resolved_email}")
        return service, resolved_email

    # =====================================================================
    # FALLBACK: No access token from middleware - try session store
    # =====================================================================
    # This branch handles cases where no token was provided in headers
    # It tries to get credentials from the session store (OAuth 2.0 flow)
    logger.debug(
        f"[{tool_name}] No access token from middleware - falling back to session store. "
        f"Provider: {provider}, Access token: {access_token}, "
        f"User email: {user_google_email}, Session ID: {session_id}"
    )

    store = get_oauth21_session_store()

    # Use the validation method to ensure session can only access its own credentials
    credentials = store.get_credentials_with_validation(
        requested_user_email=user_google_email,
        session_id=session_id,
        auth_token_email=auth_token_email,
        allow_recent_auth=allow_recent_auth,
    )

    if not credentials:
        # CRITICAL: This error indicates the middleware did NOT extract the token from headers
        # This means either:
        # 1. Headers were not sent correctly from backend
        # 2. Headers were not received by MCP server
        # 3. get_http_headers() is not working during tool calls
        logger.error(
            f"[{tool_name}] CRITICAL: No credentials found in session store AND no access token from middleware. "
            f"This indicates X-Google-Access-Token header was not processed. "
            f"User: {user_google_email}, Session: {session_id}"
        )
        raise GoogleAuthenticationError(
            f"Access denied: Cannot retrieve credentials for {user_google_email}. "
            f"You can only access credentials for your authenticated account."
        )

    # =====================================================================
    # CRITICAL SECURITY: Validate that credentials have scope information
    # =====================================================================
    # If credentials.scopes is empty/None, we cannot verify that the token
    # has the required permissions. This is a security risk - we should
    # fail securely rather than assume the token has all required scopes.
    # A credential with no scope information could be used to access APIs
    # requiring higher privileges if we bypass validation.
    if not credentials.scopes:
        raise GoogleAuthenticationError(
            f"OAuth 2.1 credentials have no scope information. Cannot verify permissions for {user_google_email}. "
            f"Required scopes: {required_scopes}. "
            f"Please re-authenticate to ensure proper scope assignment."
        )

    # Get available scopes from credentials
    scopes_available = set(credentials.scopes)

    # =====================================================================
    # CRITICAL SECURITY: Verify all required scopes are present
    # =====================================================================
    # Ensure the token has all scopes required by the tool
    # This prevents privilege escalation where a token with limited scopes
    # is used to access resources requiring additional permissions
    if not all(scope in scopes_available for scope in required_scopes):
        raise GoogleAuthenticationError(
            f"OAuth 2.1 credentials lack required scopes. Need: {required_scopes}, Have: {sorted(scopes_available)}"
        )

    service = build(service_name, version, credentials=credentials)
    logger.info(f"[{tool_name}] Authenticated {service_name} for {user_google_email}")

    return service, user_google_email


def _extract_oauth21_user_email(
    authenticated_user: Optional[str], func_name: str
) -> str:
    """
    Extract user email for OAuth 2.1 mode.

    Args:
        authenticated_user: The authenticated user from context
        func_name: Name of the function being decorated (for error messages)

    Returns:
        User email string

    Raises:
        Exception: If no authenticated user found in OAuth 2.1 mode
    """
    if authenticated_user:
        return authenticated_user

    # CRITICAL FIX: Fallback to context if authenticated_user is None
    # This handles the case where middleware authenticated but context retrieval failed
    try:
        ctx = get_context()
        if ctx:
            # Try user_email as fallback (middleware sets this in stateless mode)
            user_email = ctx.get_state("user_email")
            if user_email:
                logger.debug(
                    f"[{func_name}] Retrieved user_email from context fallback: {user_email}"
                )
                return user_email

            # Try authenticated_user_email again (in case it was set after initial check)
            authenticated_user_email = ctx.get_state("authenticated_user_email")
            if authenticated_user_email:
                logger.debug(
                    f"[{func_name}] Retrieved authenticated_user_email from context fallback: {authenticated_user_email}"
                )
                return authenticated_user_email
    except Exception as e:
        logger.debug(
            f"[{func_name}] Could not retrieve user email from context fallback: {e}"
        )

    # If we still don't have a user email, raise an error
    raise Exception(
        f"OAuth 2.1 mode requires an authenticated user for {func_name}, but none was found in context."
    )


def _extract_oauth20_user_email(
    args: tuple, kwargs: dict, wrapper_sig: inspect.Signature
) -> str:
    """
    Extract user email for OAuth 2.0 mode from function arguments.

    Args:
        args: Positional arguments passed to wrapper
        kwargs: Keyword arguments passed to wrapper
        wrapper_sig: Function signature for parameter binding

    Returns:
        User email string

    Raises:
        Exception: If user_google_email parameter not found
    """
    bound_args = wrapper_sig.bind(*args, **kwargs)
    bound_args.apply_defaults()

    user_google_email = bound_args.arguments.get("user_google_email")
    if not user_google_email:
        raise Exception("'user_google_email' parameter is required but was not found.")
    return user_google_email


def _remove_user_email_arg_from_docstring(docstring: str) -> str:
    """
    Remove user_google_email parameter documentation from docstring.

    Args:
        docstring: The original function docstring

    Returns:
        Modified docstring with user_google_email parameter removed
    """
    if not docstring:
        return docstring

    # Pattern to match user_google_email parameter documentation
    # Handles various formats like:
    # - user_google_email (str): The user's Google email address. Required.
    # - user_google_email: Description
    # - user_google_email (str) - Description
    patterns = [
        r"^\s*user_google_email\s*\([^)]*\)\s*:\s*[^\n]*\.?\s*(?:Required\.?)?\s*\n",
        r"^\s*user_google_email\s*:\s*[^\n]*\n",
        r"^\s*user_google_email\s*\([^)]*\)\s*-\s*[^\n]*\n",
    ]

    modified_docstring = docstring
    for pattern in patterns:
        modified_docstring = re.sub(pattern, "", modified_docstring, flags=re.MULTILINE)

    # Clean up any sequence of 3 or more newlines that might have been created
    modified_docstring = re.sub(r"\n{3,}", "\n\n", modified_docstring)
    return modified_docstring


# Service configuration mapping
SERVICE_CONFIGS = {
    "gmail": {"service": "gmail", "version": "v1"},
    "drive": {"service": "drive", "version": "v3"},
    "calendar": {"service": "calendar", "version": "v3"},
    "docs": {"service": "docs", "version": "v1"},
    "sheets": {"service": "sheets", "version": "v4"},
    "chat": {"service": "chat", "version": "v1"},
    "forms": {"service": "forms", "version": "v1"},
    "slides": {"service": "slides", "version": "v1"},
    "tasks": {"service": "tasks", "version": "v1"},
    "customsearch": {"service": "customsearch", "version": "v1"},
}


# Scope group definitions for easy reference
SCOPE_GROUPS = {
    # Gmail scopes
    "gmail_read": GMAIL_READONLY_SCOPE,
    "gmail_send": GMAIL_SEND_SCOPE,
    "gmail_compose": GMAIL_COMPOSE_SCOPE,
    "gmail_modify": GMAIL_MODIFY_SCOPE,
    "gmail_labels": GMAIL_LABELS_SCOPE,
    # Drive scopes
    "drive_read": DRIVE_READONLY_SCOPE,
    "drive_file": DRIVE_FILE_SCOPE,
    # Docs scopes
    "docs_read": DOCS_READONLY_SCOPE,
    "docs_write": DOCS_WRITE_SCOPE,
    # Calendar scopes
    "calendar_read": CALENDAR_READONLY_SCOPE,
    "calendar_events": CALENDAR_EVENTS_SCOPE,
    # Sheets scopes
    "sheets_read": SHEETS_READONLY_SCOPE,
    "sheets_write": SHEETS_WRITE_SCOPE,
    # Chat scopes
    "chat_read": CHAT_READONLY_SCOPE,
    "chat_write": CHAT_WRITE_SCOPE,
    "chat_spaces": CHAT_SPACES_SCOPE,
    # Forms scopes
    "forms": FORMS_BODY_SCOPE,
    "forms_read": FORMS_BODY_READONLY_SCOPE,
    "forms_responses_read": FORMS_RESPONSES_READONLY_SCOPE,
    # Slides scopes
    "slides": SLIDES_SCOPE,
    "slides_read": SLIDES_READONLY_SCOPE,
    # Tasks scopes
    "tasks": TASKS_SCOPE,
    "tasks_read": TASKS_READONLY_SCOPE,
    # Custom Search scope
    "customsearch": CUSTOM_SEARCH_SCOPE,
}


def _resolve_scopes(scopes: Union[str, List[str]]) -> List[str]:
    """Resolve scope names to actual scope URLs."""
    if isinstance(scopes, str):
        if scopes in SCOPE_GROUPS:
            return [SCOPE_GROUPS[scopes]]
        else:
            return [scopes]

    resolved = []
    for scope in scopes:
        if scope in SCOPE_GROUPS:
            resolved.append(SCOPE_GROUPS[scope])
        else:
            resolved.append(scope)
    return resolved


def _handle_token_refresh_error(
    error: RefreshError, user_email: str, service_name: str
) -> str:
    """
    Handle token refresh errors gracefully, particularly expired/revoked tokens.

    Args:
        error: The RefreshError that occurred
        user_email: User's email address
        service_name: Name of the Google service

    Returns:
        A user-friendly error message with instructions for reauthentication
    """
    error_str = str(error)

    if (
        "invalid_grant" in error_str.lower()
        or "expired or revoked" in error_str.lower()
    ):
        logger.warning(
            f"Token expired or revoked for user {user_email} accessing {service_name}"
        )

        service_display_name = f"Google {service_name.title()}"

        return (
            f"**Authentication Required: Token Expired/Revoked for {service_display_name}**\n\n"
            f"Your Google authentication token for {user_email} has expired or been revoked. "
            f"This commonly happens when:\n"
            f"- The token has been unused for an extended period\n"
            f"- You've changed your Google account password\n"
            f"- You've revoked access to the application\n\n"
            f"**To resolve this, please:**\n"
            f"1. Run `start_google_auth` with your email ({user_email}) and service_name='{service_display_name}'\n"
            f"2. Complete the authentication flow in your browser\n"
            f"3. Retry your original command\n\n"
            f"The application will automatically use the new credentials once authentication is complete."
        )
    else:
        # Handle other types of refresh errors
        logger.error(f"Unexpected refresh error for user {user_email}: {error}")
        return (
            f"Authentication error occurred for {user_email}. "
            f"Please try running `start_google_auth` with your email and the appropriate service name to reauthenticate."
        )


def require_google_service(
    service_type: str,
    scopes: Union[str, List[str]],
    version: Optional[str] = None,
):
    """
    Decorator that automatically handles Google service authentication and injection.

    Args:
        service_type: Type of Google service ("gmail", "drive", "calendar", etc.)
        scopes: Required scopes (can be scope group names or actual URLs)
        version: Service version (defaults to standard version for service type)

    Usage:
        @require_google_service("gmail", "gmail_read")
        async def search_messages(service, user_google_email: str, query: str):
            # service parameter is automatically injected
            # Original authentication logic is handled automatically
    """

    def decorator(func: Callable) -> Callable:
        original_sig = inspect.signature(func)
        params = list(original_sig.parameters.values())

        # The decorated function must have 'service' as its first parameter.
        if not params or params[0].name != "service":
            raise TypeError(
                f"Function '{func.__name__}' decorated with @require_google_service "
                "must have 'service' as its first parameter."
            )

        # Create a new signature for the wrapper that excludes the 'service' parameter.
        # In OAuth 2.1 mode, also exclude 'user_google_email' since it's automatically determined.
        if is_oauth21_enabled():
            # Remove both 'service' and 'user_google_email' parameters
            filtered_params = [p for p in params[1:] if p.name != "user_google_email"]
            wrapper_sig = original_sig.replace(parameters=filtered_params)
        else:
            # Only remove 'service' parameter for OAuth 2.0 mode
            wrapper_sig = original_sig.replace(parameters=params[1:])

        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Note: `args` and `kwargs` are now the arguments for the *wrapper*,
            # which does not include 'service'.

            tool_name = func.__name__

            try:
                # =====================================================================
                # Use shared authentication helper to reduce code duplication
                # =====================================================================
                # This centralizes all authentication logic (context retrieval,
                # OAuth version detection, email extraction, service authentication)
                (
                    service,
                    actual_user_email,
                    user_google_email,
                ) = await _authenticate_google_service_for_tool(
                    tool_name=tool_name,
                    service_type=service_type,
                    scopes=scopes,
                    version=version,
                    args=args,
                    kwargs=kwargs,
                    wrapper_sig=wrapper_sig,
                    service_type_display=service_type,
                )
            except GoogleAuthenticationError as e:
                # Get context for detailed error logging
                authenticated_user, auth_method, mcp_session_id = _get_auth_context(
                    tool_name
                )
                config = SERVICE_CONFIGS[service_type]
                service_name = config["service"]
                service_version = version or config["version"]
                logger.error(
                    f"[{tool_name}] GoogleAuthenticationError during authentication. "
                    f"Method={auth_method or 'none'}, User={authenticated_user or 'none'}, "
                    f"Service={service_name} v{service_version}, MCPSessionID={mcp_session_id or 'none'}: {e}"
                )
                # Re-raise the original error without wrapping it
                raise

            try:
                # In OAuth 2.1 mode, we need to add user_google_email to kwargs since it was removed from signature
                if is_oauth21_enabled():
                    kwargs["user_google_email"] = user_google_email

                # Prepend the fetched service object to the original arguments
                return await func(service, *args, **kwargs)
            except RefreshError as e:
                config = SERVICE_CONFIGS[service_type]
                service_name = config["service"]
                error_message = _handle_token_refresh_error(
                    e, actual_user_email, service_name
                )
                raise Exception(error_message)

        # Set the wrapper's signature to the one without 'service'
        wrapper.__signature__ = wrapper_sig

        # Conditionally modify docstring to remove user_google_email parameter documentation
        if is_oauth21_enabled():
            logger.debug(
                "OAuth 2.1 mode enabled, removing user_google_email from docstring"
            )
            if func.__doc__:
                wrapper.__doc__ = _remove_user_email_arg_from_docstring(func.__doc__)

        return wrapper

    return decorator


def require_multiple_services(service_configs: List[Dict[str, Any]]):
    """
    Decorator for functions that need multiple Google services.

    Args:
        service_configs: List of service configurations, each containing:
            - service_type: Type of service
            - scopes: Required scopes
            - param_name: Name to inject service as (e.g., 'drive_service', 'docs_service')
            - version: Optional version override

    Usage:
        @require_multiple_services([
            {"service_type": "drive", "scopes": "drive_read", "param_name": "drive_service"},
            {"service_type": "docs", "scopes": "docs_read", "param_name": "docs_service"}
        ])
        async def get_doc_with_metadata(drive_service, docs_service, user_google_email: str, doc_id: str):
            # Both services are automatically injected
    """

    def decorator(func: Callable) -> Callable:
        original_sig = inspect.signature(func)

        # In OAuth 2.1 mode, remove user_google_email from the signature
        if is_oauth21_enabled():
            params = list(original_sig.parameters.values())
            filtered_params = [p for p in params if p.name != "user_google_email"]
            wrapper_sig = original_sig.replace(parameters=filtered_params)
        else:
            wrapper_sig = original_sig

        @wraps(func)
        async def wrapper(*args, **kwargs):
            tool_name = func.__name__
            user_google_email = None  # Will be set by first service authentication

            # =====================================================================
            # Authenticate all services using shared helper function
            # =====================================================================
            # This centralizes authentication logic and ensures consistent behavior
            # across both decorators. The shared helper handles all authentication
            # steps: context retrieval, OAuth version detection, email extraction,
            # email override, and service authentication.
            for config in service_configs:
                service_type = config["service_type"]
                scopes = config["scopes"]
                param_name = config["param_name"]
                version = config.get("version")

                try:
                    # Use shared authentication helper to reduce code duplication
                    # This handles all authentication steps for each service
                    (
                        service,
                        _,
                        user_google_email,
                    ) = await _authenticate_google_service_for_tool(
                        tool_name=tool_name,
                        service_type=service_type,
                        scopes=scopes,
                        version=version,
                        args=args,
                        kwargs=kwargs,
                        original_sig=original_sig,
                        service_type_display=service_type,
                    )

                    # Inject service with specified parameter name
                    kwargs[param_name] = service

                except GoogleAuthenticationError as e:
                    # Use user_google_email from previous successful auth or fallback
                    error_user = user_google_email or "unknown"
                    logger.error(
                        f"[{tool_name}] GoogleAuthenticationError for service '{service_type}' (user: {error_user}): {e}"
                    )
                    # Re-raise the original error without wrapping it
                    raise

            # Call the original function with refresh error handling
            try:
                # In OAuth 2.1 mode, we need to add user_google_email to kwargs since it was removed from signature
                if is_oauth21_enabled():
                    kwargs["user_google_email"] = user_google_email

                return await func(*args, **kwargs)
            except RefreshError as e:
                # Handle token refresh errors gracefully
                error_message = _handle_token_refresh_error(
                    e, user_google_email, "Multiple Services"
                )
                raise Exception(error_message)

        # Set the wrapper's signature
        wrapper.__signature__ = wrapper_sig

        # Conditionally modify docstring to remove user_google_email parameter documentation
        if is_oauth21_enabled():
            logger.debug(
                "OAuth 2.1 mode enabled, removing user_google_email from docstring"
            )
            if func.__doc__:
                wrapper.__doc__ = _remove_user_email_arg_from_docstring(func.__doc__)

        return wrapper

    return decorator
