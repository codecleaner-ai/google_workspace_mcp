"""
Authentication middleware to populate context state with user information
"""

import hashlib
import jwt
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.dependencies import get_http_headers

from auth.oauth21_session_store import ensure_session_from_access_token

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_TOKEN_EXPIRATION_SECONDS = 3600  # 1 hour in seconds


@dataclass
class AccessTokenData:
    """
    Data class representing an access token object.

    This replaces SimpleNamespace for better type safety, autocompletion, and self-documentation.
    """

    token: str
    client_id: str
    scopes: list[str]
    session_id: str
    expires_at: int
    sub: Optional[str] = None
    email: Optional[str] = None


class AuthInfoMiddleware(Middleware):
    """
    Middleware to extract authentication information from JWT tokens
    and populate the FastMCP context state for use in tools and prompts.
    """

    def __init__(self):
        super().__init__()
        self.auth_provider_type = "GoogleProvider"

        # Load configuration from environment variables once at initialization
        # This improves performance and makes dependencies explicit
        # CRITICAL: Trim whitespace to prevent authentication failures from unexpected characters
        google_oauth_client_id_raw = os.getenv("GOOGLE_OAUTH_CLIENT_ID", "google")
        self.google_oauth_client_id = (
            google_oauth_client_id_raw.strip()
            if google_oauth_client_id_raw
            else "google"
        )
        self.google_validate_tokens = (
            os.getenv("GOOGLE_VALIDATE_TOKENS", "true").lower() == "true"
        )
        jwt_secret_raw = os.getenv("JWT_SECRET")
        jwt_public_key_raw = os.getenv("JWT_PUBLIC_KEY")
        self.jwt_secret = jwt_secret_raw.strip() if jwt_secret_raw else None
        self.jwt_public_key = jwt_public_key_raw.strip() if jwt_public_key_raw else None

    def _extract_token_from_headers(
        self, headers: dict
    ) -> tuple[str | None, bool, str | None]:
        """
        Extract authentication token from HTTP headers.

        Priority:
        1. X-Mcp-Google-Token (stateless mode, Cloud Run compatible)
        2. Authorization: Bearer (session mode, backward compatible)

        Args:
            headers: HTTP request headers dictionary

        Returns:
            Tuple of (token_str, is_stateless, auth_source):
            - token_str: The extracted token string, or None if not found
            - is_stateless: True if token is from X-Mcp-Google-Token (stateless mode)
            - auth_source: Source of token ("x_mcp_google_token" or "bearer_token")
        """
        # PRIORITY 1: Check X-Mcp-Google-Token header first (for Cloud Run compatibility)
        # Note: Cloud Run strips X-Google-* headers, so we use X-Mcp-Google-Token
        google_access_token = headers.get("x-mcp-google-token") or headers.get(
            "X-Mcp-Google-Token"
        )

        if google_access_token:
            # CRITICAL: Trim whitespace to prevent authentication failures from unexpected characters
            google_access_token = google_access_token.strip()
            if google_access_token:  # Only return if token is not empty after trimming
                return google_access_token, True, "x_mcp_google_token"

        # PRIORITY 2: Fallback to Authorization header (for backward compatibility)
        auth_header = headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token_str = auth_header[7:]  # Remove "Bearer " prefix
            # CRITICAL: Trim whitespace to prevent authentication failures from unexpected characters
            token_str = token_str.strip()
            if token_str:  # Only return if token is not empty after trimming
                return token_str, False, "bearer_token"

        return None, False, None

    def _validate_token_format(self, token_str: str, auth_source: str) -> bool:
        """
        Validate that token is in correct format for Google OAuth.

        Args:
            token_str: Token string to validate
            auth_source: Source of token (for error messages)

        Returns:
            True if token format is valid, False otherwise
        """
        # Google OAuth tokens must start with "ya29."
        if not token_str.startswith("ya29."):
            if auth_source == "x_mcp_google_token":
                # X-Mcp-Google-Token header MUST contain ya29.* tokens only
                logger.error(
                    f"Invalid Google OAuth token format in {auth_source}: token does not start with 'ya29.'"
                )
            return False
        return True

    async def _verify_token(self, token_str: str) -> tuple[object | None, str | None]:
        """
        Optionally verify token using auth provider.

        Token verification can be disabled via GOOGLE_VALIDATE_TOKENS environment variable.
        If verification fails or is disabled, returns (None, None) and continues with unverified token.

        Args:
            token_str: Token string to verify

        Returns:
            Tuple of (verified_auth, user_email):
            - verified_auth: AccessToken object if verification succeeded, None otherwise
            - user_email: User email extracted from verified token, None if not available
        """
        from core.server import get_auth_provider

        auth_provider = get_auth_provider()
        if not auth_provider:
            logger.warning("No auth provider available to verify Google token")
            return None, None

        # Check if token validation is enabled (loaded from config in __init__)
        if not self.google_validate_tokens:
            logger.debug("Token validation disabled (GOOGLE_VALIDATE_TOKENS=false)")
            return None, None

        try:
            # Verify the token using existing auth provider
            verified_auth = await auth_provider.verify_token(token_str)
            if verified_auth:
                # Extract user email from verified token claims
                user_email = None
                if hasattr(verified_auth, "claims"):
                    user_email = verified_auth.claims.get("email")
                logger.debug(f"Token validated successfully for user: {user_email}")
                return verified_auth, user_email
            else:
                logger.error("Token validation failed")
                return None, None
        except (ConnectionError, TimeoutError) as e:
            # Network-related errors during token verification
            logger.error(f"Network error verifying Google OAuth token: {e}")
            return None, None
        except ValueError as e:
            # Invalid token format or structure
            logger.error(f"Invalid token format during verification: {e}")
            return None, None
        except AttributeError as e:
            # Missing expected attributes on verified_auth object
            logger.error(f"Unexpected token structure during verification: {e}")
            return None, None
        except Exception as e:
            # Catch-all for unexpected errors - log with full context
            logger.error(
                f"Unexpected error verifying Google OAuth token: {e}", exc_info=True
            )
            return None, None

    def _extract_token_metadata(
        self, verified_auth: object | None, user_email: str | None
    ) -> tuple[int, str, list]:
        """
        Extract token metadata (expires_at, client_id, scopes) from verified token or use defaults.

        Args:
            verified_auth: Verified AccessToken object, or None if not verified
            user_email: User email, or None if not available

        Returns:
            Tuple of (expires_at, client_id, scopes)
        """
        # Get token expiration time
        if verified_auth and hasattr(verified_auth, "expires_at"):
            expires_at = verified_auth.expires_at
        else:
            expires_at = int(time.time()) + DEFAULT_TOKEN_EXPIRATION_SECONDS

        # Get OAuth client ID (use instance attribute loaded in __init__)
        client_id = None
        if verified_auth:
            client_id = getattr(verified_auth, "client_id", None)
        if not client_id:
            client_id = self.google_oauth_client_id

        # Get OAuth scopes
        scopes = []
        if verified_auth and hasattr(verified_auth, "scopes"):
            scopes = verified_auth.scopes

        return expires_at, client_id, scopes

    def _create_access_token_object(
        self,
        token_str: str,
        client_id: str,
        scopes: list,
        expires_at: int,
        verified_auth: object | None,
        user_email: str | None,
    ) -> AccessTokenData:
        """
        Create access token data class object.

        Args:
            token_str: The actual token string
            client_id: OAuth client ID
            scopes: List of OAuth scopes
            expires_at: Token expiration timestamp
            verified_auth: Verified AccessToken object, or None
            user_email: User email, or None

        Returns:
            AccessTokenData object with token information
        """
        # Safely extract sub from verified_auth, with fallback to user_email
        # Use getattr with default value (more Pythonic than hasattr + try/except)
        sub = getattr(verified_auth, "sub", None) if verified_auth else None
        if not sub:
            sub = user_email

        # Generate session ID using hash of token for better uniqueness and security
        # Using hash instead of token prefix prevents predictable session IDs
        token_hash = hashlib.sha256(token_str.encode()).hexdigest()[:16]
        session_id = f"google_oauth_{token_hash}"

        return AccessTokenData(
            token=token_str,
            client_id=client_id,
            scopes=scopes,
            session_id=session_id,
            expires_at=expires_at,
            sub=sub,
            email=user_email,
        )

    def _store_authentication_state(
        self,
        context: MiddlewareContext,
        access_token: AccessTokenData,
        verified_auth: object | None,
        user_email: str | None,
        is_stateless: bool,
        auth_source: str,
    ) -> None:
        """
        Store authentication state in FastMCP context.

        This method handles the critical distinction between stateless and session modes:
        - Stateless mode: Only stores in context state (NOT in session store)
        - Session mode: Stores in both context state AND session store

        Args:
            context: FastMCP middleware context
            access_token: Access token SimpleNamespace object
            verified_auth: Verified AccessToken object, or None
            user_email: User email, or None
            is_stateless: True if token is stateless (from X-Mcp-Google-Token)
            auth_source: Source of token ("x_mcp_google_token" or "bearer_token")
        """
        # Store basic authentication state in context
        context.fastmcp_context.set_state("access_token", access_token)
        context.fastmcp_context.set_state("auth_provider_type", self.auth_provider_type)
        context.fastmcp_context.set_state("token_type", "google_oauth")
        context.fastmcp_context.set_state("user_email", user_email)
        context.fastmcp_context.set_state("username", user_email)

        # CRITICAL: Set stateless mode flag
        # This tells service decorator how to handle the token
        context.fastmcp_context.set_state("stateless_mode", is_stateless)

        # CRITICAL FOR CLOUD RUN: Handle session store
        if is_stateless:
            # Stateless mode: Do NOT store in session store
            logger.debug(
                f"Stateless mode: Token from {auth_source} NOT stored in session store"
            )
            if verified_auth:
                context.fastmcp_context.set_state("access_token_obj", verified_auth)
        else:
            # Session mode: Store in session store (backward compatibility)
            mcp_session_id = getattr(context.fastmcp_context, "session_id", None)
            if verified_auth:
                ensure_session_from_access_token(
                    verified_auth, user_email, mcp_session_id
                )
                context.fastmcp_context.set_state("access_token_obj", verified_auth)

        # Set definitive authentication state
        context.fastmcp_context.set_state("authenticated_user_email", user_email)
        context.fastmcp_context.set_state(
            "authenticated_via", auth_source or "bearer_token"
        )

        # Log authentication success
        mode = "stateless mode" if is_stateless else "session mode"
        logger.info(
            f"Authenticated via Google OAuth ({mode}, source: {auth_source}): {user_email or 'token validated'}"
        )

    def _get_allowed_jwt_algorithms(self) -> list[str]:
        """
        Get the list of allowed JWT algorithms for signature verification.

        CRITICAL SECURITY: This defines a strict whitelist of allowed algorithms.
        We NEVER trust the algorithm from the token header (algorithm confusion attack).
        Only algorithms in this whitelist will be accepted.

        Returns:
            List of allowed algorithm names (e.g., ["HS256", "RS256"])
        """
        # Define allowed algorithms based on configured keys (loaded in __init__)
        allowed_algorithms = []

        # Check if HS256 is configured (symmetric/HMAC)
        if self.jwt_secret:
            allowed_algorithms.append("HS256")

        # Check if RS256 is configured (asymmetric/RSA)
        if self.jwt_public_key:
            allowed_algorithms.append("RS256")

        # If no algorithms are configured, return empty list (tokens will be rejected)
        return allowed_algorithms

    def _get_jwt_verification_key(self, algorithm: str) -> str | None:
        """
        Get the verification key for JWT signature verification.

        CRITICAL SECURITY: The algorithm parameter is NOT from the token header.
        It comes from our whitelist of allowed algorithms to prevent algorithm confusion attacks.

        This method retrieves the public key or secret needed to verify the JWT signature
        from instance attributes (loaded in __init__).

        Args:
            algorithm: The algorithm we're verifying with (from our whitelist, not token header)

        Returns:
            Verification key (secret or public key), or None if not available
        """
        if algorithm == "HS256":
            # Symmetric key algorithm - use secret (loaded in __init__)
            return self.jwt_secret
        elif algorithm == "RS256":
            # Asymmetric key algorithm - use public key (loaded in __init__)
            return self.jwt_public_key

        return None

    def _verify_jwt_token(self, token_str: str) -> dict | None:
        """
        Verify and decode JWT token with signature verification.

        CRITICAL SECURITY: This method verifies the JWT signature to ensure
        the token is authentic and hasn't been tampered with. It uses a strict
        algorithm whitelist to prevent algorithm confusion attacks.

        SECURITY FIX: We NEVER trust the algorithm from the token header.
        Instead, we maintain a strict whitelist of allowed algorithms and
        try each one until verification succeeds or all are exhausted.

        Args:
            token_str: JWT token string to verify

        Returns:
            Decoded token payload if verification succeeds, None otherwise
        """
        # Get the strict whitelist of allowed algorithms
        # CRITICAL: We do NOT trust the algorithm from the token header
        allowed_algorithms = self._get_allowed_jwt_algorithms()

        if not allowed_algorithms:
            logger.error(
                "JWT signature verification failed: No verification key configured. "
                "Set JWT_SECRET (for HS256) or JWT_PUBLIC_KEY (for RS256) environment variable. "
                "JWT tokens cannot be accepted without signature verification."
            )
            return None

        # Try each allowed algorithm until one succeeds
        # This prevents algorithm confusion attacks by not trusting the header's alg value
        for algorithm in allowed_algorithms:
            try:
                # Get the verification key for this algorithm
                verification_key = self._get_jwt_verification_key(algorithm)

                if not verification_key:
                    # Key not configured for this algorithm, try next
                    continue

                # Verify and decode the JWT with signature verification
                # CRITICAL: We pass a strict whitelist of allowed algorithms
                # PyJWT will reject tokens that don't use one of these algorithms
                token_payload = jwt.decode(
                    token_str,
                    verification_key,
                    algorithms=[algorithm],  # Strict whitelist - only this algorithm
                    options={
                        "verify_signature": True,  # CRITICAL: Always verify signature
                        "verify_exp": True,  # Verify expiration
                        "verify_iat": True,  # Verify issued at time
                    },
                )

                # Verification succeeded with this algorithm
                logger.debug(
                    f"JWT token verified successfully with {algorithm}: {list(token_payload.keys())}"
                )
                return token_payload

            except jwt.InvalidAlgorithmError:
                # Token uses a different algorithm than the one we're trying
                # Continue to next algorithm in whitelist
                continue
            except jwt.ExpiredSignatureError:
                logger.error("JWT token has expired")
                return None
            except jwt.InvalidTokenError as e:
                # This includes InvalidSignatureError, InvalidAudienceError, etc.
                # If it's an algorithm mismatch, continue to next algorithm
                # Otherwise, log and return None
                error_str = str(e).lower()
                if "algorithm" in error_str or "alg" in error_str:
                    continue
                logger.error(f"JWT token verification failed: {e}")
                return None
            except jwt.DecodeError as e:
                # Malformed token - don't try other algorithms
                logger.error(f"Failed to decode JWT: {e}")
                return None
            except (TypeError, ValueError) as e:
                # Invalid token structure or key format - don't try other algorithms
                logger.error(f"Invalid JWT token structure or key: {e}")
                return None

        # None of the allowed algorithms worked
        logger.error(
            f"JWT token verification failed: Token algorithm not in allowed list {allowed_algorithms}. "
            "This may indicate an algorithm confusion attack attempt."
        )
        return None

    def _process_jwt_token(self, context: MiddlewareContext, token_str: str) -> None:
        """
        Process JWT token from Authorization: Bearer header (backward compatibility).

        This handles non-ya29.* tokens (JWT tokens) from Authorization header.
        Note: X-Mcp-Google-Token header MUST contain ya29.* tokens only.

        SECURITY: JWT tokens are verified with signature verification enabled.
        Without a valid signature, the token will be rejected.

        Args:
            context: FastMCP middleware context
            token_str: JWT token string
        """
        # CRITICAL SECURITY: Verify JWT signature before processing
        token_payload = self._verify_jwt_token(token_str)

        if not token_payload:
            # Token verification failed - do not process
            logger.warning("JWT token rejected due to verification failure")
            return

        try:
            # Extract session ID from token payload, or generate a unique one if not found
            # Priority: sid > jti > session_id > generated UUID
            # CRITICAL: Never use static fallback like "unknown" to prevent session collisions
            session_id = (
                token_payload.get("sid")
                or token_payload.get("jti")
                or token_payload.get("session_id")
            )

            if not session_id:
                # Generate a unique UUID if no session ID found in token payload
                # This ensures each token gets a unique session ID, preventing collisions
                session_id = f"jwt_{uuid.uuid4().hex[:16]}"
                logger.debug(
                    "Generated unique session ID for JWT token without session identifier"
                )

            # Create access token object from verified JWT payload
            access_token = AccessTokenData(
                token=token_str,
                client_id=token_payload.get("client_id", "unknown"),
                scopes=token_payload.get("scope", "").split()
                if token_payload.get("scope")
                else [],
                session_id=session_id,
                expires_at=token_payload.get("exp", 0),
                sub=token_payload.get("sub"),
                email=token_payload.get("email", token_payload.get("username")),
            )

            # Store in context state
            context.fastmcp_context.set_state("access_token", access_token)
            context.fastmcp_context.set_state("user_id", token_payload.get("sub"))
            context.fastmcp_context.set_state(
                "username", token_payload.get("username", token_payload.get("email"))
            )
            context.fastmcp_context.set_state("name", token_payload.get("name"))
            context.fastmcp_context.set_state(
                "auth_time", token_payload.get("auth_time")
            )
            context.fastmcp_context.set_state("issuer", token_payload.get("iss"))
            context.fastmcp_context.set_state("audience", token_payload.get("aud"))
            context.fastmcp_context.set_state("jti", token_payload.get("jti"))
            context.fastmcp_context.set_state(
                "auth_provider_type", self.auth_provider_type
            )

            # Set authentication state
            user_email = token_payload.get("email", token_payload.get("username"))
            if user_email:
                context.fastmcp_context.set_state(
                    "authenticated_user_email", user_email
                )
                context.fastmcp_context.set_state("authenticated_via", "jwt_token")

            logger.debug("JWT token processed successfully")
        except Exception as e:
            # Unexpected error processing verified token
            logger.error(f"Error processing verified JWT token: {e}", exc_info=True)

    def _handle_stdio_authentication(self, context: MiddlewareContext) -> None:
        """
        Handle authentication for stdio transport mode.

        In stdio mode, we can safely use session store because:
        1. It's single-user (one user per process)
        2. No concurrent requests (sequential processing)
        3. Process lifetime matches session lifetime

        This is ONLY safe in stdio mode - DO NOT use in HTTP mode.

        Args:
            context: FastMCP middleware context
        """
        from core.config import get_transport_mode
        from auth.oauth21_session_store import get_oauth21_session_store

        transport_mode = get_transport_mode()
        if transport_mode != "stdio":
            return

        logger.debug("Checking for stdio mode authentication")

        # Try to get requested user from context
        requested_user = None
        if hasattr(context, "request") and hasattr(context.request, "params"):
            requested_user = context.request.params.get("user_google_email")
        elif hasattr(context, "arguments"):
            requested_user = context.arguments.get("user_google_email")

        # Check if user has a recent session
        if requested_user:
            try:
                store = get_oauth21_session_store()
                if store.has_session(requested_user):
                    logger.debug(f"Using recent stdio session for {requested_user}")
                    context.fastmcp_context.set_state(
                        "authenticated_user_email", requested_user
                    )
                    context.fastmcp_context.set_state(
                        "authenticated_via", "stdio_session"
                    )
                    context.fastmcp_context.set_state(
                        "auth_provider_type", "oauth21_stdio"
                    )
            except Exception as e:
                logger.debug(f"Error checking stdio session: {e}")

        # If no requested user, check for single-user session
        if not context.fastmcp_context.get_state("authenticated_user_email"):
            try:
                store = get_oauth21_session_store()
                single_user = store.get_single_user_email()
                if single_user:
                    logger.debug(
                        f"Defaulting to single stdio OAuth session for {single_user}"
                    )
                    context.fastmcp_context.set_state(
                        "authenticated_user_email", single_user
                    )
                    context.fastmcp_context.set_state(
                        "authenticated_via", "stdio_single_session"
                    )
                    context.fastmcp_context.set_state(
                        "auth_provider_type", "oauth21_stdio"
                    )
                    context.fastmcp_context.set_state("user_email", single_user)
                    context.fastmcp_context.set_state("username", single_user)
            except Exception as e:
                logger.debug(f"Error determining stdio single-user session: {e}")

    def _handle_mcp_session_binding(self, context: MiddlewareContext) -> None:
        """
        Handle authentication via MCP session binding.

        In some cases, an MCP session may be pre-bound to a user.
        This happens when a user authenticates via OAuth flow and the session
        is stored with a mapping to the user's email.

        This is a fallback method when no token is provided in headers.

        Args:
            context: FastMCP middleware context
        """
        if context.fastmcp_context.get_state("authenticated_user_email"):
            return

        if not hasattr(context.fastmcp_context, "session_id"):
            return

        mcp_session_id = context.fastmcp_context.session_id
        if not mcp_session_id:
            return

        try:
            from auth.oauth21_session_store import get_oauth21_session_store

            store = get_oauth21_session_store()
            bound_user = store.get_user_by_mcp_session(mcp_session_id)

            if bound_user:
                logger.debug(f"MCP session bound to {bound_user}")
                context.fastmcp_context.set_state(
                    "authenticated_user_email", bound_user
                )
                context.fastmcp_context.set_state(
                    "authenticated_via", "mcp_session_binding"
                )
                context.fastmcp_context.set_state(
                    "auth_provider_type", "oauth21_session"
                )
        except Exception as e:
            logger.debug(f"Error checking MCP session binding: {e}")

    async def _process_google_oauth_token(
        self,
        context: MiddlewareContext,
        token_str: str,
        is_stateless: bool,
        auth_source: str,
    ) -> None:
        """
        Process Google OAuth token (ya29.* format).

        This is the main handler for Google OAuth tokens. It:
        1. Validates token format
        2. Optionally verifies token
        3. Extracts metadata
        4. Creates access token object
        5. Stores authentication state

        Args:
            context: FastMCP middleware context
            token_str: Token string (must start with ya29.)
            is_stateless: True if token is stateless (from X-Mcp-Google-Token)
            auth_source: Source of token ("x_mcp_google_token" or "bearer_token")
        """
        logger.debug("Detected Google OAuth access token format")

        # Step 1: Verify token (optional)
        verified_auth, user_email = await self._verify_token(token_str)

        # Step 2: Extract token metadata
        expires_at, client_id, scopes = self._extract_token_metadata(
            verified_auth, user_email
        )

        # Step 3: Create access token object
        access_token = self._create_access_token_object(
            token_str, client_id, scopes, expires_at, verified_auth, user_email
        )

        # Step 4: Store authentication state
        self._store_authentication_state(
            context, access_token, verified_auth, user_email, is_stateless, auth_source
        )

    async def _process_request_for_auth(self, context: MiddlewareContext):
        """
        Extract, verify, and store authentication information from request headers.

        This method implements a two-tier authentication system:
        1. Stateless mode (X-Mcp-Google-Token): For Cloud Run deployment
           - Tokens are NOT stored in session store
           - Each request is independent (true stateless operation)
           - Required for Cloud Run where instances can scale to zero

        2. Session mode (Authorization: Bearer): For backward compatibility
           - Tokens are stored in session store
           - Supports stdio mode and legacy HTTP mode
           - Maintains existing behavior for non-Cloud Run deployments
        """
        # Early return: Check if FastMCP context is available
        # Without context, we cannot store authentication state
        if not context.fastmcp_context:
            logger.warning("No fastmcp_context available")
            return

        # Early return: If authentication is already set, skip processing
        # This prevents redundant token verification and improves performance
        if context.fastmcp_context.get_state("authenticated_user_email"):
            logger.info("Authentication state already set.")
            return

        # =====================================================================
        # STEP 1: Try to extract token from HTTP headers
        # =====================================================================
        try:
            headers = get_http_headers()
            # CRITICAL DEBUG: Log detailed information about get_http_headers() behavior
            logger.debug(
                f"🔍 get_http_headers() result - Type: {type(headers)}, "
                f"Is None: {headers is None}, "
                f"Is Empty Dict: {headers == {}}, "
                f"Has Keys: {list(headers.keys()) if headers else 'N/A'}, "
                f"Header Count: {len(headers) if headers else 0}"
            )
            if headers:
                # DEBUG: Log all headers received for troubleshooting
                logger.debug(
                    f"Processing HTTP headers for authentication - Header keys: {list(headers.keys())}, "
                    f"Has X-Mcp-Google-Token: {'x-mcp-google-token' in [k.lower() for k in headers.keys()] or 'X-Mcp-Google-Token' in headers}"
                )

                # Extract token from headers (X-Mcp-Google-Token or Authorization: Bearer)
                token_str, is_stateless, auth_source = self._extract_token_from_headers(
                    headers
                )

                # DEBUG: Log token extraction result
                if token_str:
                    logger.debug(
                        f"Token extracted from headers - Source: {auth_source}, "
                        f"Is stateless: {is_stateless}, Token preview: {token_str[:10]}..."
                    )
                else:
                    logger.debug("No token found in headers after extraction")

                if token_str:
                    # Validate token format (must be ya29.* for Google OAuth)
                    if not self._validate_token_format(token_str, auth_source):
                        # Invalid format - error already logged
                        return

                    # Process token based on format
                    if token_str.startswith("ya29."):
                        # Google OAuth token - process it
                        await self._process_google_oauth_token(
                            context, token_str, is_stateless, auth_source
                        )
                    elif auth_source == "bearer_token":
                        # JWT token from Authorization header (backward compatibility)
                        self._process_jwt_token(context, token_str)
                    else:
                        # Invalid: X-Mcp-Google-Token must contain ya29.* tokens only
                        logger.error(
                            f"Invalid Google OAuth token format in {auth_source}: token does not start with 'ya29.'"
                        )
                else:
                    logger.debug("No Bearer token or X-Mcp-Google-Token in headers")
            else:
                logger.debug(
                    "⚠️ get_http_headers() returned None or empty - This is expected during tool calls with SSE, "
                    "as tool calls are JSON-RPC messages within the SSE stream, not separate HTTP requests"
                )
        except Exception as e:
            logger.debug(f"Could not get HTTP request: {e}")
            # DEBUG: Log all headers received for troubleshooting
            logger.debug(
                f"Processing HTTP headers for authentication - Header keys: {list(headers.keys())}, "
                f"Has X-Mcp-Google-Token: {'x-mcp-google-token' in [k.lower() for k in headers.keys()] or 'X-Mcp-Google-Token' in headers}"
            )

            # Extract token from headers (X-Mcp-Google-Token or Authorization: Bearer)
            token_str, is_stateless, auth_source = self._extract_token_from_headers(
                headers
            )

            # DEBUG: Log token extraction result
            if token_str:
                logger.debug(
                    f"Token extracted from headers - Source: {auth_source}, "
                    f"Is stateless: {is_stateless}, Token preview: {token_str[:10]}..."
                )
            else:
                logger.debug("No token found in headers after extraction")

            if token_str:
                # Validate token format (must be ya29.* for Google OAuth)
                if not self._validate_token_format(token_str, auth_source):
                    # Invalid format - error already logged
                    return

                # Process token based on format
                if token_str.startswith("ya29."):
                    # Google OAuth token - process it
                    await self._process_google_oauth_token(
                        context, token_str, is_stateless, auth_source
                    )
                elif auth_source == "bearer_token":
                    # JWT token from Authorization header (backward compatibility)
                    self._process_jwt_token(context, token_str)
                else:
                    # Invalid: X-Mcp-Google-Token must contain ya29.* tokens only
                    logger.error(
                        f"Invalid Google OAuth token format in {auth_source}: token does not start with 'ya29.'"
                    )
            else:
                logger.debug("No Bearer token or X-Mcp-Google-Token in headers")

        # =====================================================================
        # STEP 2: Fallback authentication methods (for stdio mode)
        # =====================================================================
        # If no authentication was found via HTTP headers, try fallback methods
        if not context.fastmcp_context.get_state("authenticated_user_email"):
            logger.debug(
                "No authentication found via bearer token, checking other methods"
            )

            # Try stdio mode authentication
            self._handle_stdio_authentication(context)

            # Try MCP session binding
            self._handle_mcp_session_binding(context)

    async def _process_middleware_request(
        self, context: MiddlewareContext, call_next, request_type: str
    ) -> any:
        """
        Common middleware request processing logic.

        This method extracts authentication information and processes the request.
        It's used by both on_call_tool and on_get_prompt to eliminate code duplication.

        Args:
            context: FastMCP middleware context
            call_next: Next middleware/tool handler
            request_type: Type of request ("tool" or "prompt") for logging

        Returns:
            Result from the next handler

        Raises:
            Re-raises any exceptions that occur during processing
        """
        logger.debug(
            f"🔍 Processing {request_type} authentication - "
            f"Context available: {context.fastmcp_context is not None}, "
            f"Auth already set: {context.fastmcp_context.get_state('authenticated_user_email') if context.fastmcp_context else 'N/A'}"
        )

        try:
            # Extract and store authentication information
            # This populates context state with user email, token, etc.
            await self._process_request_for_auth(context)

            logger.debug(f"Passing {request_type} to next handler")
            # Continue to the next middleware/tool handler
            result = await call_next(context)
            logger.debug(f"{request_type.capitalize()} handler completed")
            return result

        except Exception as e:
            # Handle authentication errors gracefully
            # Don't log full traceback for authentication errors (they're expected)
            # This reduces log noise while still logging the error message
            error_type_str = str(type(e))
            error_msg_str = str(e)

            if (
                "GoogleAuthenticationError" in error_type_str
                or "Access denied: Cannot retrieve credentials" in error_msg_str
            ):
                logger.info(f"Authentication check failed in {request_type}: {e}")
            else:
                # For unexpected errors, log full traceback
                logger.error(f"Error in {request_type} middleware: {e}", exc_info=True)
            raise

    async def on_call_tool(self, context: MiddlewareContext, call_next):
        """
        Middleware hook for tool calls.

        This is called by FastMCP before executing any tool.
        It extracts authentication information from request headers and stores it in context state.
        The authentication information is then available to the tool function.
        """
        return await self._process_middleware_request(context, call_next, "tool")

    async def on_get_prompt(self, context: MiddlewareContext, call_next):
        """
        Middleware hook for prompt requests.

        This is called by FastMCP before generating prompts.
        It extracts authentication information from request headers and stores it in context state.
        The authentication information is then available to the prompt generation.
        """
        return await self._process_middleware_request(context, call_next, "prompt")
