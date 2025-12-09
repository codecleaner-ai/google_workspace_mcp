"""
OAuth 2.1 Session Store for Google Services

This module provides a global store for OAuth 2.1 authenticated sessions
that can be accessed by Google service decorators. It also includes
session context management and credential conversion functionality.
"""

import contextvars
import logging
from typing import Dict, Optional, Any, Tuple, Union
from threading import RLock
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass

from fastmcp.server.auth import AccessToken
from google.oauth2.credentials import Credentials

logger = logging.getLogger(__name__)


def _normalize_expiry_to_naive_utc(expiry: Optional[Any]) -> Optional[datetime]:
    """
    Convert expiry values to timezone-naive UTC datetimes for google-auth compatibility.

    Naive datetime inputs are assumed to already represent UTC and are returned unchanged so that
    google-auth Credentials receive naive UTC datetimes for expiry comparison.
    """
    if expiry is None:
        return None

    if isinstance(expiry, datetime):
        if expiry.tzinfo is not None:
            try:
                return expiry.astimezone(timezone.utc).replace(tzinfo=None)
            except Exception:  # pragma: no cover - defensive
                logger.debug(
                    "Failed to normalize aware expiry; returning without tzinfo"
                )
                return expiry.replace(tzinfo=None)
        return expiry  # Already naive; assumed to represent UTC

    if isinstance(expiry, str):
        try:
            parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
        except ValueError:
            logger.debug("Failed to parse expiry string '%s'", expiry)
            return None
        return _normalize_expiry_to_naive_utc(parsed)

    logger.debug("Unsupported expiry type '%s' (%s)", expiry, type(expiry))
    return None


# Context variable to store the current session information
_current_session_context: contextvars.ContextVar[Optional["SessionContext"]] = (
    contextvars.ContextVar("current_session_context", default=None)
)


@dataclass
class SessionContext:
    """Container for session-related information."""

    session_id: Optional[str] = None
    user_id: Optional[str] = None
    auth_context: Optional[Any] = None
    request: Optional[Any] = None
    metadata: Dict[str, Any] = None
    issuer: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


def set_session_context(context: Optional[SessionContext]):
    """
    Set the current session context.

    Args:
        context: The session context to set
    """
    _current_session_context.set(context)
    if context:
        logger.debug(
            f"Set session context: session_id={context.session_id}, user_id={context.user_id}"
        )
    else:
        logger.debug("Cleared session context")


def get_session_context() -> Optional[SessionContext]:
    """
    Get the current session context.

    Returns:
        The current session context or None
    """
    return _current_session_context.get()


def clear_session_context():
    """Clear the current session context."""
    set_session_context(None)


class SessionContextManager:
    """
    Context manager for temporarily setting session context.

    Usage:
        with SessionContextManager(session_context):
            # Code that needs access to session context
            pass
    """

    def __init__(self, context: Optional[SessionContext]):
        self.context = context
        self.token = None

    def __enter__(self):
        """Set the session context."""
        self.token = _current_session_context.set(self.context)
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Reset the session context."""
        if self.token:
            _current_session_context.reset(self.token)


def extract_session_from_headers(headers: Dict[str, str]) -> Optional[str]:
    """
    Extract session ID from request headers.

    Args:
        headers: Request headers

    Returns:
        Session ID if found
    """
    # Try different header names
    session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
    if session_id:
        return session_id

    session_id = headers.get("x-session-id") or headers.get("X-Session-ID")
    if session_id:
        return session_id

    # Try Authorization header for Bearer token
    auth_header = headers.get("authorization") or headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        # Extract bearer token and try to find associated session
        token = auth_header[7:]  # Remove "Bearer " prefix
        if token:
            # =====================================================================
            # O(1) LOOKUP: Use reverse mapping for efficient token lookup
            # =====================================================================
            # This prevents timing attacks and improves performance from O(N) to O(1)
            # The _access_token_mapping dictionary maps access tokens to user emails
            # This is maintained by store_session() and cleaned up by remove_session()
            store = get_oauth21_session_store()
            with store._lock:
                # Constant-time lookup using reverse mapping
                user_email = store._access_token_mapping.get(token)
                if user_email:
                    # Found session - return the session ID
                    session_info = store._sessions.get(user_email)
                    if session_info:
                        return session_info.get("session_id") or f"bearer_{user_email}"

        # =====================================================================
        # CRITICAL: Return None instead of creating unbound temporary session ID
        # =====================================================================
        # If no session is found for the bearer token, return None and let the
        # authentication middleware handle the token directly using functions like
        # ensure_session_from_access_token() or create_credentials_from_token_stateless().
        # Creating a temporary session ID that is never bound to a user causes
        # validation failures later in get_credentials_with_validation().
        # The middleware will extract the token and process it appropriately.
        return None

    return None


# =============================================================================
# OAuth21SessionStore - Main Session Management
# =============================================================================


class OAuth21SessionStore:
    """
    Global store for OAuth 2.1 authenticated sessions.

    This store maintains a mapping of user emails to their OAuth 2.1
    authenticated credentials, allowing Google services to access them.
    It also maintains a mapping from FastMCP session IDs to user emails.

    Security: Sessions are bound to specific users and can only access
    their own credentials.
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._mcp_session_mapping: Dict[
            str, str
        ] = {}  # Maps FastMCP session ID -> user email
        self._session_auth_binding: Dict[
            str, str
        ] = {}  # Maps session ID -> authenticated user email (immutable)
        self._access_token_mapping: Dict[
            str, str
        ] = {}  # Maps access token -> user email (for O(1) lookup, prevents timing attacks)
        self._oauth_states: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def _cleanup_expired_oauth_states_locked(self):
        """Remove expired OAuth state entries. Caller must hold lock."""
        now = datetime.now(timezone.utc)
        expired_states = [
            state
            for state, data in self._oauth_states.items()
            if data.get("expires_at") and data["expires_at"] <= now
        ]
        for state in expired_states:
            del self._oauth_states[state]
            logger.debug(
                "Removed expired OAuth state: %s",
                state[:8] if len(state) > 8 else state,
            )

    def store_oauth_state(
        self,
        state: str,
        session_id: Optional[str] = None,
        expires_in_seconds: int = 600,
    ) -> None:
        """Persist an OAuth state value for later validation."""
        if not state:
            raise ValueError("OAuth state must be provided")
        if expires_in_seconds < 0:
            raise ValueError("expires_in_seconds must be non-negative")

        with self._lock:
            self._cleanup_expired_oauth_states_locked()
            now = datetime.now(timezone.utc)
            expiry = now + timedelta(seconds=expires_in_seconds)
            self._oauth_states[state] = {
                "session_id": session_id,
                "expires_at": expiry,
                "created_at": now,
            }
            logger.debug(
                "Stored OAuth state %s (expires at %s)",
                state[:8] if len(state) > 8 else state,
                expiry.isoformat(),
            )

    def peek_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        """
        Peek at OAuth state metadata without consuming it.

        This method allows callers to inspect the state metadata (e.g., to extract
        session_id) without consuming the state. The state must still be validated
        and consumed via validate_and_consume_oauth_state() before use.

        Args:
            state: The OAuth state to peek at.

        Returns:
            Metadata associated with the state, or None if the state doesn't exist
            or has expired.

        Note:
            This method does NOT consume the state. The state must still be validated
            and consumed via validate_and_consume_oauth_state() to prevent reuse.
        """
        if not state:
            return None

        with self._lock:
            # Clean up expired states before peeking
            self._cleanup_expired_oauth_states_locked()
            # Return a copy of the state info to prevent external modification
            state_info = self._oauth_states.get(state)
            if state_info:
                # Return a shallow copy to prevent external modification
                return dict(state_info)
            return None

    def validate_and_consume_oauth_state(
        self,
        state: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Validate that a state value exists and consume it.

        Args:
            state: The OAuth state returned by Google.
            session_id: Optional session identifier that initiated the flow.

        Returns:
            Metadata associated with the state.

        Raises:
            ValueError: If the state is missing, expired, or does not match the session.
        """
        if not state:
            raise ValueError("Missing OAuth state parameter")

        with self._lock:
            self._cleanup_expired_oauth_states_locked()
            state_info = self._oauth_states.get(state)

            if not state_info:
                logger.error(
                    "SECURITY: OAuth callback received unknown or expired state"
                )
                raise ValueError("Invalid or expired OAuth state parameter")

            bound_session = state_info.get("session_id")
            if bound_session and session_id and bound_session != session_id:
                # Consume the state to prevent replay attempts
                del self._oauth_states[state]
                logger.error(
                    "SECURITY: OAuth state session mismatch (expected %s, got %s)",
                    bound_session,
                    session_id,
                )
                raise ValueError("OAuth state does not match the initiating session")

            # State is valid – consume it to prevent reuse
            del self._oauth_states[state]
            logger.debug(
                "Validated OAuth state %s",
                state[:8] if len(state) > 8 else state,
            )
            return state_info

    def store_session(
        self,
        user_email: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        token_uri: str = "https://oauth2.googleapis.com/token",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scopes: Optional[list] = None,
        expiry: Optional[Any] = None,
        session_id: Optional[str] = None,
        mcp_session_id: Optional[str] = None,
        issuer: Optional[str] = None,
    ):
        """
        Store OAuth 2.1 session information.

        Args:
            user_email: User's email address
            access_token: OAuth 2.1 access token
            refresh_token: OAuth 2.1 refresh token
            token_uri: Token endpoint URI
            client_id: OAuth client ID
            client_secret: OAuth client secret
            scopes: List of granted scopes
            expiry: Token expiry time
            session_id: OAuth 2.1 session ID
            mcp_session_id: FastMCP session ID to map to this user
            issuer: Token issuer (e.g., "https://accounts.google.com")
        """
        with self._lock:
            normalized_expiry = _normalize_expiry_to_naive_utc(expiry)
            # Store created_at timestamp for security validation (used by allow_recent_auth)
            # This allows us to limit the window of opportunity for the fallback mechanism
            created_at = datetime.now(timezone.utc)
            session_info = {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_uri": token_uri,
                "client_id": client_id,
                "client_secret": client_secret,
                "scopes": scopes or [],
                "expiry": normalized_expiry,
                "session_id": session_id,
                "mcp_session_id": mcp_session_id,
                "issuer": issuer,
                "created_at": created_at,  # Timestamp for security validation
            }

            # =====================================================================
            # Handle access token mapping update (remove old token if changed)
            # =====================================================================
            # If this user already has a session with a different access token,
            # remove the old token from the mapping to prevent stale entries
            existing_session = self._sessions.get(user_email)
            if existing_session:
                old_access_token = existing_session.get("access_token")
                if old_access_token and old_access_token != access_token:
                    # Remove old token mapping (token was refreshed/updated)
                    self._access_token_mapping.pop(old_access_token, None)

            self._sessions[user_email] = session_info

            # Store access token mapping for O(1) lookup (prevents timing attacks)
            # This enables efficient bearer token lookup in extract_session_from_headers
            if access_token:
                self._access_token_mapping[access_token] = user_email

            # Store MCP session mapping if provided
            if mcp_session_id:
                # Create immutable session binding (first binding wins, cannot be changed)
                if mcp_session_id not in self._session_auth_binding:
                    self._session_auth_binding[mcp_session_id] = user_email
                    logger.info(
                        f"Created immutable session binding: {mcp_session_id} -> {user_email}"
                    )
                elif self._session_auth_binding[mcp_session_id] != user_email:
                    # Security: Attempt to bind session to different user
                    logger.error(
                        f"SECURITY: Attempt to rebind session {mcp_session_id} from {self._session_auth_binding[mcp_session_id]} to {user_email}"
                    )
                    raise ValueError(
                        f"Session {mcp_session_id} is already bound to a different user"
                    )

                self._mcp_session_mapping[mcp_session_id] = user_email
                logger.info(
                    f"Stored OAuth 2.1 session for {user_email} (session_id: {session_id}, mcp_session_id: {mcp_session_id})"
                )
            else:
                logger.info(
                    f"Stored OAuth 2.1 session for {user_email} (session_id: {session_id})"
                )

            # Also create binding for the OAuth session ID
            if session_id and session_id not in self._session_auth_binding:
                self._session_auth_binding[session_id] = user_email

    def get_credentials(self, user_email: str) -> Optional[Credentials]:
        """
        Get Google credentials for a user from OAuth 2.1 session.

        Args:
            user_email: User's email address

        Returns:
            Google Credentials object or None
        """
        with self._lock:
            session_info = self._sessions.get(user_email)
            if not session_info:
                logger.debug(f"No OAuth 2.1 session found for {user_email}")
                return None

            try:
                # Create Google credentials from session info
                # Expected exceptions: KeyError (missing required field), TypeError (invalid type), ValueError (invalid value)
                credentials = Credentials(
                    token=session_info["access_token"],
                    refresh_token=session_info.get("refresh_token"),
                    token_uri=session_info["token_uri"],
                    client_id=session_info.get("client_id"),
                    client_secret=session_info.get("client_secret"),
                    scopes=session_info.get("scopes", []),
                    expiry=session_info.get("expiry"),
                )

                logger.debug(f"Retrieved OAuth 2.1 credentials for {user_email}")
                return credentials

            except (KeyError, TypeError, ValueError) as e:
                # Catch specific exceptions that can occur when creating Credentials
                # KeyError: Missing required field in session_info
                # TypeError: Invalid type for a Credentials parameter
                # ValueError: Invalid value for a Credentials parameter
                logger.error(f"Failed to create credentials for {user_email}: {e}")
                return None
            except Exception as e:
                # Catch-all for truly unexpected errors - log with full context
                # This should rarely happen, but we want to know if it does
                logger.error(
                    f"Unexpected error creating credentials for {user_email}: {e}",
                    exc_info=True,
                )
                return None

    def get_credentials_by_mcp_session(
        self, mcp_session_id: str
    ) -> Optional[Credentials]:
        """
        Get Google credentials using FastMCP session ID.

        Args:
            mcp_session_id: FastMCP session ID

        Returns:
            Google Credentials object or None
        """
        with self._lock:
            # Look up user email from MCP session mapping
            user_email = self._mcp_session_mapping.get(mcp_session_id)
            if not user_email:
                logger.debug(f"No user mapping found for MCP session {mcp_session_id}")
                return None

            logger.debug(f"Found user {user_email} for MCP session {mcp_session_id}")
            return self.get_credentials(user_email)

    def get_credentials_with_validation(
        self,
        requested_user_email: str,
        session_id: Optional[str] = None,
        auth_token_email: Optional[str] = None,
        allow_recent_auth: bool = False,
    ) -> Optional[Credentials]:
        """
        Get Google credentials with session validation.

        This method ensures that a session can only access credentials for its
        authenticated user, preventing cross-account access.

        Args:
            requested_user_email: The email of the user whose credentials are requested
            session_id: The current session ID (MCP or OAuth session)
            auth_token_email: Email from the verified auth token (if available)

        Returns:
            Google Credentials object if validation passes, None otherwise
        """
        with self._lock:
            # Priority 1: Check auth token email (most secure, from verified JWT)
            if auth_token_email:
                if auth_token_email != requested_user_email:
                    logger.error(
                        f"SECURITY VIOLATION: Token for {auth_token_email} attempted to access "
                        f"credentials for {requested_user_email}"
                    )
                    return None
                # Token email matches, allow access
                return self.get_credentials(requested_user_email)

            # Priority 2: Check session binding
            if session_id:
                bound_user = self._session_auth_binding.get(session_id)
                if bound_user:
                    if bound_user != requested_user_email:
                        logger.error(
                            f"SECURITY VIOLATION: Session {session_id} (bound to {bound_user}) "
                            f"attempted to access credentials for {requested_user_email}"
                        )
                        return None
                    # Session binding matches, allow access
                    return self.get_credentials(requested_user_email)

                # Check if this is an MCP session
                mcp_user = self._mcp_session_mapping.get(session_id)
                if mcp_user:
                    if mcp_user != requested_user_email:
                        logger.error(
                            f"SECURITY VIOLATION: MCP session {session_id} (user {mcp_user}) "
                            f"attempted to access credentials for {requested_user_email}"
                        )
                        return None
                    # MCP session matches, allow access
                    return self.get_credentials(requested_user_email)

            # =====================================================================
            # CRITICAL SECURITY: Fallback mechanism with strict time window
            # =====================================================================
            # Special case: Allow access if user has recently authenticated (for clients that don't send tokens)
            # CRITICAL SECURITY: This is ONLY allowed in stdio mode, NEVER in OAuth 2.1 mode
            # SECURITY ENHANCEMENT: Added timestamp check to limit window of opportunity
            # The session must have been created within the last 30 seconds to use this fallback
            # This drastically reduces the risk of unauthorized access
            if allow_recent_auth and requested_user_email in self._sessions:
                # Check transport mode to ensure this is only used in stdio
                try:
                    from core.config import get_transport_mode

                    transport_mode = get_transport_mode()
                    if transport_mode != "stdio":
                        logger.error(
                            f"SECURITY: Attempted to use allow_recent_auth in {transport_mode} mode. "
                            f"This is only allowed in stdio mode!"
                        )
                        return None
                except (ImportError, AttributeError, ValueError) as e:
                    # Catch specific exceptions for transport mode check
                    # ImportError: Module not found
                    # AttributeError: Function not found in module
                    # ValueError: Invalid transport mode value
                    logger.error(f"Failed to check transport mode: {e}")
                    return None
                except Exception as e:
                    # Catch-all for unexpected errors - fail securely
                    logger.error(
                        f"Unexpected error checking transport mode: {e}", exc_info=True
                    )
                    return None

                # =====================================================================
                # SECURITY: Validate session was created recently (within 30 seconds)
                # =====================================================================
                # This limits the window of opportunity for unauthorized access
                # A session created more than 30 seconds ago cannot use this fallback
                session_info = self._sessions.get(requested_user_email)
                if session_info:
                    created_at = session_info.get("created_at")
                    if created_at:
                        # Calculate time since session creation
                        time_since_creation = datetime.now(timezone.utc) - created_at
                        # Only allow if session was created within last 30 seconds
                        if time_since_creation.total_seconds() > 30:
                            logger.warning(
                                f"SECURITY: Denied allow_recent_auth for {requested_user_email} - "
                                f"session created {time_since_creation.total_seconds():.1f} seconds ago "
                                f"(max 30 seconds allowed)"
                            )
                            return None
                    else:
                        # No created_at timestamp - cannot validate, deny access
                        logger.warning(
                            f"SECURITY: Denied allow_recent_auth for {requested_user_email} - "
                            f"session has no created_at timestamp (cannot validate recency)"
                        )
                        return None

                logger.info(
                    f"Allowing credential access for {requested_user_email} based on recent authentication "
                    f"(stdio mode only - session created within last 30 seconds)"
                )
                return self.get_credentials(requested_user_email)

            # No session or token info available - deny access for security
            logger.warning(
                f"Credential access denied for {requested_user_email}: No valid session or token"
            )
            return None

    def get_user_by_mcp_session(self, mcp_session_id: str) -> Optional[str]:
        """
        Get user email by FastMCP session ID.

        Args:
            mcp_session_id: FastMCP session ID

        Returns:
            User email or None
        """
        with self._lock:
            return self._mcp_session_mapping.get(mcp_session_id)

    def get_session_info(self, user_email: str) -> Optional[Dict[str, Any]]:
        """
        Get complete session information including issuer.

        Args:
            user_email: User's email address

        Returns:
            Session information dictionary or None
        """
        with self._lock:
            return self._sessions.get(user_email)

    def remove_session(self, user_email: str):
        """Remove session for a user."""
        with self._lock:
            # Get session info before removing (for cleanup)
            session_info = self._sessions.pop(user_email, None)
            if not session_info:
                # Session doesn't exist, nothing to remove
                return

            # Extract session IDs for cleanup
            mcp_session_id = session_info.get("mcp_session_id")
            session_id = session_info.get("session_id")
            access_token = session_info.get("access_token")

            # Remove from access token mapping if exists (using pop for safe removal)
            if access_token:
                self._access_token_mapping.pop(access_token, None)

            # Remove from MCP mapping if exists (using pop for safe removal)
            if mcp_session_id:
                self._mcp_session_mapping.pop(mcp_session_id, None)
                # Also remove from auth binding
                self._session_auth_binding.pop(mcp_session_id, None)
                logger.info(
                    f"Removed OAuth 2.1 session for {user_email} and MCP mapping for {mcp_session_id}"
                )
            else:
                logger.info(f"Removed OAuth 2.1 session for {user_email}")

            # Remove OAuth session binding if exists (using pop for safe removal)
            if session_id:
                self._session_auth_binding.pop(session_id, None)

    def has_session(self, user_email: str) -> bool:
        """Check if a user has an active session."""
        with self._lock:
            return user_email in self._sessions

    def has_mcp_session(self, mcp_session_id: str) -> bool:
        """Check if an MCP session has an associated user session."""
        with self._lock:
            return mcp_session_id in self._mcp_session_mapping

    def get_single_user_email(self) -> Optional[str]:
        """Return the sole authenticated user email when exactly one session exists."""
        with self._lock:
            if len(self._sessions) == 1:
                return next(iter(self._sessions))
            return None

    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "users": list(self._sessions.keys()),
                "mcp_session_mappings": len(self._mcp_session_mapping),
                "mcp_sessions": list(self._mcp_session_mapping.keys()),
                "access_token_mappings": len(self._access_token_mapping),
            }


# Global instance
_global_store = OAuth21SessionStore()


def get_oauth21_session_store() -> OAuth21SessionStore:
    """Get the global OAuth 2.1 session store."""
    return _global_store


# =============================================================================
# Google Credentials Bridge (absorbed from oauth21_google_bridge.py)
# =============================================================================

# Global auth provider instance (set during server initialization)
_auth_provider = None


def set_auth_provider(provider):
    """Set the global auth provider instance."""
    global _auth_provider
    _auth_provider = provider
    logger.debug("OAuth 2.1 session store configured")


def get_auth_provider():
    """Get the global auth provider instance."""
    return _auth_provider


def _resolve_client_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Resolve OAuth client credentials from the active provider or configuration."""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None

    if _auth_provider:
        client_id = getattr(_auth_provider, "_upstream_client_id", None)
        secret_obj = getattr(_auth_provider, "_upstream_client_secret", None)
        if secret_obj is not None:
            if hasattr(secret_obj, "get_secret_value"):
                try:
                    client_secret = secret_obj.get_secret_value()  # type: ignore[call-arg]
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug(
                        f"Failed to resolve client secret from provider: {exc}"
                    )
            elif isinstance(secret_obj, str):
                client_secret = secret_obj

    if not client_id or not client_secret:
        try:
            from auth.oauth_config import get_oauth_config

            cfg = get_oauth_config()
            client_id = client_id or cfg.client_id
            client_secret = client_secret or cfg.client_secret
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to resolve client credentials from config: {exc}")

    return client_id, client_secret


def _build_credentials_from_provider(
    access_token: AccessToken,
) -> Optional[Credentials]:
    """Construct Google credentials from the provider cache."""
    if not _auth_provider:
        return None

    access_entry = getattr(_auth_provider, "_access_tokens", {}).get(access_token.token)
    if not access_entry:
        access_entry = access_token

    client_id, client_secret = _resolve_client_credentials()

    refresh_token_value = getattr(_auth_provider, "_access_to_refresh", {}).get(
        access_token.token
    )
    refresh_token_obj = None
    if refresh_token_value:
        refresh_token_obj = getattr(_auth_provider, "_refresh_tokens", {}).get(
            refresh_token_value
        )

    expiry = None
    expires_at = getattr(access_entry, "expires_at", None)
    if expires_at:
        try:
            expiry_candidate = datetime.fromtimestamp(expires_at, tz=timezone.utc)
            expiry = _normalize_expiry_to_naive_utc(expiry_candidate)
        except Exception:  # pragma: no cover - defensive
            expiry = None

    scopes = getattr(access_entry, "scopes", None)

    return Credentials(
        token=access_token.token,
        refresh_token=refresh_token_obj.token if refresh_token_obj else None,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
        expiry=expiry,
    )


def ensure_session_from_access_token(
    access_token: AccessToken,
    user_email: Optional[str],
    mcp_session_id: Optional[str] = None,
) -> Optional[Credentials]:
    """Ensure credentials derived from an access token are cached and returned."""

    if not access_token:
        return None

    email = user_email
    if not email and getattr(access_token, "claims", None):
        email = access_token.claims.get("email")

    credentials = _build_credentials_from_provider(access_token)
    store_expiry: Optional[datetime] = None

    if credentials is None:
        client_id, client_secret = _resolve_client_credentials()
        expiry = None
        expires_at = getattr(access_token, "expires_at", None)
        if expires_at:
            try:
                expiry = datetime.fromtimestamp(expires_at, tz=timezone.utc)
            except Exception:  # pragma: no cover - defensive
                expiry = None

        normalized_expiry = _normalize_expiry_to_naive_utc(expiry)
        credentials = Credentials(
            token=access_token.token,
            refresh_token=None,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=getattr(access_token, "scopes", None),
            expiry=normalized_expiry,
        )
        store_expiry = expiry
    else:
        store_expiry = credentials.expiry

    if email:
        try:
            store = get_oauth21_session_store()
            store.store_session(
                user_email=email,
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_uri=credentials.token_uri,
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                scopes=credentials.scopes,
                expiry=store_expiry,
                session_id=f"google_{email}",
                mcp_session_id=mcp_session_id,
                issuer="https://accounts.google.com",
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Failed to cache credentials for {email}: {exc}")

    return credentials


def create_credentials_from_token_stateless(
    access_token: Union[AccessToken, Any],
    user_email: Optional[str] = None,
    scopes: Optional[list] = None,
    expires_at: Optional[int] = None,
) -> Optional[Credentials]:
    """
    Create Google credentials from access token WITHOUT storing in session store.

    This is for stateless mode (Cloud Run) where tokens are passed per-request.
    Each request is independent - no server-side state is maintained.

    CRITICAL: This function does NOT call store.store_session() - it only creates
    credentials for the current request. This ensures true stateless operation.

    Args:
        access_token: AccessToken object or AccessTokenData dataclass with token attribute
        user_email: User email (optional, for logging only)
        scopes: Token scopes (optional, extracted from access_token if not provided)
        expires_at: Token expiration timestamp (optional, extracted from access_token if not provided)

    Returns:
        Google Credentials object (NOT stored in session store)
    """
    # =====================================================================
    # STEP 1: Validate input
    # =====================================================================
    # Ensure we have an access token to work with
    if not access_token:
        return None

    # =====================================================================
    # STEP 2: Extract token string from access_token object
    # =====================================================================
    # The access_token parameter can be:
    # - AccessToken object (from FastMCP auth provider) with .token attribute
    # - AccessTokenData dataclass (from middleware) with .token attribute
    # - String (direct token string, though less common)
    # We need to handle all cases to be flexible
    token_str = None
    if hasattr(access_token, "token"):
        # Most common case: AccessToken or AccessTokenData object
        token_str = access_token.token
    elif isinstance(access_token, str):
        # Direct token string (fallback case)
        token_str = access_token
    else:
        # Invalid type - cannot proceed
        logger.error("Invalid access_token type for stateless credentials creation")
        return None

    # Validate that we successfully extracted a token string
    if not token_str:
        logger.error("No token string found in access_token")
        return None

    # =====================================================================
    # STEP 3: Extract token metadata (scopes and expiration)
    # =====================================================================
    # Extract scopes from access_token if not explicitly provided
    # Scopes define what permissions the token has (e.g., Gmail read, Drive write)
    if scopes is None:
        scopes = getattr(access_token, "scopes", None)

    # Extract expires_at from access_token if not explicitly provided
    # expires_at is a Unix timestamp (seconds since epoch) indicating when
    # the token will expire and need to be refreshed
    if expires_at is None:
        expires_at = getattr(access_token, "expires_at", None)

    # =====================================================================
    # STEP 4: Calculate token expiry datetime
    # =====================================================================
    # Convert expires_at timestamp to datetime object for google-auth library
    # The google-auth Credentials class requires a datetime object for expiry
    expiry = None
    if expires_at:
        try:
            # Convert Unix timestamp to timezone-aware datetime (UTC)
            expiry = datetime.fromtimestamp(expires_at, tz=timezone.utc)
        except Exception as exc:
            # If timestamp parsing fails, log and continue with None
            # We'll use a default expiry below
            logger.debug(f"Failed to parse expires_at timestamp: {exc}")
            expiry = None

    # =====================================================================
    # STEP 5: Normalize expiry to naive UTC datetime
    # =====================================================================
    # CRITICAL: google-auth library requires timezone-naive UTC datetimes
    # This is a quirk of the library - it assumes naive datetimes are UTC
    # The _normalize_expiry_to_naive_utc helper handles the conversion
    normalized_expiry = _normalize_expiry_to_naive_utc(expiry)

    # =====================================================================
    # STEP 6: Set default expiry if not provided
    # =====================================================================
    # If no expiry was provided or parsing failed, default to 1 hour from now
    # This is typical for Google OAuth access tokens (they usually expire
    # after 1 hour and need to be refreshed)
    if normalized_expiry is None:
        normalized_expiry = _normalize_expiry_to_naive_utc(
            datetime.now(timezone.utc) + timedelta(hours=1)
        )

    # =====================================================================
    # STEP 7: Resolve OAuth client credentials
    # =====================================================================
    # Get client_id and client_secret from:
    # 1. Auth provider (if configured)
    # 2. OAuth config (environment variables or config file)
    # These are required to create Credentials object, even though we're
    # using an access token (not initiating OAuth flow)
    client_id, client_secret = _resolve_client_credentials()

    # Validate that we have both client_id and client_secret
    # Without these, we cannot create valid Credentials object
    if not client_id or not client_secret:
        logger.error(
            "Cannot create stateless credentials: client_id or client_secret not configured"
        )
        return None

    # =====================================================================
    # STEP 8: Create Credentials object WITHOUT storing in session store
    # =====================================================================
    # CRITICAL: This is the key difference from ensure_session_from_access_token()
    # We create the Credentials object but do NOT call store.store_session()
    # This ensures:
    # - True stateless operation (no server-side state)
    # - Cloud Run compatibility (instances can scale to zero)
    # - Per-request independence (each request uses its own token)
    # - No token caching between requests (prevents cross-user contamination)

    credentials = Credentials(
        token=token_str,  # The actual OAuth access token (ya29.* format)
        refresh_token=None,  # No refresh token in stateless mode (tokens are passed per-request)
        token_uri="https://oauth2.googleapis.com/token",  # Google OAuth token endpoint
        client_id=client_id,  # OAuth client ID (from config)
        client_secret=client_secret,  # OAuth client secret (from config)
        scopes=scopes,  # Token scopes (what permissions the token has)
        expiry=normalized_expiry,  # When the token expires (naive UTC datetime)
    )

    # Log that credentials were created (for debugging)
    # Note: We explicitly log that credentials are NOT stored in session store
    # This helps distinguish stateless mode from session mode in logs
    logger.debug(
        f"Created stateless Google credentials for {user_email or 'unknown'} (NOT stored in session store)"
    )
    return credentials


def get_credentials_from_token(
    access_token: str, user_email: Optional[str] = None
) -> Optional[Credentials]:
    """
    Convert a bearer token to Google credentials.

    Args:
        access_token: The bearer token
        user_email: Optional user email for session lookup

    Returns:
        Google Credentials object or None
    """
    try:
        store = get_oauth21_session_store()

        # If we have user_email, try to get credentials from store
        if user_email:
            credentials = store.get_credentials(user_email)
            if credentials and credentials.token == access_token:
                logger.debug(f"Found matching credentials from store for {user_email}")
                return credentials

        # If the FastMCP provider is managing tokens, sync from provider storage
        if _auth_provider:
            access_record = getattr(_auth_provider, "_access_tokens", {}).get(
                access_token
            )
            if access_record:
                logger.debug("Building credentials from FastMCP provider cache")
                return ensure_session_from_access_token(access_record, user_email)

        # Otherwise, create minimal credentials with just the access token
        # Assume token is valid for 1 hour (typical for Google tokens)
        expiry = _normalize_expiry_to_naive_utc(
            datetime.now(timezone.utc) + timedelta(hours=1)
        )
        client_id, client_secret = _resolve_client_credentials()

        credentials = Credentials(
            token=access_token,
            refresh_token=None,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=None,
            expiry=expiry,
        )

        logger.debug("Created fallback Google credentials from bearer token")
        return credentials

    except (KeyError, TypeError, ValueError, AttributeError) as e:
        # Catch specific exceptions that can occur during credential creation
        # KeyError: Missing key in token_response or session_info
        # TypeError: Invalid type for Credentials parameters
        # ValueError: Invalid value for Credentials parameters
        # AttributeError: Missing attribute on access_record or provider
        logger.error(f"Failed to create Google credentials from token: {e}")
        return None
    except Exception as e:
        # Catch-all for truly unexpected errors - log with full context
        logger.error(
            f"Unexpected error creating Google credentials from token: {e}",
            exc_info=True,
        )
        return None


def store_token_session(
    token_response: dict, user_email: str, mcp_session_id: Optional[str] = None
) -> str:
    """
    Store a token response in the session store.

    Args:
        token_response: OAuth token response from Google
        user_email: User's email address
        mcp_session_id: Optional FastMCP session ID to map to this user

    Returns:
        Session ID
    """
    if not _auth_provider:
        logger.error("Auth provider not configured")
        return ""

    try:
        # Try to get FastMCP session ID from context if not provided
        if not mcp_session_id:
            try:
                from core.context import get_fastmcp_session_id

                mcp_session_id = get_fastmcp_session_id()
                if mcp_session_id:
                    logger.debug(
                        f"Got FastMCP session ID from context: {mcp_session_id}"
                    )
            except (ImportError, AttributeError) as e:
                # Catch specific exceptions for context retrieval
                # ImportError: Module not found
                # AttributeError: Function not found in module
                logger.debug(f"Could not get FastMCP session from context: {e}")
            except Exception as e:
                # Catch-all for unexpected errors - log but don't fail
                logger.debug(
                    f"Unexpected error getting FastMCP session from context: {e}",
                    exc_info=True,
                )

        # Store session in OAuth21SessionStore
        store = get_oauth21_session_store()

        session_id = f"google_{user_email}"
        client_id, client_secret = _resolve_client_credentials()
        scopes = token_response.get("scope", "")
        scopes_list = scopes.split() if scopes else None
        expiry = datetime.now(timezone.utc) + timedelta(
            seconds=token_response.get("expires_in", 3600)
        )

        store.store_session(
            user_email=user_email,
            access_token=token_response.get("access_token"),
            refresh_token=token_response.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=scopes_list,
            expiry=expiry,
            session_id=session_id,
            mcp_session_id=mcp_session_id,
            issuer="https://accounts.google.com",
        )

        if mcp_session_id:
            logger.info(
                f"Stored token session for {user_email} with MCP session {mcp_session_id}"
            )
        else:
            logger.info(f"Stored token session for {user_email}")

        return session_id

    except (KeyError, TypeError, ValueError) as e:
        # Catch specific exceptions that can occur during session storage
        # KeyError: Missing key in token_response
        # TypeError: Invalid type for session storage
        # ValueError: Invalid value for session storage
        logger.error(f"Failed to store token session: {e}")
        return ""
    except Exception as e:
        # Catch-all for truly unexpected errors - log with full context
        logger.error(f"Unexpected error storing token session: {e}", exc_info=True)
        return ""
