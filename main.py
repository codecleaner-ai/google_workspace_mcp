import argparse
import logging
import os
import sys
from dotenv import load_dotenv
import uvicorn
from starlette.applications import Starlette

from auth.oauth_config import reload_oauth_config
from core.log_formatter import EnhancedLogFormatter, setup_enhanced_logging
from core.logger import initialize_loggers
from core import server as mcp_server
from core.config import get_transport_mode, set_transport_mode
from auth.api_key_middleware import APIKeyMiddleware
from auth.mcp_session_middleware import MCPSessionMiddleware
from core.tool_registry import (
    set_enabled_tools as set_enabled_tool_names,
    wrap_server_tool_method,
    filter_server_tools,
)
from auth.scopes import set_enabled_tools


# Load environment variables from .env.local (local overrides, git-ignored) or .env (shared defaults)
# This matches the pattern used by the restart-server.sh script
env_dir = os.path.dirname(os.path.abspath(__file__))
env_local_path = os.path.join(env_dir, ".env.local")
env_path = os.path.join(env_dir, ".env")

# Try .env.local first (local overrides), then .env (shared defaults)
if os.path.exists(env_local_path):
    load_dotenv(dotenv_path=env_local_path, override=True)
elif os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)

# ============================================================================
# IMPORT ALL TOOL MODULES TO REGISTER THEIR @server.tool() DECORATORS
# ============================================================================
# CRITICAL: Tool modules must be imported for their decorators to execute
# and register tools with the server. Without these imports, only tools
# defined directly in core.server (like start_google_auth) will be available.
# ============================================================================
import gmail.gmail_tools  # noqa: F401
import gdrive.drive_tools  # noqa: F401
import gcalendar.calendar_tools  # noqa: F401
import gdocs.docs_tools  # noqa: F401
import gsheets.sheets_tools  # noqa: F401
import gchat.chat_tools  # noqa: F401
import gforms.forms_tools  # noqa: F401
import gslides.slides_tools  # noqa: F401
import gtasks.tasks_tools  # noqa: F401
import gsearch.search_tools  # noqa: F401

# Configure tool registration
wrap_server_tool_method(mcp_server.server)

# Enable all tools and services by default
all_services = [
    "gmail",
    "drive",
    "calendar",
    "docs",
    "sheets",
    "chat",
    "forms",
    "slides",
    "tasks",
    "search",
]
set_enabled_tools(all_services)  # Set enabled services for scopes
set_enabled_tool_names(None)  # Don't filter individual tools - enable all

# Filter tools based on configuration (if any tool filtering is configured)
filter_server_tools(mcp_server.server)

# Suppress googleapiclient discovery cache warning
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

reload_oauth_config()

# Initialize structured logging (file-based, JSON format)
# This must be called before basicConfig to ensure proper logger setup
initialize_loggers()

# Configure console logging with enhanced formatter (for human-readable output)
# This is separate from file logging and provides nice console output
debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
console_log_level = logging.DEBUG if debug_mode else logging.INFO
setup_enhanced_logging(log_level=console_log_level, use_colors=True)

# Get the standard logger for this module
logger = logging.getLogger(__name__)


def safe_print(text):
    # Don't print to stderr when running as MCP server via uvx to avoid JSON parsing errors
    # Check if we're running as MCP server (no TTY and uvx in process name)
    if not sys.stderr.isatty():
        # Running as MCP server, suppress output to avoid JSON parsing errors
        logger.debug(f"[MCP Server] {text}")
        return

    try:
        print(text, file=sys.stderr)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode(), file=sys.stderr)


def configure_safe_logging():
    class SafeEnhancedFormatter(EnhancedLogFormatter):
        """Enhanced ASCII formatter with additional Windows safety."""

        def format(self, record):
            try:
                return super().format(record)
            except UnicodeEncodeError:
                # Fallback to ASCII-safe formatting
                service_prefix = self._get_ascii_prefix(record.name, record.levelname)
                safe_msg = (
                    str(record.getMessage())
                    .encode("ascii", errors="replace")
                    .decode("ascii")
                )
                return f"{service_prefix} {safe_msg}"

    # Replace all console handlers' formatters with safe enhanced ones
    for handler in logging.root.handlers:
        # Only apply to console/stream handlers, keep file handlers as-is
        if isinstance(handler, logging.StreamHandler) and handler.stream.name in [
            "<stderr>",
            "<stdout>",
        ]:
            safe_formatter = SafeEnhancedFormatter(use_colors=True)
            handler.setFormatter(safe_formatter)


def main():
    """
    Main entry point for the Google Workspace MCP server.

    This function sets up the server, configures the transport mode,
    and starts the Uvicorn server.
    """
    parser = argparse.ArgumentParser(description="Google Workspace MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default=os.getenv("TRANSPORT", "stdio"),
        help="Transport mode (stdio or streamable-http)",
    )
    args = parser.parse_args()

    # Set the transport mode based on command-line arguments or environment variables
    set_transport_mode(args.transport)

    if get_transport_mode() == "streamable-http":
        # Configure authentication providers for HTTP transport
        # This MUST be called before the server starts
        mcp_server.configure_server_for_http()

        # =====================================================================
        # Get FastMCP application first to access its lifespan
        # =====================================================================
        mcp_app = mcp_server.server.streamable_http_app()

        # =====================================================================
        # Create a top-level Starlette application with FastMCP lifespan
        # =====================================================================
        # CRITICAL: Pass the FastMCP lifespan to Starlette so that
        # StreamableHTTPSessionManager task group is properly initialized.
        # Without this, MCP client connections will fail with "Task group is not initialized".
        app = Starlette(debug=True, lifespan=mcp_app.lifespan)

        # =====================================================================
        # Register Middleware (Order is CRITICAL: Outer to Inner)
        # =====================================================================
        # 1. APIKeyMiddleware (Outermost): Runs first, protects everything.
        app.add_middleware(APIKeyMiddleware)

        # 2. MCPSessionMiddleware: Runs second, handles session context.
        app.add_middleware(MCPSessionMiddleware)

        # =====================================================================
        # Mount the FastMCP application as a sub-app
        # =====================================================================
        # All requests that pass the middleware will be forwarded to the
        # FastMCP server's own HTTP application.
        app.mount("/", app=mcp_app)

        # Get port from environment variable or default to 3003
        port = int(os.getenv("PORT", 3003))

        # Run the main Starlette application
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        # For stdio transport, run the MCP server directly
        mcp_server.server.run()


if __name__ == "__main__":
    main()
