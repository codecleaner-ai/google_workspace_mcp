"""
Logger utility for Google Workspace MCP Server

Provides structured logging similar to the backend logger utility, with support for:
- DEBUG mode (controlled by DEBUG_MODE environment variable)
- Separate log files for different severity levels
- JSON formatting for structured logs
- File logging to logs/server.log
"""

import logging
import json
import os
import sys
from datetime import datetime, timezone

# Global logger instances (lazy initialization)
info_logger = None
error_logger = None
debug_logger = None

# Environment variables
debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
log_folder = "logs"

# ISO 8601 UTC with microsecond precision (same shape as JsonFormatter timestamps)
_TIMESTAMP_FMT = "%Y-%m-%dT%H:%M:%S.%f"
_TIMESTAMP_SUFFIX = "Z"


def get_timestamp_with_microseconds() -> str:
    """
    Return current time as ISO 8601 UTC with microsecond precision.

    Use for API parameters (e.g. timeMin/timeMax) or log payloads when exact
    ordering or RFC3339 format is needed. Matches the format used by this
    package's JsonFormatter for consistency.
    """
    return datetime.now(timezone.utc).strftime(_TIMESTAMP_FMT) + _TIMESTAMP_SUFFIX


class JsonFormatter(logging.Formatter):
    """
    Formats log records as JSON objects.
    Combines standard log record attributes with the message
    (expected to be a dictionary or string).
    """

    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            return ct.strftime(datefmt)
        return super().formatTime(record, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        log_object = {
            # Use formatTime for consistent timestamp formatting
            "timestamp": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%f") + "Z",
            "level": record.levelname,
        }
        # Check if the message is already a dictionary
        if isinstance(record.msg, dict):
            log_object.update(record.msg)
        else:
            # Fallback: use the standard message formatting
            log_object["message"] = record.getMessage()

        # Add exception info if present
        if record.exc_info:
            # Use formatException to get the traceback string
            log_object["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            log_object["stack_info"] = self.formatStack(record.stack_info)

        # Add module, function, and line number for better debugging
        log_object["module"] = record.module
        log_object["function"] = record.funcName
        log_object["line"] = record.lineno

        return json.dumps(log_object)


def setup_info_logger(output_folder: str):
    """
    Set up the info logger.

    Args:
        output_folder: Directory where log files will be stored

    Returns:
        Configured info logger instance
    """
    global info_logger

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Setup the info logger - give it a name
    _info_logger = logging.getLogger("mcp_info_logger")

    # Check if the handler already exists
    if not _info_logger.handlers:
        # Define the file path where we store the log
        log_file = os.path.join(output_folder, "server.log")

        # Set the log level to INFO
        _info_logger.setLevel(logging.INFO)

        # Create a file handler (append mode)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        # Ensure all levels above INFO are logged
        file_handler.setLevel(logging.INFO)

        # Create a JSON formatter
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%f")
        # Add the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        _info_logger.addHandler(file_handler)

        # Prevent propagation to the root logger
        _info_logger.propagate = False

    info_logger = _info_logger
    return _info_logger


def setup_error_logger(output_folder: str):
    """
    Set up the error logger.

    Args:
        output_folder: Directory where log files will be stored

    Returns:
        Configured error logger instance
    """
    global error_logger

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Setup the error logger - give it a name
    _error_logger = logging.getLogger("mcp_error_logger")

    # Check if the handler already exists
    if not _error_logger.handlers:
        # Define the file path where we store the log
        log_file = os.path.join(output_folder, "server.log")

        # Set the log level to ERROR
        _error_logger.setLevel(logging.ERROR)

        # Create a file handler (append mode)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        # Ensure all levels above ERROR are logged
        file_handler.setLevel(logging.ERROR)

        # Create a JSON formatter
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%f")
        # Add the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        _error_logger.addHandler(file_handler)

        # Prevent propagation to the root logger
        _error_logger.propagate = False

    error_logger = _error_logger
    return _error_logger


def setup_debug_logger(output_folder: str):
    """
    Set up the debug logger.

    Args:
        output_folder: Directory where log files will be stored

    Returns:
        Configured debug logger instance
    """
    global debug_logger

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Setup the debug logger - give it a name
    _debug_logger = logging.getLogger("mcp_debug_logger")

    # Check if the handler already exists to prevent duplicates
    if not _debug_logger.handlers:
        # Define the file path where we store the log
        log_file = os.path.join(output_folder, "server.log")

        # Set the log level to DEBUG (captures everything)
        _debug_logger.setLevel(logging.DEBUG)

        # Create a file handler (append mode)
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        # Ensure all levels from DEBUG upwards are logged by this handler
        file_handler.setLevel(logging.DEBUG)

        # Create a JSON formatter
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%f")
        # Add the formatter to the file handler
        file_handler.setFormatter(formatter)

        # Add the handler to the logger
        _debug_logger.addHandler(file_handler)

        # Prevent propagation to the root logger
        _debug_logger.propagate = False

    debug_logger = _debug_logger
    return _debug_logger


def initialize_loggers():
    """
    Initialize the appropriate loggers based on the environment and DEBUG_MODE setting.
    Should be called once at application startup.

    This function:
    - Sets up a file handler on the root logger that writes to logs/server.log
    - Sets the root logger level based on DEBUG_MODE
    - Ensures all child loggers (created with logging.getLogger(__name__)) inherit the file handler
    """
    global info_logger, error_logger, debug_logger, log_folder, debug_mode

    # Reload environment variables
    debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"

    print(f"Initializing MCP loggers. Debug Mode: {debug_mode}")

    # Create the output folder if it does not exist
    os.makedirs(log_folder, exist_ok=True)

    # Get the root logger
    root_logger = logging.getLogger()

    # Check if we already have a file handler for logs/server.log
    log_file = os.path.join(log_folder, "server.log")
    has_file_handler = False
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename.endswith(
            "server.log"
        ):
            has_file_handler = True
            break

    # Add file handler to root logger if it doesn't exist
    if not has_file_handler:
        file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")

        # Set file handler level based on DEBUG_MODE
        if debug_mode:
            file_handler.setLevel(logging.DEBUG)
        else:
            file_handler.setLevel(logging.INFO)

        # Create a JSON formatter
        formatter = JsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S.%f")
        file_handler.setFormatter(formatter)

        # Add the handler to the root logger
        root_logger.addHandler(file_handler)

        print(
            f"File logging configured to: {log_file} (level: {'DEBUG' if debug_mode else 'INFO'})"
        )

    # Configure root logger level based on DEBUG_MODE
    if debug_mode:
        root_logger.setLevel(logging.DEBUG)
        print("Root logger level set to DEBUG")
    else:
        root_logger.setLevel(logging.INFO)
        print("Root logger level set to INFO")

    # Initialize the structured loggers for write_log() function
    info_logger = setup_info_logger(log_folder)
    error_logger = setup_error_logger(log_folder)
    if debug_mode:
        debug_logger = setup_debug_logger(log_folder)


def write_log(log_entry: dict, severity: str):
    """
    Writes a log entry to the appropriate destination based on severity and DEBUG_MODE.

    Args:
        log_entry (dict): The log data. Similar structure to GCP's log_struct payload.
        severity (str): The severity level (e.g., "INFO", "ERROR", "DEBUG", "WARNING").
    """
    global info_logger, error_logger, debug_logger, debug_mode

    # Convert severity string to uppercase for consistent comparison
    severity = severity.upper()

    # Error logging (always available)
    if severity == "ERROR":
        if error_logger:
            error_logger.error(log_entry)
        else:
            print("Local ERROR log (logger not started)", file=sys.stderr)

    # Info logging (always available)
    elif severity == "INFO":
        if info_logger:
            info_logger.info(log_entry)
        else:
            print("Local INFO log (logger not started)", file=sys.stderr)

    # Warning logging (goes to info logger)
    elif severity == "WARNING":
        if info_logger:
            info_logger.warning(log_entry)
        else:
            print("Local WARNING log (logger not started)", file=sys.stderr)

    # Debug logging (only if DEBUG_MODE is enabled)
    if debug_mode:
        if debug_logger:
            # Log using the method corresponding to the original severity
            if severity == "ERROR":
                debug_logger.error(log_entry)
            elif severity == "WARNING":
                debug_logger.warning(log_entry)
            elif severity == "INFO":
                debug_logger.info(log_entry)
            elif severity == "DEBUG":
                debug_logger.debug(log_entry)
            else:
                # Handle custom or other severities
                debug_logger.log(
                    # Log with INFO level but include original severity
                    logging.INFO,
                    f"[{severity}] {log_entry}",
                )
        else:
            # This case should ideally not happen if initialize_loggers worked
            # and debug_mode is True. Print fallback if it does.
            print(
                f"Local {severity} log (debug logger not init despite debug_mode=True): {json.dumps(log_entry)}",
                file=sys.stderr,
            )
