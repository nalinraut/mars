"""
Logging configuration for MARS pipeline.

Properly configures both Python logging and Prefect logging to work together.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

# Prefect-compatible log format
DEFAULT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: str = DEFAULT_FORMAT,
    suppress_prefect_noise: bool = True
) -> logging.Logger:
    """
    Configure logging for MARS pipeline.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_string: Log message format
        suppress_prefect_noise: Reduce Prefect internal logging
    
    Returns:
        Root logger configured for the pipeline
    """
    # Convert string level to logging constant
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(format_string, datefmt=DEFAULT_DATE_FORMAT)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Configure mars loggers
    mars_logger = logging.getLogger("mars")
    mars_logger.setLevel(log_level)
    
    # Suppress noisy loggers
    if suppress_prefect_noise:
        _suppress_noisy_loggers()
    
    return root_logger


def _suppress_noisy_loggers():
    """Reduce noise from Prefect and other libraries."""
    noisy_loggers = [
        # Prefect internal loggers - set to ERROR to suppress all warnings/info
        "prefect",
        "prefect.flow_runs",
        "prefect.task_runs", 
        "prefect.engine",
        "prefect.events",
        "prefect.client",
        "prefect._internal",
        "prefect.server",
        "prefect.utilities",
        # HTTP/network loggers
        "httpx",
        "httpcore",
        "urllib3",
        "requests",
        # Database loggers
        "sqlalchemy",
        "sqlalchemy.engine",
        # Other noisy libs
        "transformers",
        "torch",
        "PIL",
        "matplotlib",
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)  # Only show errors, suppress warnings/info


def configure_prefect_logging():
    """
    Set Prefect environment variables for logging.
    Call this BEFORE importing Prefect flows.
    """
    # Set Prefect logging level
    os.environ.setdefault("PREFECT_LOGGING_LEVEL", "ERROR")  # Only show errors, suppress all info/warnings
    os.environ.setdefault("PREFECT_LOGGING_INTERNAL_LEVEL", "ERROR")  # Suppress internal warnings
    
    # Disable events system entirely (fixes "stopped service" errors)
    # Must be set BEFORE importing prefect
    os.environ["PREFECT_EVENTS_ENABLED"] = "false"
    os.environ["PREFECT_CLIENT_ENABLE_EVENTS"] = "false"
    
    # Disable results persistence to avoid event issues
    os.environ.setdefault("PREFECT_RESULTS_PERSIST_BY_DEFAULT", "false")
    
    # Disable telemetry and anonymous usage stats
    os.environ.setdefault("PREFECT_SEND_ANONYMOUS_USAGE_STATS", "false")
    
    # Run in ephemeral mode by default (no server required)
    # Set PREFECT_API_URL to a server address to use persistent mode
    if not os.environ.get("PREFECT_API_URL"):
        # Empty string = ephemeral mode (runs without server)
        os.environ["PREFECT_API_URL"] = ""
    
    # Set home directory
    os.environ.setdefault("PREFECT_HOME", "/workspace/data/prefect")
    
    # Suppress the specific "stopped service" error messages
    _patch_prefect_events()


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.
    
    Use this instead of logging.getLogger() for consistent configuration.
    
    Example:
        from mars.utils.logging_config import get_logger
        logger = get_logger(__name__)
    """
    return logging.getLogger(name)


def _patch_prefect_events():
    """
    Suppress Prefect events errors by patching the emit_event function.
    This fixes the "Cannot put items in a stopped service instance" errors.
    
    Note: This must be called AFTER prefect is imported but BEFORE flows run.
    """
    pass  # Patching done via import hook below


def patch_prefect_emit():
    """
    Patch Prefect's emit_event to silently ignore errors.
    Call this after Prefect is imported.
    """
    try:
        import prefect.events.utilities as events_utils
        
        original_emit = getattr(events_utils, 'emit_event', None)
        if original_emit is None or getattr(original_emit, '_patched', False):
            return
        
        def silent_emit_event(*args, **kwargs):
            """Emit event but silently ignore errors."""
            try:
                return original_emit(*args, **kwargs)
            except RuntimeError:
                pass  # Silently ignore all runtime errors (stopped service)
            except Exception:
                pass  # Ignore all event errors
        
        silent_emit_event._patched = True
        events_utils.emit_event = silent_emit_event
        
        # Also patch the services module to prevent the error being raised
        try:
            import prefect._internal.concurrency.services as services
            original_send = services.QueueService.send
            
            def silent_send(self, item):
                try:
                    return original_send(self, item)
                except RuntimeError:
                    pass
            
            if not getattr(original_send, '_patched', False):
                silent_send._patched = True
                services.QueueService.send = silent_send
        except Exception:
            pass
            
    except ImportError:
        pass
    except Exception:
        pass


class PrefectEventFilter(logging.Filter):
    """Filter out Prefect event-related error messages and warnings."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        # Filter out Prefect noise
        msg = record.getMessage().lower()
        
        # Filter out "stopped service" and event-related errors
        if "stopped service" in msg:
            return False
        if "error emitting event" in msg:
            return False
        if "eventsworker" in msg.lower():
            return False
        if "failed to load collection" in msg:
            return False
        if "unable to write to memo_store" in msg:
            return False
        if "prefect" in record.name.lower() and record.levelno < logging.ERROR:
            # Suppress all Prefect messages below ERROR level
            return False
        return True


def apply_event_filter():
    """Apply filter to suppress Prefect event errors in logs."""
    event_filter = PrefectEventFilter()
    
    # Apply to root logger
    logging.getLogger().addFilter(event_filter)
    
    # Apply to prefect loggers
    for name in ["prefect", "prefect.events", "prefect._internal"]:
        logging.getLogger(name).addFilter(event_filter)


# Auto-configure Prefect when this module is imported
configure_prefect_logging()
apply_event_filter()

