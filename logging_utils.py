"""
Logging Utilities

Provides unified logging functions that output to both Python logger
and Streamlit UI for user feedback.
"""
import logging
import streamlit as st
from typing import Optional

logger = logging.getLogger("bus-planning")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def report_error(
    msg: str,
    exception: Optional[Exception] = None,
    user: bool = True,
    fatal: bool = False
) -> None:
    """
    Log error message and optionally display in Streamlit UI.
    
    Args:
        msg: Error message to log
        exception: Optional exception object for stack trace
        user: Whether to display message in Streamlit UI
        fatal: Whether to raise exception and stop execution
    """
    if exception:
        logger.exception(msg)
    else:
        logger.error(msg)

    if user:
        st.error(f"( ｡ •̀ ᴖ •́ ｡) {msg}")
    if fatal:
        raise exception if exception else RuntimeError(msg)


def report_warning(msg: str, user: bool = True) -> None:
    """
    Log warning message and optionally display in Streamlit UI.
    
    Args:
        msg: Warning message to log
        user: Whether to display message in Streamlit UI
    """
    logger.warning(msg)
    if user:
        st.warning(f" (╥ᆺ╥；)  {msg}")


def report_info(msg: str, user: bool = False) -> None:
    """
    Log info message and optionally display in Streamlit UI.
    
    Args:
        msg: Info message to log
        user: Whether to display message in Streamlit UI
    """
    logger.info(msg)
    if user:
        st.info(msg)