# utils/logging_utils.py
import logging
import streamlit as st

logger = logging.getLogger("bus-planning")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def report_error(msg: str, exception: Exception = None, user: bool = True, fatal: bool = False):
    """Log and optionally show error in Streamlit."""
    if exception:
        logger.exception(msg)
    else:
        logger.error(msg)

    if user:
        st.error(f"( ｡ •̀ ᴖ •́ ｡) {msg}")
    if fatal:
        raise exception if exception else RuntimeError(msg)

def report_warning(msg: str, user: bool = True):
    logger.warning(msg)
    if user:
        st.warning(f" (╥ᆺ╥；)  {msg}")

def report_info(msg: str, user: bool = False):
    logger.info(msg)
    if user:
        st.info(f"ℹ️ {msg}")