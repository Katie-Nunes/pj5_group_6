import functools
import streamlit as st

@st.dialog("Error")
def error(func, e):
    st.error(f"Error in {func.__name__}: {e}")
    if st.expander("Show debug info", key=f"debug_{func.__name__}"):
        st.exception(e)

def streamlit_error_handler(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error(func, e)
    return wrapper
