"""Review desk typography, colors and responsive spacing."""

import streamlit as st


def apply_theme():
    st.markdown(
        """
    <style>
    .stApp {background:#f6f7f3;color:#172e2b}
    .block-container {max-width:1320px;padding-top:3rem}
    [data-testid="stHeader"] {background:transparent}
    h1,h2,h3 {letter-spacing:-.045em;line-height:1.12}
    h1 {font-size:clamp(2.4rem,5vw,4.7rem)!important}
    [data-testid="stMetric"] {padding:.85rem 1rem;background:#e7eee7;border-top:3px solid #326354}
    .atlas-brand {display:flex;gap:.7rem;align-items:center;font-weight:750;color:#204439;
      border-bottom:1px solid #9db2a7;padding-bottom:.8rem;margin-bottom:1.4rem}
    .atlas-brand b {display:inline-grid;place-items:center;background:#244b40;color:#fff;
      width:2.15rem;height:2.15rem;border-radius:.5rem;font-size:1.2rem}
    .stButton button,.stFormSubmitButton button {border-radius:.45rem}
    :focus-visible {outline:3px solid #825000!important;outline-offset:3px}
    @media(prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
    </style>
    <div class="atlas-brand"><b aria-hidden="true">A</b> Access Obstacle Atlas</div>
    """,
        unsafe_allow_html=True,
    )
