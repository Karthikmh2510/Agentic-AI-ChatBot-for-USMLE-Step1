# style/chatbot_style.py
from pathlib import Path
import base64


def build_bg_rule(background_url: str | None) -> str:
    if background_url:
        return f'background: url("{background_url}") center/cover fixed;'

    encoded = base64.b64encode(Path("style/main_background.png").read_bytes()).decode()
    return f'background: url("data:image/png;base64,{encoded}") center/cover fixed;'


def app_css(background_url: str | None = None) -> str:
    bg_rule = build_bg_rule(background_url)

    return f"""
    <style>
    /* ─── PAGE LAYOUT ─────────────────────────────────────────────────── */
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
        {bg_rule}
        color: #000 !important;
    }}

    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
    }}

    /* ─── TITLE ───────────────────────────────────────────────────────── */
    .app-title {{
        width: 100%;
        text-align: center;
        margin: 1rem 0 1.25rem;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        line-height: 1.2;
        text-shadow: 0 0 6px rgba(255,255,255,.65);
    }}

    /* ─── CHAT BUBBLES ────────────────────────────────────────────────── */
    .user-msg,
    .bot-msg {{
        padding: .70rem 1.05rem;
        margin:  .35rem 0;
        max-width: 90%;
        font-size: .95rem;
        line-height: 1.45;
        border-radius: .65rem;
        backdrop-filter: blur(3px);
        -webkit-backdrop-filter: blur(3px);
        color: #000 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,.10);
        display:flex;
        align-items:flex-start;
        gap:.55rem;
    }}

    .user-msg {{
        margin-left: auto;
        background: rgba(0,123,255,.15);
        border: 1px solid rgba(0,123,255,.35);
    }}

    .bot-msg {{
        margin-right: auto;
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(0,0,0,.08);
    }}

    .msg-icon {{
        font-size: 1.45rem;
        line-height: 1;
        flex-shrink: 0;
    }}

    .msg-text {{
        flex: 1;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}

    /* ─── TYPING INDICATOR ───────────────────────────────────────────── */
    .typing span {{
        display: inline-block;
        width: 6px;
        height: 6px;
        margin: 0 2px;
        border-radius: 50%;
        background: #666;
        opacity: 0;
        animation: blink 1.2s infinite;
    }}
    .typing span:nth-child(2) {{ animation-delay: .2s; }}
    .typing span:nth-child(3) {{ animation-delay: .4s; }}

    @keyframes blink {{
        0%, 80% {{ opacity: 0; }}
        40%     {{ opacity: 1; }}
    }}

    /* ─── SIDEBAR TOGGLE ─────────────────────────────────────────────── */
    header[data-testid="stHeader"] button[kind="header"] {{
        z-index: 999 !important;
    }}
    header[data-testid="stHeader"] button[kind="header"] svg {{
        color: #000 !important;
    }}

    [data-testid="stSidebar"][aria-expanded="false"] {{
        width: 6px !important;
        cursor: pointer;
        background: rgba(255,255,255,.35);
        transition: background .25s;
    }}
    [data-testid="stSidebar"][aria-expanded="false"]:hover {{
        background: rgba(255,255,255,.6);
    }}
    </style>
    """
