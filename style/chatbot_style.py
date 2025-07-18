
def app_css():

    return f"""
    <style>
    
    /*   Keep the Streamlit header always visible */
    header[data-testid="stHeader"] {{
        position: fixed;            /* stay at the top */
        top: 0; left: 0; right: 0;
        z-index: 1000;
        background: #e9f4ff;        /* match your pale-blue theme */
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
    }}

    /* Give the page body breathing room beneath the fixed header */
    .block-container {{
        margin-top: 3.5rem;         /* ≈ header height */
        padding-top: 0 !important;
        padding-bottom: 1rem !important;
        max-width: 1400px !important;  /*   widen chat column */
        width: 90%;
    }}

    /* ───── MESSAGE CARDS (kept from previous answer, plus tweaks) ─────── */
    .user-msg,
    .bot-msg {{
        width: 100%;
        padding: .9rem 1.2rem;
        margin:  .45rem 0;
        display: flex;
        gap: .75rem;
        background: #ffffffee;
        box-shadow: 0 1px 4px rgba(0,0,0,.08);
        border-radius: .45rem;
        border-inline-start: 6px solid var(--accent);
    }}
    .user-msg {{ --accent: #0068d6; 
        background: rgba(0, 104, 214, .20);  /* was .15 → a touch darker       */
        border-inline-start-color: var(--accent);
    }}
    .bot-msg  {{ --accent: #02a89e; }}

    .msg-icon {{ font-size: 1.45rem; line-height: 1; flex-shrink: 0; }}
    .msg-text {{
        flex: 1;
        white-space: normal;
        overflow-wrap: anywhere;
        overflow-x: auto;           /* table fallback */
    }}

    /*   Wide tables should fill the card, not shrink it */
    .msg-text table {{
        width: 100%;               /* use all the card space */
        border-collapse: collapse;
    }}

    /* ───── SIDEBAR HANDLE (unchanged) ─────────────────────────────────── */
    header[data-testid="stHeader"] button[kind="header"]{{z-index:1001!important;}}
    header[data-testid="stHeader"] button[kind="header"] svg{{color:#000!important;}}
    [data-testid="stSidebar"][aria-expanded="false"]{{
        width:6px!important;cursor:pointer;background:rgba(255,255,255,.35);
        transition:background .25s;
    }}
    [data-testid="stSidebar"][aria-expanded="false"]:hover{{background:rgba(255,255,255,.6);}}

    /* ───── TYPING ANIMATION ─────────────────────────────────── */
    .typing-ellipsis {{
        display: inline-block;
        width: 1.5em;
        text-align: left;
    }}
    .typing-ellipsis span {{
        display: inline-block;
        width: .3em;
        height: .3em;
        margin-right: .15em;
        background: #02a89e;
        border-radius: 50%;
        opacity: 0.3;
        animation: ellipsis-bounce 1.2s infinite;
    }}
    .typing-ellipsis span:nth-child(2) {{
        animation-delay: 0.2s;
    }}
    .typing-ellipsis span:nth-child(3) {{
        animation-delay: 0.4s;
    }}
    @keyframes ellipsis-bounce {{
        0%, 80%, 100% {{ opacity: 0.3; transform: translateY(0);}}
        40% {{ opacity: 1; transform: translateY(-0.25em);}}
    }}

    </style>
    """
