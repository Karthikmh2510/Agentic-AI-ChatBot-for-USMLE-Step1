# streamlit_app.py
import streamlit as st
import requests, shelve, os, time
from PIL import Image
from style.chatbot_style import app_css          # your helper

# ─── Paths & constants ──────────────────────────────────────────────
ARTIFACT_DIR   = "artifacts"
DB_PATH        = os.path.join(ARTIFACT_DIR, "chat_history")
BACKGROUND_URL = "/static/bg.png"                # served from Docker
SIDEBAR_LOGO   = "style/slidebar_background.png"
API_URL        = "http://backend:8080/chat"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

USER_EMOJI, BOT_EMOJI = "👤", "🤖"

# ─── CSS injection ─────────────────────────────────────────────────
st.markdown(app_css(BACKGROUND_URL), unsafe_allow_html=True)

# ─── Title ─────────────────────────────────────────────────────────
st.markdown("<h1 class='app-title'>USMLE‑RAG Chatbot</h1>", unsafe_allow_html=True)

# ─── Chat‑history helpers ──────────────────────────────────────────
def load_history():
    with shelve.open(DB_PATH) as db:
        return db.get("messages", [])
def save_history(msgs):
    with shelve.open(DB_PATH, writeback=True) as db:
        db["messages"] = msgs; db.sync()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image(Image.open(SIDEBAR_LOGO), use_container_width=True)
    st.markdown(
        """
        ### 🧠 About  
        Ask MCQs, definitions, tables – anything high‑yield for **USMLE Step 1**.
        """,
        unsafe_allow_html=True,
    )
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages.clear()
        save_history([])

# ─── Render chat history ───────────────────────────────────────────
for m in st.session_state.messages:
    cls   = "user-msg" if m["role"]=="user" else "bot-msg"
    emoji = USER_EMOJI  if m["role"]=="user" else BOT_EMOJI

    st.markdown(
        f"<div class='{cls}'>"
        f"  <span class='msg-icon'>{emoji}</span>"
        f"  <span class='msg-text'>{m['content']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─── Input → backend call ──────────────────────────────────────────
if prompt := st.chat_input("Ask your Step‑1 question here…"):
    # show user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div class='user-msg'>"
        f"  <span class='msg-icon'>{USER_EMOJI}</span>"
        f"  <span class='msg-text'>{prompt}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )


    # typing placeholder
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class='bot-msg'>
            <span class='msg-icon'>{BOT_EMOJI}</span>
            <b>Typing</b>
            <span class='typing'><span></span><span></span><span></span></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # call backend
    try:
        resp    = requests.post(API_URL, json={"question": prompt}, timeout=90)
        time.sleep(1)       # small delay for “Typing…” feel
        answer  = resp.json().get("answer", "❌ No answer.") if resp.ok else "❌ Error."
    except Exception as e:
        answer = f"❌ Backend error: {e}"

    # replace placeholder with real answer
    placeholder.empty()
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.markdown(
        f"<div class='bot-msg'>"
        f"  <span class='msg-icon'>{BOT_EMOJI}</span>"
        f"  <span class='msg-text'>{answer}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    save_history(st.session_state.messages)
