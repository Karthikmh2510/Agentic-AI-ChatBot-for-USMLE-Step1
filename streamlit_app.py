import streamlit as st
import shelve, os
from style.chatbot_style import app_css
from src.Agentic_RAG import ask

# ─── Paths & constants ──────────────────────────────────────────────
ARTIFACT_DIR     = "artifacts"
DB_PATH          = os.path.join(ARTIFACT_DIR, "chat_history")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

USER_EMOJI, BOT_EMOJI = "👤", " 🤖 "

# ─── CSS injection ─────────────────────────────────────────────────
st.markdown(app_css(), unsafe_allow_html=True)

# ─── Title ─────────────────────────────────────────────────────────
st.markdown("<h1 class='app-title'>USMLE‑RAG Chatbot</h1>", unsafe_allow_html=True)

# ─── Chat-history helpers ──────────────────────────────────────────
def load_history():
    with shelve.open(DB_PATH) as db:
        return db.get("messages", [])
def save_history(msgs):
    with shelve.open(DB_PATH, writeback=True) as db:
        db["messages"] = msgs
        db.sync()

if "messages" not in st.session_state:
    st.session_state.messages = load_history()

# ─── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.image("style/slidebar_image.png", use_container_width="auto")

    st.markdown(
        """
        ### 🧠 About

        High-yield Q&A for **USMLE Step 1**.

        **How to use:**  
        • Ask any MCQ, definition, or topic.<br>
        • Get concise, cited explanations.<br>
        • Use 🗑️ to clear chat.

        **Topics covered:**  
        • All systems: cardio, neuro, renal, etc.<br>
        • Biochem, genetics, micro, pharm, and more.

        _For study support only—always verify with trusted sources._
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

# ─── Input → ask() call ───────────────────────────────────────────
if prompt := st.chat_input("Ask your Step‑1 question here…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div class='user-msg'>"
        f"  <span class='msg-icon'>{USER_EMOJI}</span>"
        f"  <span class='msg-text'>{prompt}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class='bot-msg'>
            <span class='msg-icon'>{BOT_EMOJI}</span>
            <b>Typing</b>
            <span class='typing-ellipsis'>
                <span></span><span></span><span></span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        answer = ask(prompt)
    except Exception as e:
        answer = f"❌ Error: {e}"

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
