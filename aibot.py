# improved_aibot.py
import os
import json
import time
import random
import streamlit as st
from datetime import datetime
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

# ---------------------------
# Configuration / Constants
# ---------------------------
load_dotenv()  # load .env if present
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
CHAT_HISTORY_FILE = "chat_history.json"
DEFAULT_MODEL = "gemini-2.0-flash"

# System persona & behavior
SYSTEM_PROMPT = (
    "You are AI-Assistant, a friendly, helpful chatbot. "
    "Be concise, use emojis moderately, and prefer plain language similar to chat/DM style. "
    "When asked for code include code blocks. Always be helpful and factual."
)

# Limits & behavior tuning
MAX_MESSAGES_BEFORE_SUMMARY = 18   # when exceeded, older messages will be summarized
SUMMARY_PROMPT = "Summarize the following conversation briefly (2-3 sentences) preserving important facts and any user's preferences."

# Retry/backoff config
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
BACKOFF_MULTIPLIER = 2.0

# ---------------------------
# Helper utilities
# ---------------------------
def timestamp() -> str:
    return datetime.now().strftime("%I:%M %p")

def load_history(filepath: str = CHAT_HISTORY_FILE) -> List[Dict[str, Any]]:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history: List[Dict[str, Any]], filepath: str = CHAT_HISTORY_FILE) -> None:
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # don't fail the UI if disk save fails

def exponential_backoff_call(func, *args, **kwargs):
    """
    Call func(*args, **kwargs) with exponential backoff for rate-limit and transient errors.
    Returns (result, error) where error is None if success.
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = func(*args, **kwargs)
            return res, None
        except google_exceptions.ResourceExhausted as e:
            # Rate limit — advise user and retry after backoff
            if attempt == MAX_RETRIES:
                return None, e
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER
            continue
        except google_exceptions.ServiceUnavailable as e:
            if attempt == MAX_RETRIES:
                return None, e
            time.sleep(backoff)
            backoff *= BACKOFF_MULTIPLIER
            continue
        except Exception as e:
            # For other errors, don't aggressively retry — return the error
            return None, e
    return None, Exception("Max retries exceeded")

# ---------------------------
# Set up Gemini client
# ---------------------------
if not GEMINI_API_KEY:
    st.error("Gemini API key not found in environment. Set GEMINI_API_KEY and restart the app.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model_obj = genai.GenerativeModel(DEFAULT_MODEL)
except Exception as e:
    st.error(f"Failed to configure Gemini client: {e}")
    st.stop()

# ---------------------------
# Conversation utilities
# ---------------------------
def start_chat_session():
    try:
        return model_obj.start_chat(history=[])
    except Exception as e:
        raise e

def safe_send_message(chat_session, user_text: str):
    """
    Send a single user message using the provided chat_session object.
    Uses exponential backoff for rate limits.
    """
    def _send():
        return chat_session.send_message(user_text)

    response, err = exponential_backoff_call(_send)
    return response, err

def summarize_history(chat_session, messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Summarize older messages using the model.
    Returns summary string or None on failure.
    """
    try:
        # Build a single text block to summarize
        convo_text = "\n".join([f"{m['role']}: {m['text']}" for m in messages])
        summ_prompt = f"{SUMMARY_PROMPT}\n\n{convo_text}"
        resp, err = exponential_backoff_call(lambda: chat_session.send_message(summ_prompt))
        if err or not resp:
            return None
        # try extract text safely
        if hasattr(resp, "text") and resp.text:
            return resp.text.strip()
        return str(resp).strip()
    except Exception:
        return None

# ---------------------------
# Streamlit UI & state init
# ---------------------------
st.set_page_config(page_title="AI-Assistant (Improved)", page_icon="🤖", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = load_history()
if "chat_session" not in st.session_state:
    # create fresh session
    try:
        st.session_state.chat_session = start_chat_session()
    except Exception as e:
        st.error(f"Could not create chat session: {e}")
        st.stop()
if "system_note" not in st.session_state:
    st.session_state.system_note = SYSTEM_PROMPT
if "dev_mode" not in st.session_state:
    # disable developer debug details by default
    st.session_state.dev_mode = os.getenv("DEV_MODE", "0") == "1"

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.title("⚙ AI-Assistant")
    st.markdown("**Settings & Tools**")

    model_choice = st.selectbox("Model", [DEFAULT_MODEL], index=0, help="Change model name if you have other models enabled.")
    st.write("---")
    st.markdown("**Appearance**")
    theme_choice = st.selectbox("Theme", ["Light", "Dark", "Sunset"], index=0)
    st.write("---")
    st.markdown("**Conversation**")
    if st.button("Clear local history"):
        st.session_state.history = []
        save_history(st.session_state.history)
        st.experimental_rerun()

    if st.button("Export chat (JSON)"):
        save_history(st.session_state.history)
        st.download_button("Download chat JSON", data=json.dumps(st.session_state.history, ensure_ascii=False, indent=2),
                           file_name="chat_history.json")

    st.write("---")
    st.markdown("**Helpers**")
    if st.button("Summarize long history now"):
        # run summarization on older messages
        if len(st.session_state.history) > 6:
            older = st.session_state.history[:-6]
            s = summarize_history(st.session_state.chat_session, older)
            if s:
                st.success("Summary created and saved as system note.")
                st.session_state.system_note = s
                # drop older messages and keep last N
                st.session_state.history = st.session_state.history[-6:]
                save_history(st.session_state.history)
            else:
                st.warning("Could not summarize right now (try again later).")
        else:
            st.info("Not enough history to summarize.")
    st.write("---")
    st.checkbox("Developer mode (detailed errors)", value=st.session_state.dev_mode, key="dev_mode")

# ---------------------------
# Styling (simple)
# ---------------------------
st.markdown("""
<style>
/* container */
.chat-wrapper { max-width: 1000px; margin: 0 auto; }
.row { display:flex; gap:1rem; }
.col { flex:1; }
/* bubbles */
.user-bubble, .bot-bubble {
  padding: 0.8rem 1rem;
  border-radius: 14px;
  max-width: 80%;
  word-wrap: break-word;
}
.user-bubble { background: #e1f5fe; align-self: flex-end; margin-left:auto; }
.bot-bubble  { background: #f1f1f1; align-self: flex-start; margin-right:auto; }
.meta { font-size: 0.75rem; color: #666; margin-top:0.25rem; }
.suggest-btn { margin: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Main UI layout
# ---------------------------
st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
st.header("💬 AI-Assistant — improved")

# suggested prompts row
suggested = ["Explain machine learning like I'm 5", "Give me a 5-point study plan for superconductivity", "Summarize my previous messages", "Generate a short poem about AI"]
st.markdown("**Suggested prompts:**")
cols = st.columns(len(suggested))
for i, s in enumerate(suggested):
    if cols[i].button(s, key=f"suggest_{i}", help="Click to send this prompt"):
        # inject into input
        st.session_state._prefill = s

# file upload for context
uploaded = st.file_uploader("Upload a .txt file to include as context (optional)", type=["txt"])
if uploaded:
    try:
        file_text = uploaded.getvalue().decode("utf-8")
        st.info("File loaded — it will be included as context in the next message.")
        st.session_state._file_context = file_text
    except Exception:
        st.warning("Could not read uploaded file.")

# show conversation area (scrollable)
st.markdown("### Conversation")
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta'>[{msg['ts']}] You</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-bubble'>{msg['text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='meta'>[{msg['ts']}] AI-Assistant</div>", unsafe_allow_html=True)

# input area
st.markdown("---")
prefill = st.session_state.pop("_prefill", "")
user_input = st.text_area("Your message", value=prefill, placeholder="Type a message and press Send", key="user_input_area", height=120)
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    send = st.button("Send")
with col2:
    quick1 = st.button("Explain simply")
with col3:
    quick2 = st.button("Outline")

# quick replies map
if quick1:
    user_input = "Explain this simply: " + (user_input or "the topic")
if quick2:
    user_input = "Give me an outline:" + (user_input or "")

if send and user_input:
    # assemble full user text with optional file context and system note
    pieces = []
    if st.session_state.system_note:
        pieces.append(f"[SYSTEM NOTE] {st.session_state.system_note}")
    if hasattr(st.session_state, "_file_context"):
        pieces.append("[FILE CONTEXT]\n" + st.session_state._file_context)
        # clear file context after using once
        del st.session_state._file_context
    pieces.append(f"[USER] {user_input}")
    full_text = "\n\n".join(pieces)

    # append user message to local history
    st.session_state.history.append({"role": "user", "text": user_input, "ts": timestamp()})
    save_history(st.session_state.history)

    # send message (with retry/backoff)
    with st.spinner("AI-Assistant is thinking..."):
        try:
            # create fresh session for each call to avoid stale state and for stability
            chat_session = start_chat_session()
            response, err = exponential_backoff_call(lambda: chat_session.send_message(full_text))
            if err:
                # handle rate limit specifically
                if isinstance(err, google_exceptions.ResourceExhausted):
                    reply_text = ("⚠️ Rate limit reached. Please wait a minute or reduce request frequency. "
                                  "If this persists, request a quota increase in your Google Cloud project.")
                else:
                    reply_text = f"⚠️ Error: {str(err)}"
                    if st.session_state.dev_mode:
                        reply_text += f"\n\nDevTrace: {repr(err)}"
            else:
                # extract response text
                if hasattr(response, "text") and response.text:
                    reply_text = response.text.strip()
                else:
                    reply_text = str(response).strip()
        except Exception as e:
            reply_text = f"⚠️ Unexpected error: {str(e)}"
            if st.session_state.dev_mode:
                import traceback
                reply_text += "\n\n" + traceback.format_exc(limit=5)

    # append bot reply
    sticker = random.choice(["✨", "💬", "🌟", "😊", "👍", "🦾", "🎉", "🤖"])
    reply_with_sticker = f"{reply_text} {sticker}"
    st.session_state.history.append({"role": "assistant", "text": reply_with_sticker, "ts": timestamp()})
    save_history(st.session_state.history)

    # if history grows too long, summarize older part
    if len(st.session_state.history) > MAX_MESSAGES_BEFORE_SUMMARY:
        older_part = st.session_state.history[:-8]
        recent_part = st.session_state.history[-8:]
        # summarize older messages
        try:
            chat_session = start_chat_session()
            summary = summarize_history(chat_session, older_part)
            if summary:
                st.session_state.system_note = summary
                # keep only recent part in history (older part summarized into system note)
                st.session_state.history = recent_part
                save_history(st.session_state.history)
                st.success("🔖 Long history summarized to system note to preserve context and reduce token usage.")
        except Exception:
            # if summarization fails, keep history as-is
            pass

    # re-render page to show updated messages
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)
