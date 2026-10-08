"""CityAI: a single-file Streamlit front end for a LangChain city assistant.

Run with:  streamlit run app.py
Requires a .env file (or environment variables) with:
    MISTRAL_API_KEY, OPENWEATHER_API_KEY, TAVILY_API_KEY
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
import uuid
from html import escape

import requests
import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from tavily import TavilyClient

st.set_page_config(
    page_title="CityAI",
    page_icon="◉",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================================================================== #
# Backend: tools + agent
# =========================================================================== #
MODEL_NAME = "labs-leanstral-1-5"
SYSTEM_PROMPT = "You are a helpful city assistant"
DENIAL_MESSAGE = "Tool call denied by user."

REQUIRED_KEYS = {
    "MISTRAL_API_KEY": "Mistral AI",
    "OPENWEATHER_API_KEY": "OpenWeather",
    "TAVILY_API_KEY": "Tavily",
}


@tool
def get_weather(city: str) -> str:
    """Get Current Weather of a City"""
    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": os.getenv("OPENWEATHER_API_KEY"),
                "units": "metric",
            },
            timeout=10,
        )
        data = response.json()
    except requests.RequestException:
        return "Error: the weather service could not be reached."
    except ValueError:
        return "Error: the weather service returned an unreadable response."

    if str(data.get("cod")) != "200":
        return f"Error: {data.get('message', 'Could not fetch weather')}"

    temp = data["main"]["temp"]
    desc = data["weather"][0]["description"]
    return f"Weather in {city}: {desc}, {temp}°C"


@tool
def get_news(city: str) -> str:
    """Get Latest news about the city"""
    try:
        # Created lazily so the app can start and show a setup screen without a key.
        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search(
            query=f"latest news in {city}",
            topic="news",
            max_results=3,
        )
    except Exception as exc:
        return f"Error: the news service failed ({type(exc).__name__})."

    results = response.get("results", [])
    if not results:
        return f"No recent news found for {city}."

    news = [
        f"Title: {r.get('title', '')}\n"
        f"URL: {r.get('url', '')}\n"
        f"Summary: {r.get('content', '')}"
        for r in results
    ]
    return "\n\n".join(news)


TOOLS = [get_weather, get_news]


@st.cache_resource(show_spinner=False)
def get_agent():
    """Build the agent once. Every tool call pauses for user approval.

    Streamlit can't block on input(), so approval uses LangChain's
    HumanInTheLoopMiddleware: the agent pauses, the UI asks, then resumes.
    """
    return create_agent(
        ChatMistralAI(model=MODEL_NAME),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        middleware=[HumanInTheLoopMiddleware(interrupt_on={t.name: True for t in TOOLS})],
        checkpointer=InMemorySaver(),
    )


def missing_keys() -> list[str]:
    return [key for key in REQUIRED_KEYS if not os.getenv(key)]


def to_text(content) -> str:
    """Normalise model output (str or list of content blocks) to plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts).strip()
    return ""


def extract_actions(interrupts) -> list[dict]:
    """Pull the pending tool calls out of an agent interrupt."""
    actions: list[dict] = []
    for item in interrupts or []:
        value = getattr(item, "value", item)
        if isinstance(value, dict):
            actions.extend(value.get("action_requests", []))
    return actions


def describe_error(exc: Exception) -> str:
    text = str(exc).lower()
    if "401" in text or "unauthorized" in text or "api key" in text:
        return "The AI service rejected the API key. Check MISTRAL_API_KEY in your .env file."
    if "429" in text or "rate limit" in text:
        return "The AI service is rate limiting requests. Wait a moment, then try again."
    if "timeout" in text or "connection" in text:
        return "The AI service could not be reached. Check your connection and try again."
    return "Something went wrong while generating a response. Try again, or rephrase your question."


# =========================================================================== #
# Copy
# =========================================================================== #
SUGGESTIONS = [
    "What's the weather in Dhaka?",
    "Show me the latest news in London",
    "What's happening in Tokyo today?",
]

# How each tool is described to the user (internals are never shown).
TOOL_COPY = {
    "get_weather": {
        "title": "Weather information",
        "ask": "check the current weather",
        "running": "Checking current weather",
        "done": "Checked weather",
    },
    "get_news": {
        "title": "Latest news",
        "ask": "search the latest news",
        "running": "Searching latest city news",
        "done": "Searched news",
    },
}
FALLBACK_COPY = {
    "title": "Assistant action",
    "ask": "run an action",
    "running": "Working",
    "done": "Completed action",
}


def tool_copy(name: str) -> dict:
    return TOOL_COPY.get(name, FALLBACK_COPY)


def action_city(action: dict) -> str:
    return str((action.get("args") or {}).get("city") or "").strip()


# =========================================================================== #
# Styles
# =========================================================================== #
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;450;500;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
  --bg: #0A0B0E; --surface: #101216; --surface-2: #15181D;
  --border: rgba(255,255,255,.07); --border-strong: rgba(255,255,255,.12);
  --text: #E8EAEE; --muted: #9097A6; --faint: #656B7C;
  --accent: #8B9CFF; --accent-soft: rgba(139,156,255,.10); --accent-line: rgba(139,156,255,.24);
  --ok: #5BD6A0; --warn: #E8B45A; --danger: #F07C82;
  --sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --serif: 'Instrument Serif', Georgia, 'Times New Roman', serif;
}

/* Base and Streamlit chrome */
.stApp {
  background: radial-gradient(900px 420px at 50% -8%, rgba(139,156,255,.07), transparent 62%), var(--bg);
  color: var(--text); font-family: var(--sans);
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppDeployButton"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; height: 0; min-height: 0; }
[data-stale="true"] { opacity: 1 !important; }
.block-container, [data-testid="stMainBlockContainer"] { max-width: 820px; padding: 1.75rem 1.5rem 9rem; }
[data-testid="stVerticalBlock"] { gap: 1rem; }
::selection { background: var(--accent-line); }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.08); border-radius: 10px; border: 3px solid var(--bg); }

/* Header */
.cai-header { display: flex; align-items: center; justify-content: space-between;
  padding-bottom: 1.25rem; margin-bottom: .5rem; border-bottom: 1px solid var(--border); }
.cai-brand { display: flex; align-items: center; gap: .8rem; }
.cai-mark { width: 28px; height: 28px; border-radius: 8px; position: relative;
  background: linear-gradient(145deg, #B4C0FF, #6F80EE);
  box-shadow: 0 0 22px rgba(139,156,255,.30), inset 0 0 0 1px rgba(255,255,255,.25); }
.cai-mark::after { content: ""; position: absolute; inset: 8px; border-radius: 50%; background: var(--bg); }
.cai-name { font-size: 1.05rem; font-weight: 600; letter-spacing: -.01em; line-height: 1.2; }
.cai-sub { font-size: .8rem; color: var(--muted); line-height: 1.3; margin-top: 1px; }
.cai-status { display: inline-flex; align-items: center; gap: .5rem; font-size: .78rem; color: var(--muted);
  padding: .35rem .7rem; border: 1px solid var(--border); border-radius: 999px; background: rgba(255,255,255,.02); }
.cai-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--ok); box-shadow: 0 0 8px var(--ok); display: inline-block; }
.cai-status.is-warn .cai-dot { background: var(--warn); box-shadow: 0 0 8px var(--warn); }

/* Empty state */
.cai-hero { text-align: center; padding: 14vh 0 2.25rem; animation: cai-rise .7s cubic-bezier(.2,.7,.2,1) both; }
.cai-hero h1 { font-family: var(--serif); font-weight: 400; font-size: clamp(2.4rem, 6vw, 3.6rem);
  line-height: 1.08; letter-spacing: -.015em; margin: 0 0 1rem; padding: 0; color: var(--text); }
.cai-hero p { color: var(--muted); font-size: 1.05rem; margin: 0; }
.st-key-suggestions { max-width: 760px; margin: 0 auto; }
.st-key-suggestions button { min-height: 3.6rem; text-align: left; justify-content: flex-start;
  color: var(--muted); background: var(--surface); line-height: 1.4; }
.st-key-suggestions button p { text-align: left; font-size: .88rem; }
.st-key-suggestions button:hover { color: var(--text); background: var(--surface-2); }

/* Buttons */
.stButton > button { font-family: var(--sans); font-weight: 500; font-size: .9rem; border-radius: 9px;
  padding: .55rem 1rem; background: var(--surface-2); color: var(--text); border: 1px solid var(--border-strong);
  transition: border-color .15s, background .15s, box-shadow .15s, transform .1s; }
.stButton > button:hover { border-color: var(--accent-line); color: var(--text); background: var(--surface-2); }
.stButton > button:active { transform: translateY(1px); }
.stButton > button:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background: var(--accent); color: #0B0D18; border-color: transparent;
  box-shadow: 0 0 0 1px rgba(255,255,255,.15) inset, 0 6px 20px rgba(139,156,255,.22); }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background: #A0AEFF; color: #0B0D18; }

/* Messages */
[data-testid="stChatMessage"] { background: transparent; border: 1px solid var(--border);
  border-radius: 12px; padding: 1.1rem 1.35rem; gap: 0; align-items: flex-start; }
[data-testid="stChatMessage"] [data-testid^="stChatMessageAvatar"],
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] { display: none; }
[data-testid="stChatMessageContent"] { flex: 1; min-width: 0; }
[data-testid="stChatMessage"]:has(.cai-user) { margin-left: auto; width: fit-content; max-width: 88%;
  background: var(--accent-soft); border-color: var(--accent-line); }
[data-testid="stChatMessage"]:has(.cai-assistant) {
  background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.012));
  box-shadow: 0 10px 40px rgba(0,0,0,.25); }
.cai-user, .cai-assistant { display: none; }
.cai-role { display: flex; align-items: center; gap: .5rem; font-size: .78rem; font-weight: 500;
  color: var(--faint); margin-bottom: .45rem; }
.cai-role-mark { width: 8px; height: 8px; border-radius: 3px; background: var(--accent); box-shadow: 0 0 10px rgba(139,156,255,.5); }
.cai-user-text { font-size: .97rem; line-height: 1.6; white-space: pre-wrap; overflow-wrap: anywhere; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li { font-size: .97rem; line-height: 1.75; color: #D9DCE3; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin-bottom: .8rem; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
  font-size: 1.05rem; font-weight: 600; letter-spacing: -.01em; margin: 1.1rem 0 .5rem; padding: 0; }
[data-testid="stChatMessage"] a { color: var(--accent); text-decoration: none; border-bottom: 1px solid var(--accent-line); }
[data-testid="stChatMessage"] a:hover { border-bottom-color: var(--accent); }
[data-testid="stChatMessage"] strong { color: var(--text); font-weight: 600; }
[data-testid="stChatMessage"] code { background: rgba(255,255,255,.06); border-radius: 5px; padding: .1rem .35rem; font-size: .88em; }

/* Tool activity */
.cai-chips { display: flex; flex-wrap: wrap; gap: .4rem; margin: .1rem 0 .85rem; }
.cai-chip { display: inline-flex; align-items: center; gap: .45rem; font-size: .78rem; color: var(--muted);
  padding: .25rem .6rem; border: 1px solid var(--border); border-radius: 7px; background: rgba(255,255,255,.02); }
.cai-chip-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
.cai-chip-detail { color: var(--text); padding-left: .45rem; border-left: 1px solid var(--border-strong); }
.cai-chip.is-denied .cai-chip-dot { background: var(--danger); }
.cai-working { display: flex; align-items: center; gap: .65rem; font-size: .92rem; padding: .15rem 0; }
.cai-working-dot { width: 9px; height: 9px; border-radius: 50%; border: 2px solid var(--accent); animation: cai-pulse 1.5s ease-in-out infinite; }
.cai-shimmer { background: linear-gradient(90deg, var(--faint) 20%, var(--text) 50%, var(--faint) 80%);
  background-size: 220% 100%; -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; animation: cai-shimmer 2s linear infinite; }

/* Approval card */
.st-key-approval { background: var(--surface); border: 1px solid var(--border-strong); border-radius: 14px;
  padding: 1.4rem 1.5rem 1.25rem; gap: .35rem;
  box-shadow: 0 16px 50px rgba(0,0,0,.5), 0 0 70px rgba(139,156,255,.06);
  animation: cai-rise .35s cubic-bezier(.2,.7,.2,1) both; }
.cai-approval-eyebrow { display: flex; align-items: center; gap: .5rem; font-size: .8rem; color: var(--warn); font-weight: 500; }
.cai-approval-eyebrow .cai-dot { background: var(--warn); box-shadow: 0 0 8px var(--warn); }
.cai-approval-count { margin-left: auto; color: var(--faint); font-weight: 400; }
.cai-approval-title { font-size: 1.15rem; font-weight: 600; letter-spacing: -.01em; margin-top: .75rem; }
.cai-approval-body { color: #C5CAD6; font-size: .95rem; line-height: 1.6; margin-top: .3rem; }
.cai-approval-body strong { color: var(--text); font-weight: 600; }
.cai-approval-note { color: var(--faint); font-size: .8rem; margin: .5rem 0 .6rem; }

/* Notices and errors */
.cai-error { font-size: .93rem; line-height: 1.65; color: #F1C7C9; padding: .75rem 1rem; border-radius: 9px;
  background: rgba(240,124,130,.07); border: 1px solid rgba(240,124,130,.25); }
.cai-notice { margin-top: 8vh; padding: 1.6rem 1.75rem; border-radius: 14px; background: var(--surface);
  border: 1px solid var(--border-strong); box-shadow: 0 16px 50px rgba(0,0,0,.4); }
.cai-notice h3 { font-size: 1.15rem; font-weight: 600; margin: 0 0 .5rem; padding: 0; }
.cai-notice p { color: var(--muted); font-size: .93rem; line-height: 1.65; margin: 0 0 1rem; }
.cai-notice ul { list-style: none; margin: 0; padding: 0; display: grid; gap: .5rem; }
.cai-notice li { display: flex; justify-content: space-between; align-items: center; gap: 1rem;
  padding: .6rem .8rem; border: 1px solid var(--border); border-radius: 8px; background: rgba(255,255,255,.02); }
.cai-notice li span { color: var(--faint); font-size: .85rem; }
.cai-notice code { background: rgba(255,255,255,.06); border-radius: 5px; padding: .1rem .4rem; color: var(--text); font-size: .85em; }

/* Chat input */
[data-testid="stBottom"] { background: linear-gradient(to top, var(--bg) 55%, rgba(10,11,14,0)); }
[data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] { background: transparent; }
[data-testid="stBottomBlockContainer"] { max-width: 820px; padding: 0 1.5rem 1.5rem; }
[data-testid="stChatInput"] { background: var(--surface); border: 1px solid var(--border-strong); border-radius: 14px;
  box-shadow: 0 10px 34px rgba(0,0,0,.45); transition: border-color .2s, box-shadow .2s; }
[data-testid="stChatInput"]:focus-within { border-color: var(--accent-line);
  box-shadow: 0 10px 34px rgba(0,0,0,.45), 0 0 0 4px var(--accent-soft); }
[data-testid="stChatInput"] > div, [data-testid="stChatInput"] div { background: transparent; border: none; }
[data-testid="stChatInput"] textarea { font-family: var(--sans); font-size: .97rem; color: var(--text);
  background: transparent; padding: .9rem 1rem; }
[data-testid="stChatInput"] textarea::placeholder { color: var(--faint); }
[data-testid="stChatInput"] button { color: var(--accent); }
[data-testid="stChatInput"] button:disabled { color: var(--faint); }

/* Motion */
@keyframes cai-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes cai-pulse { 0%, 100% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(139,156,255,.45); }
                       50% { transform: scale(.8); opacity: .55; box-shadow: 0 0 0 6px rgba(139,156,255,0); } }
@keyframes cai-shimmer { from { background-position: 120% 0; } to { background-position: -120% 0; } }
@media (prefers-reduced-motion: reduce) {
  .cai-hero, .st-key-approval, .cai-working-dot, .cai-shimmer { animation: none; }
}

/* Responsive */
@media (max-width: 768px) {
  .block-container, [data-testid="stMainBlockContainer"] { padding: 1.25rem 1rem 8rem; }
  [data-testid="stBottomBlockContainer"] { padding: 0 1rem 1rem; }
  [data-testid="stChatMessage"] { padding: .95rem 1.05rem; }
  [data-testid="stChatMessage"]:has(.cai-user) { max-width: 94%; }
  .cai-hero { padding-top: 8vh; }
}
@media (max-width: 480px) {
  .cai-sub { display: none; }
  .cai-notice li { flex-direction: column; align-items: flex-start; gap: .2rem; }
}
"""


# =========================================================================== #
# UI components
# =========================================================================== #
def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_header(online: bool) -> None:
    status_class = "" if online else " is-warn"
    status_text = "AI Online" if online else "Setup required"
    html(
        f"""
        <header class="cai-header">
          <div class="cai-brand">
            <div class="cai-mark"></div>
            <div>
              <div class="cai-name">CityAI</div>
              <div class="cai-sub">Your intelligent city assistant</div>
            </div>
          </div>
          <div class="cai-status{status_class}"><span class="cai-dot"></span>{status_text}</div>
        </header>
        """
    )


def render_setup_notice(missing: list[str]) -> None:
    rows = "".join(
        f"<li><code>{escape(key)}</code><span>{escape(REQUIRED_KEYS[key])}</span></li>"
        for key in missing
    )
    html(
        f"""
        <section class="cai-notice">
          <h3>Add your API keys to continue</h3>
          <p>CityAI needs these environment variables. Add them to a
          <code>.env</code> file next to <code>app.py</code>, then restart the app.</p>
          <ul>{rows}</ul>
        </section>
        """
    )


def render_empty_state() -> None:
    html(
        """
        <section class="cai-hero">
          <h1>Where should we explore?</h1>
          <p>Ask me about weather or the latest news from any city.</p>
        </section>
        """
    )
    with st.container(key="suggestions"):
        for col, text in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            col.button(text, key=f"suggest-{text}", on_click=submit, args=(text,),
                       use_container_width=True)


def chips_html(activity: list[dict]) -> str:
    chips = []
    for item in activity:
        copy = tool_copy(item["tool"])
        if item["status"] == "approved":
            label, cls = copy["done"], ""
        else:
            label, cls = f"{copy['title']} declined", " is-denied"
        city = escape(item.get("city", ""))
        detail = f'<span class="cai-chip-detail">{city}</span>' if city else ""
        chips.append(
            f'<span class="cai-chip{cls}"><span class="cai-chip-dot"></span>{escape(label)}{detail}</span>'
        )
    return f'<div class="cai-chips">{"".join(chips)}</div>' if chips else ""


def render_message(message: dict) -> None:
    role = message["role"]
    with st.chat_message(role):
        if role == "user":
            html(
                '<span class="cai-user"></span>'
                '<div class="cai-role">You</div>'
                f'<div class="cai-user-text">{escape(message["content"])}</div>'
            )
            return

        html(
            '<span class="cai-assistant"></span>'
            '<div class="cai-role"><span class="cai-role-mark"></span>CityAI</div>'
            + chips_html(message.get("activity", []))
        )
        if message.get("kind") == "error":
            html(f'<div class="cai-error">{escape(message["content"])}</div>')
        else:
            st.markdown(message["content"])


def render_working(status: str) -> None:
    """Assistant bubble with an animated activity line while the agent runs."""
    with st.chat_message("assistant"):
        html(
            '<span class="cai-assistant"></span>'
            '<div class="cai-role"><span class="cai-role-mark"></span>CityAI</div>'
            '<div class="cai-working"><span class="cai-working-dot"></span>'
            f'<span class="cai-shimmer">{escape(status)}…</span></div>'
        )


def running_status(actions: list[dict]) -> str:
    labels = []
    for action in actions:
        copy = tool_copy(action["name"])
        city = action_city(action)
        labels.append(f"{copy['running']} in {city}" if city else copy["running"])
    return " and ".join(labels)


def render_approval_card(action: dict, index: int, total: int) -> None:
    copy = tool_copy(action["name"])
    city = action_city(action)
    target = f" for <strong>{escape(city)}</strong>" if city else ""
    counter = f'<span class="cai-approval-count">{index + 1} of {total}</span>' if total > 1 else ""

    with st.container(key="approval"):
        html(
            f"""
            <div class="cai-approval-eyebrow"><span class="cai-dot"></span>Action requires approval{counter}</div>
            <div class="cai-approval-title">{escape(copy['title'])}</div>
            <div class="cai-approval-body">The assistant wants to {escape(copy['ask'])}{target}.</div>
            <div class="cai-approval-note">Nothing runs until you approve.</div>
            """
        )
        _, deny_col, approve_col = st.columns([3, 1, 1.15])
        deny_col.button("Deny", key=f"deny-{index}", use_container_width=True,
                        on_click=record_decision, args=(False,))
        approve_col.button("Approve", key=f"approve-{index}", type="primary",
                           use_container_width=True, on_click=record_decision, args=(True,))


# =========================================================================== #
# State and agent flow
# =========================================================================== #
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("thread_id", str(uuid.uuid4()))
    ss.setdefault("messages", [])   # what the UI displays
    ss.setdefault("pending", None)  # tool calls waiting for approval
    ss.setdefault("job", None)      # next agent call to run
    ss.setdefault("activity", [])   # tool activity for the current turn


def submit(prompt: str) -> None:
    ss = st.session_state
    ss.messages.append({"role": "user", "content": prompt})
    ss.activity = []
    ss.job = {
        "payload": {"messages": [{"role": "user", "content": prompt}]},
        "status": "Thinking",
    }


def record_decision(approved: bool) -> None:
    """Approve / Deny button callback. Resumes the agent once all calls are decided."""
    ss = st.session_state
    pending = ss.pending
    action = pending["actions"][len(pending["decisions"])]

    decision = {"type": "approve"} if approved else {"type": "reject", "message": DENIAL_MESSAGE}
    pending["decisions"].append(decision)
    ss.activity.append(
        {"tool": action["name"], "city": action_city(action),
         "status": "approved" if approved else "denied"}
    )

    if len(pending["decisions"]) == len(pending["actions"]):
        approved_actions = [
            a for a, d in zip(pending["actions"], pending["decisions"]) if d["type"] == "approve"
        ]
        ss.job = {
            "payload": Command(resume={"decisions": pending["decisions"]}),
            "status": running_status(approved_actions) if approved_actions else "Wrapping up",
        }
        ss.pending = None


def run_job(agent) -> None:
    """Run the queued agent call, then store an approval request or the answer."""
    ss = st.session_state
    job = ss.job
    render_working(job["status"])
    config = {"configurable": {"thread_id": ss.thread_id}}

    try:
        result = agent.invoke(job["payload"], config=config)
    except Exception as exc:  # shown as a polished message, never a traceback
        ss.messages.append(
            {"role": "assistant", "kind": "error", "content": describe_error(exc),
             "activity": ss.activity}
        )
        ss.activity, ss.job, ss.pending = [], None, None
        st.rerun()

    actions = extract_actions(result.get("__interrupt__"))
    if actions:
        ss.pending = {"actions": actions, "decisions": []}
    else:
        text = to_text(result["messages"][-1].content) or (
            "I couldn't put together a response that time. Try rephrasing your question."
        )
        ss.messages.append({"role": "assistant", "content": text, "activity": ss.activity})
        ss.activity = []
    ss.job = None
    st.rerun()


# =========================================================================== #
# Page
# =========================================================================== #
def main() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    init_state()
    ss = st.session_state

    problems = missing_keys()
    render_header(online=not problems)
    if problems:
        render_setup_notice(problems)
        st.stop()

    try:
        agent = get_agent()
    except Exception as exc:
        render_message({"role": "assistant", "kind": "error", "content": describe_error(exc)})
        st.stop()

    if not ss.messages and not ss.job and not ss.pending:
        render_empty_state()
    for message in ss.messages:
        render_message(message)

    if ss.pending:
        done = len(ss.pending["decisions"])
        render_approval_card(ss.pending["actions"][done], done, len(ss.pending["actions"]))

    prompt = st.chat_input("Ask about a city...", disabled=bool(ss.pending or ss.job))
    if prompt:
        submit(prompt)
        st.rerun()

    if ss.job:
        run_job(agent)


main()