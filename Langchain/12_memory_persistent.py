# =============================================================================
# FILE: 12_memory_persistent.py
# PART: 3 - Memory  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   In-memory history evaporates when your script ends —
#   like chalk on a sidewalk in the rain.
#
#   Persistent history writes to disk. It survives restarts.
#   You can close the terminal, reopen it tomorrow, and the AI
#   still remembers what you talked about last week.
#
#   This is the foundation of real-world chatbot applications.
#
# WHAT YOU WILL LEARN:
#   1. FileChatMessageHistory — save conversation to a JSON file on disk
#   2. Inspecting the saved JSON file
#   3. How the conversation persists across script restarts
#   4. Conversation summarization — compress old messages to save tokens
#   5. BaseChatMessageHistory — the interface for any custom backend
#
# HOW THIS CONNECTS:
#   Previous: 11_memory_in_memory.py — in-memory history (lost on restart)
#   Next:     13_document_loaders.py — loading external documents into LangChain
# =============================================================================

import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini"
)

print("=" * 60)
print("  CHAPTER 12: Persistent Memory")
print("=" * 60)

# =============================================================================
# SECTION 1: FileChatMessageHistory — Save to Disk
# =============================================================================
# FileChatMessageHistory saves the conversation as a JSON file.
# Every message is immediately written to disk.
# On the next run, it reads the file and restores the conversation.

print("\n--- Section 1: FileChatMessageHistory ---")

try:
    from langchain_community.chat_message_histories import FileChatMessageHistory

    # Each session gets its own file
    history_dir = "Langchain/chat_histories"
    os.makedirs(history_dir, exist_ok=True)

    def get_file_session_history(session_id: str) -> FileChatMessageHistory:
        """Return history backed by a JSON file for this session."""
        file_path = f"{history_dir}/{session_id}.json"
        return FileChatMessageHistory(file_path)

    # Build the prompt and chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Remember what the user tells you about themselves."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm
    persistent_chat = RunnableWithMessageHistory(
        chain,
        get_file_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    session_config = {"configurable": {"session_id": "dinesh_persistent"}}
    file_path = f"{history_dir}/dinesh_persistent.json"

    # Check if we have an existing conversation
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            existing = json.load(f)
        msg_count = len(existing.get("messages", []))
        print(f"\nExisting conversation found with {msg_count} messages!")
        print("The AI remembers your previous conversation.")
    else:
        print("\nFirst time running — starting fresh conversation.")
        print("Run this script again to see persistence in action!")

    # Have a conversation
    r1 = persistent_chat.invoke(
        {"input": "Hello! I'm Dinesh. I'm learning LangChain for AI automation projects."},
        config=session_config,
    )
    print(f"\nYou: Hello! I'm Dinesh. I'm learning LangChain...")
    print(f"AI : {r1.content}")

    r2 = persistent_chat.invoke(
        {"input": "What's the most important LangChain concept I should learn first?"},
        config=session_config,
    )
    print(f"\nYou: What's the most important LangChain concept I should learn first?")
    print(f"AI : {r2.content[:300]}...")

    # =============================================================================
    # SECTION 2: INSPECTING THE SAVED FILE
    # =============================================================================
    print("\n--- Section 2: What's Inside the JSON File? ---")

    with open(file_path, "r") as f:
        saved_data = json.load(f)

    print(f"\nFile: {file_path}")
    print(f"Total messages saved: {len(saved_data.get('messages', []))}")
    print("\nFirst 2 messages structure:")
    for i, msg in enumerate(saved_data.get("messages", [])[:2]):
        print(f"\n  Message {i+1}:")
        print(f"    type   : {msg.get('type')}")
        print(f"    content: {str(msg.get('data', {}).get('content', ''))[:80]}...")

    print(f"\n(Full file: {file_path})")
    print("Open it to see the raw JSON — each message is stored with type and content.")

except ImportError:
    print("\nNote: FileChatMessageHistory requires langchain_community.")
    print("Install: pip install langchain-community")

# =============================================================================
# SECTION 3: MANUAL PERSISTENT HISTORY (Pure Python — No Dependencies)
# =============================================================================
# No extra packages needed — just Python's built-in json module.
# This teaches the underlying pattern that all persistence solutions use.

print("\n--- Section 3: Manual JSON-Based History ---")

MANUAL_HISTORY_FILE = "Langchain/manual_chat_history.json"

def load_history() -> list:
    """Load message history from JSON file."""
    if os.path.exists(MANUAL_HISTORY_FILE):
        with open(MANUAL_HISTORY_FILE, "r") as f:
            data = json.load(f)
            # Convert dicts back to message objects
            messages = []
            for msg in data:
                if msg["role"] == "human":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    messages.append(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    messages.append(SystemMessage(content=msg["content"]))
            return messages
    return []

def save_history(messages: list) -> None:
    """Save message history to JSON file."""
    data = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            data.append({"role": "human", "content": msg.content})
        elif isinstance(msg, AIMessage):
            data.append({"role": "ai", "content": msg.content})
        elif isinstance(msg, SystemMessage):
            data.append({"role": "system", "content": msg.content})
    with open(MANUAL_HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# Load any existing history
history = load_history()
print(f"\nLoaded {len(history)} messages from disk.")

# Add a new exchange
system_msg = SystemMessage(content="You are a helpful assistant.")
new_human = HumanMessage(content="What year was Python created?")

# Build the full message list for the LLM
all_messages = [system_msg] + history + [new_human]
response = llm.invoke(all_messages)

# Append to history and save
history.append(new_human)
history.append(AIMessage(content=response.content))
save_history(history)

print(f"AI response: {response.content}")
print(f"Saved {len(history)} messages to {MANUAL_HISTORY_FILE}")

# =============================================================================
# SECTION 4: CONVERSATION SUMMARIZATION STRATEGY
# =============================================================================
# Problem: After 50+ messages, history gets massive and expensive.
# Strategy: When history > 10 messages, summarize the OLD messages,
# then replace them with a summary message.
# Keep the last 4 messages verbatim (for recent context).

print("\n--- Section 4: Conversation Summarization Strategy ---")

from langchain_core.chat_history import InMemoryChatMessageHistory

def get_or_create_store():
    return {}

def summarize_history_if_needed(
    history: InMemoryChatMessageHistory,
    max_messages: int = 10
) -> None:
    """
    If history exceeds max_messages, summarize the older messages.
    Keeps the last 4 messages verbatim, summarizes everything before.
    """
    messages = history.messages
    if len(messages) <= max_messages:
        return  # No action needed

    # Split: older messages (to summarize) and recent messages (to keep)
    older_messages = messages[:-4]
    recent_messages = messages[-4:]

    # Use the LLM to create a summary of older messages
    summary_prompt = [
        SystemMessage(content="Summarize this conversation history concisely in 3-4 sentences."),
        *older_messages,
    ]
    summary_response = llm.invoke(summary_prompt)
    summary_text = f"[Earlier conversation summary]: {summary_response.content}"

    # Replace history: summary message + recent messages
    history.clear()
    history.add_message(SystemMessage(content=summary_text))
    for msg in recent_messages:
        history.add_message(msg)

    print(f"\n  Summarization applied: {len(messages)} messages → {len(history.messages)} messages")
    print(f"  Summary: {summary_text[:150]}...")

# Demo: Create an artificially long history and trigger summarization
print("\nCreating a long conversation history to trigger summarization...")
demo_store = {}

def get_demo_history(session_id: str):
    if session_id not in demo_store:
        demo_store[session_id] = InMemoryChatMessageHistory()
    return demo_store[session_id]

# Manually add 12 messages to trigger summarization
demo_history = get_demo_history("demo")
for i in range(6):
    demo_history.add_message(HumanMessage(content=f"Question {i+1}: Tell me about AI concept number {i+1}."))
    demo_history.add_message(AIMessage(content=f"AI concept {i+1} is about learning and adaptation."))

print(f"Before summarization: {len(demo_history.messages)} messages")
summarize_history_if_needed(demo_history, max_messages=10)
print(f"After summarization : {len(demo_history.messages)} messages")

# =============================================================================
# SECTION 5: THE BaseChatMessageHistory INTERFACE
# =============================================================================
print("\n--- Section 5: BaseChatMessageHistory — Build Your Own Backend ---")
print("""
  All history classes implement BaseChatMessageHistory with:
    .messages      → List of messages
    .add_message() → Add a single message
    .add_messages()→ Add multiple messages
    .clear()       → Delete all messages

  Popular implementations:
    InMemoryChatMessageHistory    — RAM only (built-in)
    FileChatMessageHistory        — JSON file (langchain_community)
    SQLiteChatMessageHistory      — SQLite database (langchain_community)
    RedisChatMessageHistory       — Redis (langchain_community)
    MongoDBChatMessageHistory     — MongoDB (langchain_community)

  To use any of them, just swap it into get_session_history():
    def get_session_history(session_id):
        return SQLiteChatMessageHistory(
            session_id=session_id,
            connection_string="sqlite:///chat_history.db"
        )
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 12")
print("=" * 60)
print("""
  1. FileChatMessageHistory saves to a JSON file — survives restarts
  2. Each session_id maps to its own file (or DB record)
  3. The JSON file stores: role + content for each message
  4. Manual approach: load_history() / save_history() with Python's json module
  5. Summarization strategy: compress old messages to control token costs
  6. BaseChatMessageHistory is the interface — swap backends without changing code

  PART 3 COMPLETE — Memory fully mastered.

  Next up: 13_document_loaders.py
  Loading text, PDFs, CSV, JSON, and web pages into LangChain.
""")
