# =============================================================================
# FILE: 24_agents_openai_functions.py
# PART: 6 - Tools & Agents  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   ReAct agents think in TEXT — they write "Action: calculator"
#   and hope the parser reads it correctly. Sometimes they typo.
#   Sometimes the parser fails. Sometimes the tool call is malformed.
#
#   OpenAI Function Calling is different.
#   Instead of text, the model fills a STRUCTURED JSON FORM.
#   No text parsing. No format guessing. Fewer failures.
#
#   create_tool_calling_agent is the modern, preferred approach.
#   Works with any model that supports tool calling (GPT-4o, Claude, Gemini, etc.)
#
# WHAT YOU WILL LEARN:
#   1. create_tool_calling_agent() — modern, reliable agent
#   2. The agent_scratchpad placeholder — where tool history goes
#   3. Streaming agent output — see tool calls and answers as they arrive
#   4. Comparing ReAct vs Tool Calling side by side
#   5. return_intermediate_steps — inspect tool call history
#
# HOW THIS CONNECTS:
#   Previous: 23_agents_react.py — ReAct agents with text-based reasoning
#   Next:     25_agents_multi_tool.py — full multi-tool agent with memory
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.tools.base import ToolException
from datetime import datetime
import math

load_dotenv()

print("=" * 60)
print("  CHAPTER 24: Tool-Calling Agents (Modern Approach)")
print("=" * 60)

# =============================================================================
# SETUP: LLM and Tools
# =============================================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,
)

# Reusing tools from chapter 22-23 (same tools, different agent)
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Supports +, -, *, /, **, sqrt, abs, round."""
    safe_env = {"sqrt": math.sqrt, "abs": abs, "round": round, "pi": math.pi}
    try:
        import ast
        result = eval(compile(ast.parse(expression, mode='eval'), '<string>', 'eval'),
                      {"__builtins__": {}}, safe_env)
        return f"{expression} = {result}"
    except Exception as e:
        raise ToolException(f"Calculation error: {e}")

@tool
def get_capital(country: str) -> str:
    """Get the capital city of a country."""
    capitals = {
        "france": "Paris", "germany": "Berlin", "japan": "Tokyo",
        "india": "New Delhi", "usa": "Washington D.C.", "brazil": "Brasília",
        "australia": "Canberra", "canada": "Ottawa", "china": "Beijing",
    }
    capital = capitals.get(country.lower().strip())
    if not capital:
        raise ToolException(f"Capital for '{country}' not found.")
    return f"The capital of {country.title()} is {capital}."

@tool
def get_today() -> str:
    """Get today's date and day of the week."""
    return datetime.now().strftime("%A, %B %d, %Y")

@tool
def word_counter(text: str) -> str:
    """Count words, sentences, and characters in a text."""
    words = len(text.split())
    sentences = len([s for s in text.split('.') if s.strip()])
    return f"Words: {words}, Sentences: {sentences}, Characters: {len(text)}"

tools = [calculator, get_capital, get_today, word_counter]

# =============================================================================
# SECTION 1: THE TOOL-CALLING PROMPT
# =============================================================================
# Key difference from ReAct:
# The prompt doesn't need to explain the Thought/Action/Observation format.
# The model knows to call tools natively.
# The ONLY required special element is MessagesPlaceholder("agent_scratchpad")
# — this is where tool call history (tool messages, tool responses) gets stored.

print("\n--- Section 1: Tool-Calling Prompt Structure ---")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant with access to tools. "
        "Use tools when needed to answer accurately. "
        "Be concise in your final answer."
    ),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),  # ← Tool call history goes HERE
])

print("Prompt structure:")
print("  1. System message (agent's role and instructions)")
print("  2. Human message (user's question: {input})")
print("  3. agent_scratchpad (tool call history — managed automatically)")

# =============================================================================
# SECTION 2: CREATE THE TOOL-CALLING AGENT
# =============================================================================

print("\n--- Section 2: Creating the Tool-Calling Agent ---")

# create_tool_calling_agent: binds LLM + tools + prompt
# Under the hood: LLM gets tool schemas attached via llm.bind_tools(tools)
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# AgentExecutor: manages the tool-call → response → tool-call loop
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # See tool calls and responses
    max_iterations=10,
    handle_tool_error=True,    # Catch ToolException, pass error to agent
)

print("Tool-calling agent and AgentExecutor ready.")

# =============================================================================
# SECTION 3: RUNNING THE AGENT
# =============================================================================
print()
print("=" * 50)
print("  DEMO: Tool-Calling Agent in Action")
print("=" * 50)

# Task 1: Single tool
print("\n[Task 1]: Single tool use")
result = agent_executor.invoke({"input": "What is 2 to the power of 12?"})
print(f"\nAnswer: {result['output']}")

# Task 2: Two tools in sequence
print("\n[Task 2]: Two tools in sequence")
result = agent_executor.invoke({
    "input": "What is the capital of Australia? And how many letters does that city name have?"
})
print(f"\nAnswer: {result['output']}")

# Task 3: Multiple tools
print("\n[Task 3]: Multi-tool task")
result = agent_executor.invoke({
    "input": "What day is today? Also, what is the square root of today's day number multiplied by 10? (Use the date's day number)"
})
print(f"\nAnswer: {result['output']}")

# =============================================================================
# SECTION 4: STREAMING AGENT OUTPUT
# =============================================================================
# Agent streaming works differently from LLM streaming.
# .stream() yields chunks for each STAGE: action, observation, final answer.

print("\n--- Section 4: Streaming Agent Output ---")

print("\nStreaming a task (watch chunks arrive by stage):")
print("-" * 40)

for chunk in agent_executor.stream({
    "input": "Count the words in this sentence: 'The quick brown fox jumps over the lazy dog.'"
}):
    # Chunk types:
    # {"actions": [...]}    - when agent calls a tool
    # {"steps": [...]}      - when tool returns a result
    # {"output": "..."}     - when agent gives final answer

    if "actions" in chunk:
        for action in chunk["actions"]:
            print(f"  [TOOL CALL] {action.tool}({action.tool_input})")

    elif "steps" in chunk:
        for step in chunk["steps"]:
            print(f"  [TOOL RESULT] {step.observation[:100]}")

    elif "output" in chunk:
        print(f"  [FINAL ANSWER] {chunk['output']}")

# =============================================================================
# SECTION 5: INTERMEDIATE STEPS — INSPECT TOOL CALL HISTORY
# =============================================================================
print("\n--- Section 5: Inspecting Intermediate Steps ---")

# Return intermediate steps to see the full tool call history
result_with_steps = agent_executor.invoke(
    {"input": "What is the capital of France? And what is 100 divided by that city's name length?"},
)

# Note: intermediate_steps are in result['intermediate_steps'] if returned
print(f"\nFinal answer: {result_with_steps['output']}")

# =============================================================================
# SECTION 6: ReAct vs TOOL CALLING — COMPARISON
# =============================================================================
print("\n--- Section 6: ReAct vs Tool Calling ---")
print("""
  ┌──────────────────────┬──────────────────────┬──────────────────────┐
  │ Feature              │ ReAct Agent          │ Tool-Calling Agent   │
  ├──────────────────────┼──────────────────────┼──────────────────────┤
  │ How tools are called │ Text parsing         │ Structured JSON API  │
  │ Reliability          │ Lower (text errors)  │ Higher (schema)      │
  │ Model support        │ Any LLM              │ Needs tool support   │
  │ Transparency         │ Full thought chain   │ Tool calls visible   │
  │ Speed                │ Slower               │ Faster               │
  │ Error rate           │ Higher               │ Lower                │
  │ Prompt complexity    │ Complex format rules │ Simple               │
  │ Recommended for      │ Older models, custom │ Production apps      │
  └──────────────────────┴──────────────────────┴──────────────────────┘

  RECOMMENDATION:
  → Use create_tool_calling_agent for all new projects
  → Use create_react_agent only if your model doesn't support tool calling
  → Both use the same AgentExecutor — easy to swap
""")

# =============================================================================
# INTERACTIVE MODE
# =============================================================================
print("=" * 50)
print("  Interactive Tool-Calling Agent (type 'exit' to stop)")
print(f"  Tools available: {[t.name for t in tools]}")
print("=" * 50 + "\n")

while True:
    user_input = input("You: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    try:
        result = agent_executor.invoke({"input": user_input})
        print(f"\nAgent: {result['output']}\n")
    except Exception as e:
        print(f"\nError: {e}\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 24")
print("=" * 60)
print("""
  1. create_tool_calling_agent is the modern, preferred approach
  2. Prompt needs MessagesPlaceholder("agent_scratchpad") — tool history goes here
  3. No special format instructions needed — the model handles tool calling natively
  4. Tool-calling uses structured JSON — much more reliable than text parsing
  5. .stream() yields chunks: actions (tool calls), steps (results), output (final answer)
  6. handle_tool_error=True prevents crashes on ToolException
  7. Works with GPT-4o, Claude, Gemini, Llama 3 (any tool-calling model)

  Next up: 25_agents_multi_tool.py
  A complete customer support agent with 5 tools AND conversation memory.
""")
