# =============================================================================
# FILE: 23_agents_react.py
# PART: 6 - Tools & Agents  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   ReAct stands for Reason + Act.
#   Unlike a chain (which follows a fixed path), an agent THINKS.
#
#   The agent's inner monologue:
#     Thought: "I need to know the customer's subscription plan."
#     Action : lookup_customer(customer_id="C001")
#     Observation: "Plan: Professional, Monthly spend: $299"
#     Thought: "Now I have the info. I can answer."
#     Final Answer: "Customer C001 is on the Professional plan at $299/month."
#
#   Set verbose=True to watch this reasoning unfold in real time.
#   That's what makes agents fascinating — the visible thought process.
#
# WHAT YOU WILL LEARN:
#   1. create_react_agent() — build a ReAct agent
#   2. AgentExecutor — the loop runner
#   3. verbose=True — see Thought/Action/Observation in action
#   4. max_iterations — prevent infinite loops
#   5. handle_parsing_errors — recover from malformed tool calls
#   6. The ReAct reasoning loop visualized
#
# HOW THIS CONNECTS:
#   Previous: 22_tools_custom.py — building custom tools
#   Next:     24_agents_openai_functions.py — modern tool-calling agents
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_core.tools.base import ToolException
from langchain_core.prompts import PromptTemplate
from datetime import datetime
import math

load_dotenv()

print("=" * 60)
print("  CHAPTER 23: ReAct Agents")
print("=" * 60)

# =============================================================================
# SETUP: LLM and Tools
# =============================================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,  # Agents work better with low temperature for consistent tool calls
)

# Tools for this agent
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression. Use Python syntax.
    Supports: +, -, *, /, **, %, sqrt, abs, round, pi, e
    Examples: '2 + 2', '15 * 8', '2 ** 10', 'sqrt(144)'
    """
    safe_env = {"sqrt": math.sqrt, "abs": abs, "round": round, "pi": math.pi, "e": math.e}
    try:
        import ast
        tree = ast.parse(expression, mode='eval')
        result = eval(compile(tree, '<string>', 'eval'), {"__builtins__": {}}, safe_env)
        return f"{expression} = {result}"
    except Exception as e:
        raise ToolException(f"Calculation error: {e}")

@tool
def get_current_date() -> str:
    """Get today's date. Use this when the user asks about the current date."""
    return datetime.now().strftime("%A, %B %d, %Y")

@tool
def get_country_capital(country: str) -> str:
    """
    Get the capital city of a country.
    Use this for geography questions about capital cities.
    """
    capitals = {
        "france": "Paris", "germany": "Berlin", "japan": "Tokyo",
        "india": "New Delhi", "usa": "Washington D.C.", "united states": "Washington D.C.",
        "uk": "London", "united kingdom": "London", "china": "Beijing",
        "australia": "Canberra", "brazil": "Brasília", "canada": "Ottawa",
        "russia": "Moscow", "south africa": "Pretoria", "egypt": "Cairo",
        "nigeria": "Abuja", "argentina": "Buenos Aires", "mexico": "Mexico City",
    }
    country_lower = country.lower().strip()
    capital = capitals.get(country_lower, None)
    if not capital:
        raise ToolException(f"Capital for '{country}' not in database. Try the country's full name.")
    return f"The capital of {country.title()} is {capital}."

@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert between common units: km/miles, kg/lbs, celsius/fahrenheit, meters/feet.
    Example: value=5, from_unit='km', to_unit='miles'
    """
    conversions = {
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("celsius", "fahrenheit"): lambda x: x * 9/5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("meters", "feet"): lambda x: x * 3.28084,
        ("feet", "meters"): lambda x: x * 0.3048,
    }
    key = (from_unit.lower(), to_unit.lower())
    if key not in conversions:
        raise ToolException(f"Conversion from {from_unit} to {to_unit} not supported.")
    result = conversions[key](value)
    return f"{value} {from_unit} = {result:.4f} {to_unit}"

tools = [calculator, get_current_date, get_country_capital, convert_units]

# =============================================================================
# SECTION 1: THE ReAct PROMPT
# =============================================================================
# The ReAct prompt template is critical — it instructs the agent HOW to reason.
# It uses a specific format:
#   Thought: (reasoning step)
#   Action: (tool_name)
#   Action Input: (tool arguments)
#   Observation: (tool result)
#   ... (repeat until...)
#   Final Answer: (the answer to give the user)

print("\n--- Section 1: The ReAct Prompt ---")

# Build the ReAct prompt manually (can also use hub.pull("hwchase17/react"))
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

react_prompt = PromptTemplate.from_template(react_prompt_template)

print("ReAct prompt template loaded.")
print("Key variables: {tools}, {tool_names}, {input}, {agent_scratchpad}")

# =============================================================================
# SECTION 2: CREATE THE ReAct AGENT
# =============================================================================

print("\n--- Section 2: Creating the ReAct Agent ---")

# create_react_agent binds: LLM + tools + ReAct prompt
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

# AgentExecutor runs the ReAct loop:
# invoke → agent decides action → call tool → observation → agent decides again
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,            # Show Thought/Action/Observation (VERY educational!)
    max_iterations=10,       # Stop if agent gets stuck in a loop (safety)
    handle_parsing_errors=True,  # Don't crash on malformed tool call format
)

print("ReAct agent and AgentExecutor ready.")

# =============================================================================
# SECTION 3: WATCH THE AGENT REASON
# =============================================================================
print()
print("=" * 50)
print("  DEMO: Watching the Agent Reason")
print("=" * 50)

# Task 1: Simple tool use
print("\n[Task 1]: Simple calculation")
result1 = agent_executor.invoke({
    "input": "What is 15% of 2340? Round to 2 decimal places."
})
print(f"\nFinal Answer: {result1['output']}")

# Task 2: Multiple tool use
print("\n" + "="*50)
print("[Task 2]: Multiple tools in sequence")
result2 = agent_executor.invoke({
    "input": "What is the capital of Japan? And if the flight from Mumbai to that capital is 4850 km, how many miles is that?"
})
print(f"\nFinal Answer: {result2['output']}")

# Task 3: Multi-step reasoning
print("\n" + "="*50)
print("[Task 3]: Multi-step reasoning")
result3 = agent_executor.invoke({
    "input": "Today is what date? Also, what is 2 to the power of 8?"
})
print(f"\nFinal Answer: {result3['output']}")

# =============================================================================
# SECTION 4: WHAT MAKES ReAct WORK (AND FAIL)
# =============================================================================
print("\n--- Section 4: ReAct Strengths and Limitations ---")
print("""
  STRENGTHS:
    + Works with any LLM that can follow text instructions
    + Transparent reasoning — you can see every step
    + Easy to debug — check the Thought/Action/Observation chain
    + No special model support needed

  LIMITATIONS:
    - Slower than direct tool calling (more LLM calls for reasoning)
    - Relies on text parsing — can misformat tool calls
    - Use max_iterations to prevent infinite loops
    - Not great for very complex multi-agent tasks (use LangGraph instead)
    - handle_parsing_errors=True is essential in production

  WHEN TO USE ReAct:
    + Simple single-agent tasks with 2-5 tools
    + When you need visible reasoning for auditing/debugging
    + When the model doesn't support native tool calling

  ALTERNATIVE:
    → 24_agents_openai_functions.py (create_tool_calling_agent)
    Uses native JSON tool calling — more reliable, less text parsing
""")

# =============================================================================
# INTERACTIVE MODE
# =============================================================================
print()
print("=" * 50)
print("  Interactive ReAct Agent (type 'exit' to stop)")
print("  Available tools:", [t.name for t in tools])
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
print("  KEY INSIGHTS FROM CHAPTER 23")
print("=" * 60)
print("""
  1. ReAct = Reason + Act: Thought → Action → Observation → repeat
  2. create_react_agent(llm, tools, prompt) creates the reasoning engine
  3. AgentExecutor runs the ReAct loop and stops at Final Answer or max_iterations
  4. verbose=True shows every reasoning step — essential for learning and debugging
  5. max_iterations prevents infinite loops (always set this!)
  6. handle_parsing_errors=True prevents crashes on malformed tool call format
  7. Tools need clear docstrings — the agent reads them to decide which to use

  Next up: 24_agents_openai_functions.py
  The modern alternative: create_tool_calling_agent — faster and more reliable.
""")
