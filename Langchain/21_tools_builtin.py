# =============================================================================
# FILE: 21_tools_builtin.py
# PART: 6 - Tools & Agents  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   An LLM is brilliant but blind.
#   It can't browse the web. It can't check today's stock price.
#   It can't tell you the current weather. It lives in a knowledge snapshot.
#
#   Tools are its hands and eyes.
#   When you give an LLM a tool, it can decide WHEN and HOW to use it.
#   "I need today's weather." → uses get_weather tool
#   "I need to calculate 147 * 23." → uses calculator tool
#
#   This file covers built-in tools ready to use out of the box.
#
# WHAT YOU WILL LEARN:
#   1. What a Tool is: name + description + function
#   2. @tool decorator — the simplest way to create a tool
#   3. Tool introspection: .name, .description, .args
#   4. tool.invoke() — calling a tool directly (before adding to an agent)
#   5. Built-in tools: DuckDuckGoSearchRun, WikipediaQueryRun
#   6. Tool with Pydantic args_schema for complex inputs
#
# HOW THIS CONNECTS:
#   Previous: 20_rag_conversational.py — conversational RAG
#   Next:     22_tools_custom.py — building your own tools
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

print("=" * 60)
print("  CHAPTER 21: Built-in Tools")
print("=" * 60)

# =============================================================================
# SECTION 1: THE @tool DECORATOR — SIMPLEST FORM
# =============================================================================
# The @tool decorator converts any Python function into a LangChain Tool.
# Rules:
#   1. The function's DOCSTRING becomes the tool's description
#      (The LLM reads this to decide when to use the tool)
#   2. Type hints on parameters become the tool's input schema
#   3. The function name becomes the tool's name

print("\n--- Section 1: The @tool Decorator ---")

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """
    Calculate the Body Mass Index (BMI) given weight and height.
    Use this when someone asks about their BMI or weight category.

    Args:
        weight_kg: Body weight in kilograms
        height_m: Height in meters (e.g., 1.75 for 175cm)
    """
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"

    return f"BMI: {bmi:.2f} ({category})"

# =============================================================================
# SECTION 2: INSPECTING A TOOL
# =============================================================================
# Every tool exposes its metadata — this is what the LLM agent reads
# to decide when and how to use each tool.

print("\n--- Section 2: Tool Introspection ---")

print(f"\nTool name       : {calculate_bmi.name}")
print(f"Tool description: {calculate_bmi.description[:100]}...")
print(f"Tool args schema: {calculate_bmi.args}")
print(f"Tool return type: (function returns a string)")

# =============================================================================
# SECTION 3: CALLING A TOOL DIRECTLY WITH .invoke()
# =============================================================================
# You can test tools directly with .invoke() before wiring them to an agent.
# This is essential for debugging — always test tools in isolation first.

print("\n--- Section 3: Calling a Tool Directly ---")

# Method 1: Pass a dict of arguments
result = calculate_bmi.invoke({"weight_kg": 75, "height_m": 1.80})
print(f"\nBMI result: {result}")

# Method 2: The tool handles type conversion automatically
result2 = calculate_bmi.invoke({"weight_kg": 90, "height_m": 1.65})
print(f"BMI result 2: {result2}")

# =============================================================================
# SECTION 4: MORE EXAMPLE TOOLS
# =============================================================================

print("\n--- Section 4: More Tool Examples ---")

@tool
def get_current_time() -> str:
    """Get the current date and time. Use this when the user asks about the current time or date."""
    from datetime import datetime
    now = datetime.now()
    return f"Current date and time: {now.strftime('%A, %B %d, %Y at %I:%M %p')}"

@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """
    Convert temperature between Celsius, Fahrenheit, and Kelvin.

    Args:
        value: The temperature value to convert
        from_unit: Source unit - 'celsius', 'fahrenheit', or 'kelvin'
        to_unit: Target unit - 'celsius', 'fahrenheit', or 'kelvin'
    """
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    # Convert to Celsius first
    if from_unit == "celsius":
        celsius = value
    elif from_unit == "fahrenheit":
        celsius = (value - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = value - 273.15
    else:
        return f"Unknown unit: {from_unit}"

    # Convert from Celsius to target
    if to_unit == "celsius":
        result = celsius
    elif to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    else:
        return f"Unknown unit: {to_unit}"

    return f"{value}° {from_unit.capitalize()} = {result:.2f}° {to_unit.capitalize()}"

@tool
def count_words(text: str) -> str:
    """Count the number of words, sentences, and characters in a text."""
    words = len(text.split())
    sentences = len([s for s in text.split('.') if s.strip()])
    chars = len(text)
    return f"Words: {words}, Sentences: {sentences}, Characters: {chars}"

# Test the tools
print(f"\nTime: {get_current_time.invoke({})}")
print(f"\nTemperature: {convert_temperature.invoke({'value': 100, 'from_unit': 'celsius', 'to_unit': 'fahrenheit'})}")
print(f"\nWord count: {count_words.invoke({'text': 'LangChain is a powerful framework. It helps build AI applications.'})}")

# =============================================================================
# SECTION 5: TOOL WITH Pydantic args_schema
# =============================================================================
# For tools with complex inputs, define a Pydantic schema.
# This gives the LLM precise field descriptions for each argument.

print("\n--- Section 5: Tool with Pydantic Schema ---")

class WeatherQuery(BaseModel):
    """Input schema for the weather tool."""
    city: str = Field(description="The city name to get weather for")
    country_code: str = Field(
        default="IN",
        description="2-letter country code (e.g., 'US', 'IN', 'GB')"
    )
    unit: str = Field(
        default="celsius",
        description="Temperature unit: 'celsius' or 'fahrenheit'"
    )

@tool(args_schema=WeatherQuery)
def get_weather(city: str, country_code: str = "IN", unit: str = "celsius") -> str:
    """
    Get the current weather for a city. Returns temperature, condition, and humidity.
    Use this when the user asks about weather in any city.
    """
    # NOTE: This is a mock — in production, call a real weather API
    weather_data = {
        "Mumbai": {"temp_c": 32, "condition": "Humid and partly cloudy", "humidity": 85},
        "London": {"temp_c": 15, "condition": "Overcast with light rain", "humidity": 90},
        "Tokyo": {"temp_c": 22, "condition": "Clear and sunny", "humidity": 60},
        "New York": {"temp_c": 18, "condition": "Windy with some clouds", "humidity": 55},
    }

    data = weather_data.get(city, {"temp_c": 20, "condition": "Partly cloudy", "humidity": 65})
    temp = data["temp_c"] if unit == "celsius" else data["temp_c"] * 9/5 + 32
    unit_symbol = "°C" if unit == "celsius" else "°F"

    return (
        f"Weather in {city}, {country_code}: "
        f"{temp:.0f}{unit_symbol}, {data['condition']}, "
        f"Humidity: {data['humidity']}%"
    )

# Inspect the schema
print(f"\nget_weather args schema:")
for field_name, field_info in get_weather.args.items():
    print(f"  {field_name}: {field_info}")

# Test it
result = get_weather.invoke({"city": "Mumbai", "country_code": "IN", "unit": "celsius"})
print(f"\n{result}")

result2 = get_weather.invoke({"city": "London", "unit": "fahrenheit"})
print(f"{result2}")

# =============================================================================
# SECTION 6: BUILT-IN TOOLS — DuckDuckGo and Wikipedia
# =============================================================================
print("\n--- Section 6: Built-in Community Tools ---")

# DuckDuckGoSearchRun — web search (no API key needed!)
print("\n[DuckDuckGo Search Tool]")
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()

    print(f"\nTool name: {search.name}")
    print(f"Tool desc: {search.description[:100]}...")

    # Uncomment to actually run (requires internet):
    # search_result = search.invoke("LangChain latest version 2025")
    # print(f"\nSearch result: {search_result[:300]}")
    print("(Uncomment the invoke() above to run a live search)")

except ImportError:
    print("Install: pip install duckduckgo-search")

# WikipediaQueryRun — Wikipedia lookup
print("\n[Wikipedia Tool]")
try:
    from langchain_community.tools import WikipediaQueryRun
    from langchain_community.utilities import WikipediaAPIWrapper

    wiki = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
    )

    print(f"Tool name: {wiki.name}")
    print(f"Tool desc: {wiki.description[:100]}...")

    # Uncomment to actually run:
    # wiki_result = wiki.invoke("LangChain AI framework")
    # print(f"\nWikipedia result: {wiki_result[:300]}")
    print("(Uncomment the invoke() above to fetch from Wikipedia)")

except ImportError:
    print("Install: pip install wikipedia")

# =============================================================================
# SECTION 7: TOOL COLLECTION — READY FOR AN AGENT
# =============================================================================
print("\n--- Section 7: Assembling a Tool Collection ---")

# Gather all your tools into a list
# This list gets passed to an agent in the next files
tools = [
    calculate_bmi,
    get_current_time,
    convert_temperature,
    count_words,
    get_weather,
]

print(f"\nTools ready for an agent ({len(tools)} total):")
for t in tools:
    print(f"  - {t.name:30} | {t.description[:50]}...")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 21")
print("=" * 60)
print("""
  1. @tool decorator: converts Python function → LangChain Tool
  2. The DOCSTRING is the tool description (the LLM reads this!)
  3. Type hints become the input schema
  4. tool.name, tool.description, tool.args — inspect any tool
  5. tool.invoke({"arg": value}) — test tools before adding to agents
  6. Pydantic args_schema gives precise field descriptions to the LLM
  7. Built-in tools: DuckDuckGoSearchRun (no API key), WikipediaQueryRun

  Next up: 22_tools_custom.py
  Building tools that talk to databases, APIs, and your own systems.
""")
