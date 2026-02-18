# =============================================================================
# FILE: 09_output_parsers_advanced.py
# PART: 2 - LCEL Deep Dive  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Sometimes the LLM refuses to follow your format instructions perfectly.
#   It adds extra text, forgets a curly brace, or uses the wrong key name.
#
#   OutputFixingParser is the safety net. When the primary parser fails,
#   it sends the malformed output BACK to the LLM with a "fix this" prompt.
#   The LLM sees its own mistake and corrects it. Automated error correction.
#
# WHAT YOU WILL LEARN:
#   1. OutputFixingParser — auto-fix malformed LLM output with another LLM call
#   2. Streaming with JsonOutputParser — partial JSON as it arrives
#   3. Custom output parser — subclass BaseOutputParser for any format
#   4. When parsers fail and how to debug them
#
# HOW THIS CONNECTS:
#   Previous: 08_lcel_runnables.py — retry, fallback, config, bind
#   Next:     10_structured_output.py — llm.with_structured_output() (modern way)
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    BaseOutputParser,
)
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Any

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

print("=" * 60)
print("  CHAPTER 9: Advanced Output Parsers")
print("=" * 60)

# =============================================================================
# SECTION 1: WHAT HAPPENS WHEN A PARSER FAILS?
# =============================================================================
# Let's intentionally create a situation where the LLM gives bad JSON.
# This demonstrates the problem that OutputFixingParser solves.

print("\n--- Section 1: When Parsers Fail ---")

# Force the LLM to produce bad JSON by giving a misleading instruction
bad_prompt = ChatPromptTemplate.from_template(
    "Return the following text exactly as-is without changes:\n"
    "{{name: Alice, age: thirty, hobbies: [painting, reading]}}\n"
    "Note: This is NOT valid JSON — it has unquoted keys and a text number."
)

raw_output = (bad_prompt | llm | StrOutputParser()).invoke({})
print(f"\nLLM produced: {raw_output}")

# Now try to parse it as JSON — this will fail
try:
    import json
    json.loads(raw_output)
    print("Parsed successfully (unexpected!)")
except json.JSONDecodeError as e:
    print(f"\nJSON parsing failed (as expected): {e}")
    print("This is where OutputFixingParser saves the day.")

# =============================================================================
# SECTION 2: OutputFixingParser — Auto-Fix Malformed Output
# =============================================================================
# OutputFixingParser wraps another parser.
# When the wrapped parser fails, it calls the LLM again with:
#   "You output this: <bad_output>. It failed parsing with: <error>. Fix it."
# The LLM self-corrects. Magic.

print("\n--- Section 2: OutputFixingParser — The Safety Net ---")

from langchain.output_parsers import OutputFixingParser

# Define what we expect
class PersonInfo(BaseModel):
    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age as an integer")
    hobbies: List[str] = Field(description="List of hobbies")

base_parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Wrap it with OutputFixingParser
# If base_parser fails, fixing_parser uses the LLM to fix the bad output
fixing_parser = OutputFixingParser.from_llm(parser=base_parser, llm=llm)

# Test: give the fixing parser raw "broken" text
# Simulating what a bad LLM output might look like
bad_output_text = "name: Alice, age: thirty, hobbies: painting and reading"

print(f"\nAttempting to fix malformed output:")
print(f"  Input text: '{bad_output_text}'")

try:
    fixed_result: PersonInfo = fixing_parser.parse(bad_output_text)
    print(f"\nFixed result:")
    print(f"  Name   : {fixed_result.name}")
    print(f"  Age    : {fixed_result.age}")
    print(f"  Hobbies: {fixed_result.hobbies}")
    print("(OutputFixingParser automatically corrected the malformed input!)")
except Exception as e:
    print(f"\nNote: OutputFixingParser made its best effort. Error: {e}")

# =============================================================================
# SECTION 3: Streaming with JsonOutputParser
# =============================================================================
# JsonOutputParser supports streaming! As the LLM generates JSON token by token,
# the parser assembles it incrementally and yields progressively more complete dicts.
# Each yielded chunk is a partial (but growing) Python dict.

print("\n--- Section 3: Streaming JSON (Partial Chunks) ---")

json_stream_prompt = ChatPromptTemplate.from_template(
    "Create a detailed profile for a fictional character: name, age, city, "
    "occupation, hobbies (list of 3), and a short bio (2 sentences).\n"
    "Return ONLY valid JSON. {format_instructions}"
)

json_parser = JsonOutputParser()
streaming_json_chain = json_stream_prompt | llm | json_parser

print("\nStreaming JSON output (watch it build up):")
print("(Each line shows the progressively assembled dict)")

previous_keys = set()
for partial_dict in streaming_json_chain.stream({
    "format_instructions": json_parser.get_format_instructions()
}):
    # Only print when new keys appear (cleaner output)
    new_keys = set(partial_dict.keys()) - previous_keys
    if new_keys:
        for k in new_keys:
            print(f"  {k}: {partial_dict[k]}")
    previous_keys = set(partial_dict.keys())

print("\n(Full dict assembled from streaming tokens)")

# =============================================================================
# SECTION 4: Building a Custom Output Parser
# =============================================================================
# Sometimes you need a parser that doesn't exist in LangChain.
# Subclass BaseOutputParser[T] and implement:
#   - parse(text: str) → T   (the actual parsing logic)
#   - _type property          (a unique name for this parser)

print("\n--- Section 4: Custom Output Parser ---")

class NumberedListParser(BaseOutputParser[List[str]]):
    """
    Parses a numbered list like:
      1. Item one
      2. Item two
      3. Item three
    And returns: ["Item one", "Item two", "Item three"]
    """

    @property
    def _type(self) -> str:
        return "numbered_list_parser"

    def parse(self, text: str) -> List[str]:
        items = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # Match lines starting with a number and period: "1. ", "2. ", etc.
            if line and line[0].isdigit() and "." in line:
                # Remove the numbering prefix
                content = line.split(".", 1)[-1].strip()
                if content:
                    items.append(content)
        return items

    def get_format_instructions(self) -> str:
        return "Respond with a numbered list. Each item on its own line: 1. Item"


# Use the custom parser in a chain
numbered_parser = NumberedListParser()

list_prompt = ChatPromptTemplate.from_template(
    "List 5 tips for learning {skill}.\n{format_instructions}"
)

custom_chain = list_prompt | llm | numbered_parser

result_list: List[str] = custom_chain.invoke({
    "skill": "Python programming",
    "format_instructions": numbered_parser.get_format_instructions()
})

print(f"\nCustom parser result (List[str]):")
for i, tip in enumerate(result_list, 1):
    print(f"  {i}. {tip}")

# =============================================================================
# SECTION 5: DEBUGGING PARSER ISSUES
# =============================================================================
print("\n--- Section 5: Debugging Tips ---")
print("""
  When a parser fails, debug like this:

  Step 1: Get the raw LLM output first
    raw_chain = prompt | llm | StrOutputParser()
    raw_output = raw_chain.invoke(inputs)
    print("Raw output:", raw_output)

  Step 2: Try parsing manually
    try:
        result = parser.parse(raw_output)
    except Exception as e:
        print("Parse error:", e)

  Step 3: Check format_instructions
    print(parser.get_format_instructions())
    # Make sure your prompt actually includes {format_instructions}

  Step 4: Use OutputFixingParser as a wrapper
    fixing_parser = OutputFixingParser.from_llm(parser=your_parser, llm=llm)
    # This adds an automatic retry with self-correction
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 9")
print("=" * 60)
print("""
  1. OutputFixingParser wraps any parser and auto-fixes malformed output
  2. It works by calling the LLM again with the error message
  3. JsonOutputParser supports streaming — yields partial dicts as tokens arrive
  4. Custom parsers: subclass BaseOutputParser[T], implement parse() and _type
  5. Always test with raw StrOutputParser first when debugging
  6. get_format_instructions() generates the text that guides the LLM's format

  Next up: 10_structured_output.py
  The modern way: llm.with_structured_output() — no prompt engineering needed.
""")
