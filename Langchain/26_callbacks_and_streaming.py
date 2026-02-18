# =============================================================================
# FILE: 26_callbacks_and_streaming.py
# PART: 7 - Callbacks & Streaming  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   Callbacks are LangChain's event system.
#   Every time the LLM starts thinking, a chain begins, or a tool gets called,
#   an event FIRES. You can listen to any of these events and add your logic.
#
#   Use cases:
#   - Logging every LLM call to a file
#   - Tracking token usage across a session
#   - Measuring latency of each chain step
#   - Sending alerts when cost exceeds a threshold
#
# WHAT YOU WILL LEARN:
#   1. StdOutCallbackHandler — log everything to console
#   2. BaseCallbackHandler — write your own callback
#   3. Key events: on_llm_start, on_llm_end, on_chain_start, on_tool_start
#   4. TimingAndCostCallback — measure latency and track spending
#   5. get_openai_callback() — deep dive into token tracking
#   6. Streaming: llm.stream() and chain.stream() for real-time output
#
# HOW THIS CONNECTS:
#   Previous: 25_agents_multi_tool.py — complete multi-tool agent
#   Next:     27_streaming_advanced.py — async streaming and astream_events
# =============================================================================

import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_community.callbacks import get_openai_callback

load_dotenv()

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
)

prompt = ChatPromptTemplate.from_template("Answer briefly: {question}")
parser = StrOutputParser()
chain = prompt | llm | parser

print("=" * 60)
print("  CHAPTER 26: Callbacks and Streaming")
print("=" * 60)

# =============================================================================
# SECTION 1: StdOutCallbackHandler — See EVERYTHING
# =============================================================================
# This is the simplest callback — it prints every event to stdout.
# Great for learning what events fire and in what order.

print("\n--- Section 1: StdOutCallbackHandler ---")

from langchain_core.callbacks import StdOutCallbackHandler

print("\nRunning chain with StdOutCallbackHandler:")
print("(Watch the events fire in order)")
print("-" * 40)

result = chain.invoke(
    {"question": "What is Python?"},
    config={"callbacks": [StdOutCallbackHandler()]}
)
print("-" * 40)
print(f"Final result: {result[:100]}...")

# =============================================================================
# SECTION 2: BUILDING A CUSTOM CALLBACK
# =============================================================================
# Subclass BaseCallbackHandler and override the events you care about.
# Each event method receives specific data about what happened.

print("\n--- Section 2: Custom BaseCallbackHandler ---")

class TimingAndCostCallback(BaseCallbackHandler):
    """
    Custom callback that tracks:
    - When each LLM call starts and ends
    - How long each call took (latency)
    - How many tokens were used
    - Total cost accumulated
    """

    def __init__(self):
        self.call_count = 0
        self.total_tokens = 0
        self.total_latency_ms = 0
        self._start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Fired when the LLM starts generating."""
        self.call_count += 1
        self._start_time = time.time()
        print(f"\n  [CALLBACK] LLM call #{self.call_count} started")
        print(f"  [CALLBACK] Input length: {len(prompts[0])} chars")

    def on_llm_end(self, response, **kwargs):
        """Fired when the LLM finishes generating."""
        latency_ms = (time.time() - self._start_time) * 1000
        self.total_latency_ms += latency_ms

        # Extract token usage from response
        token_usage = {}
        if response.llm_output and "token_usage" in response.llm_output:
            token_usage = response.llm_output["token_usage"]
            self.total_tokens += token_usage.get("total_tokens", 0)

        print(f"  [CALLBACK] LLM call #{self.call_count} finished in {latency_ms:.0f}ms")
        print(f"  [CALLBACK] Tokens this call: {token_usage.get('total_tokens', 'N/A')}")

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Fired when a chain starts executing."""
        chain_name = serialized.get("id", ["unknown"])[-1]
        print(f"\n  [CALLBACK] Chain started: {chain_name}")

    def on_chain_end(self, outputs, **kwargs):
        """Fired when a chain finishes executing."""
        print(f"  [CALLBACK] Chain finished")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Fired when a tool is called by an agent."""
        tool_name = serialized.get("name", "unknown")
        print(f"  [CALLBACK] Tool called: {tool_name}({input_str[:50]})")

    def on_tool_end(self, output, **kwargs):
        """Fired when a tool returns a result."""
        print(f"  [CALLBACK] Tool returned: {str(output)[:80]}...")

    def summary(self) -> str:
        avg_latency = self.total_latency_ms / self.call_count if self.call_count > 0 else 0
        return (
            f"\n  === Callback Summary ===\n"
            f"  LLM calls made   : {self.call_count}\n"
            f"  Total tokens used: {self.total_tokens}\n"
            f"  Total latency    : {self.total_latency_ms:.0f}ms\n"
            f"  Avg latency/call : {avg_latency:.0f}ms"
        )


# Use the custom callback
timing_callback = TimingAndCostCallback()

print("\nRunning 3 questions with TimingAndCostCallback:")
questions = [
    "What is machine learning?",
    "Name one planet in our solar system.",
    "What is 2 + 2?",
]

for q in questions:
    chain.invoke(
        {"question": q},
        config={"callbacks": [timing_callback]}
    )

print(timing_callback.summary())

# =============================================================================
# SECTION 3: get_openai_callback() — TOKEN TRACKING (DEEP DIVE)
# =============================================================================
# get_openai_callback() is a context manager that tracks ALL LLM calls
# made within its `with` block.
# It works across multiple chain invocations — perfect for session-level tracking.

print("\n--- Section 3: get_openai_callback() Deep Dive ---")

# Track costs across multiple invocations
print("\nTracking costs across 3 chain invocations:")

with get_openai_callback() as cb:
    r1 = chain.invoke({"question": "Explain photosynthesis in 10 words."})
    r2 = chain.invoke({"question": "What is the speed of light?"})
    r3 = chain.invoke({"question": "Who invented the telephone?"})

    print(f"\nAnswer 1: {r1}")
    print(f"Answer 2: {r2}")
    print(f"Answer 3: {r3}")

    print(f"\n  --- Token Usage Report ---")
    print(f"  Prompt tokens     : {cb.prompt_tokens}")
    print(f"  Completion tokens : {cb.completion_tokens}")
    print(f"  Total tokens      : {cb.total_tokens}")
    print(f"  Estimated cost    : ${cb.total_cost:.6f} USD")
    print(f"\n  (All 3 LLM calls tracked in ONE context manager)")

# =============================================================================
# SECTION 4: STREAMING — llm.stream()
# =============================================================================
# Instead of waiting for the full response, stream yields tokens one by one.
# Essential for good UX in chat applications.

print("\n--- Section 4: Basic Streaming ---")

print("\nllm.stream() — raw token streaming:")
print("AI: ", end="", flush=True)

for chunk in llm.stream("Write a 2-sentence description of the Python programming language."):
    # chunk is an AIMessageChunk — .content has the new tokens
    print(chunk.content, end="", flush=True)

print("\n")

# =============================================================================
# SECTION 5: chain.stream() — STREAM THROUGH THE WHOLE CHAIN
# =============================================================================
# chain.stream() passes the input through all chain steps.
# For prompt | llm | parser chains, you get the parser output streamed.

print("--- Section 5: chain.stream() ---")

print("\nchain.stream() — streaming through prompt | llm | parser:")
print("AI: ", end="", flush=True)

for token in chain.stream({"question": "Explain what an API is, step by step in 3 steps."}):
    # token is now a plain string (because of StrOutputParser)
    print(token, end="", flush=True)

print("\n")

# =============================================================================
# SECTION 6: CALLBACKS + STREAMING TOGETHER
# =============================================================================
# You can combine callbacks and streaming for real-time monitoring.

print("--- Section 6: Callbacks + Streaming Together ---")

class StreamMonitor(BaseCallbackHandler):
    """Monitor streaming performance."""
    def __init__(self):
        self.chunk_count = 0
        self.total_chars = 0
        self.start_time = None

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        self.chunk_count = 0
        self.total_chars = 0
        print("\n  [Monitor] Streaming started...")

    def on_llm_new_token(self, token: str, **kwargs):
        """Fired for each new streaming token."""
        self.chunk_count += 1
        self.total_chars += len(token)

    def on_llm_end(self, response, **kwargs):
        elapsed = time.time() - self.start_time
        tps = self.chunk_count / elapsed if elapsed > 0 else 0
        print(f"\n  [Monitor] Streaming done: {self.chunk_count} chunks, "
              f"{self.total_chars} chars, {tps:.1f} tokens/sec")

monitor = StreamMonitor()

# Create streaming LLM with the monitor callback
streaming_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    streaming=True,
    callbacks=[monitor],  # Attach callback at construction time
)

streaming_chain = prompt | streaming_llm | parser

print("\nRunning streaming chain with performance monitor:")
print("AI: ", end="", flush=True)

for token in streaming_chain.stream({"question": "What are 3 benefits of Python?"}):
    print(token, end="", flush=True)

print()

# =============================================================================
# SECTION 7: CALLBACK ATTACHMENT METHODS
# =============================================================================
print("\n--- Section 7: Three Ways to Attach Callbacks ---")
print("""
  METHOD 1: Per-invocation (most flexible)
    result = chain.invoke(input, config={"callbacks": [my_callback]})
    → Only applies to this one call

  METHOD 2: Chain construction (permanent for this chain)
    chain = (prompt | llm | parser).with_config({"callbacks": [my_callback]})
    → Every invoke() of this chain uses the callback

  METHOD 3: LLM constructor (global to all uses of this LLM)
    llm = ChatOpenAI(..., callbacks=[my_callback])
    → Every chain that uses this LLM fires the callback

  RECOMMENDATION:
  → Use METHOD 1 for request-level logging (e.g., user_id tracking)
  → Use METHOD 2 for chain-level monitoring (e.g., production chains)
  → Use METHOD 3 for global cost tracking across all calls
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 26")
print("=" * 60)
print("""
  1. Callbacks: event hooks that fire at each stage (chain, LLM, tool)
  2. StdOutCallbackHandler: prints all events — great for learning
  3. BaseCallbackHandler: subclass and override on_llm_start, on_llm_end, etc.
  4. get_openai_callback(): context manager for session-level token tracking
  5. llm.stream(): yields AIMessageChunk tokens in real time
  6. chain.stream(): streams through the whole pipeline
  7. streaming=True + callbacks=[monitor] = real-time performance monitoring

  Next up: 27_streaming_advanced.py
  Async streaming, astream_events(), and building real-time chat UIs.
""")
