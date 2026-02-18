# =============================================================================
# FILE: 31_async_and_batching.py
# PART: 9 - Production  |  LEVEL: Expert
# =============================================================================
#
# THE STORY:
#   You built a great LangChain app. Users love it.
#   Then 10 users hit it at once. Then 100. Then 1000.
#   Suddenly your sequential llm.invoke() loop takes 5 minutes
#   because it processes users one by one.
#
#   Production apps don't wait. They batch. They go async.
#   They handle 100 requests in the time it used to take to handle 10.
#   This chapter shows you how to scale.
#
# WHAT YOU WILL LEARN:
#   1. Sequential vs batch() timing comparison
#   2. chain.batch() with max_concurrency control
#   3. asyncio + astream() for non-blocking I/O
#   4. asyncio.gather() for true parallel execution
#   5. Semaphore-based rate limiting for API throttle
#   6. When to use each pattern
#
# HOW THIS CONNECTS:
#   Previous: 30_langgraph_multi_agent.py — multi-agent pipelines
#   Next:     32_caching_and_optimization.py — speed up with caching
# =============================================================================

import os
import time
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

print("=" * 60)
print("  CHAPTER 31: Async and Batching")
print("=" * 60)

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.0,
)

prompt = ChatPromptTemplate.from_template(
    "Give a one-sentence definition of: {concept}"
)
chain = prompt | llm | StrOutputParser()

CONCEPTS = [
    "machine learning",
    "neural networks",
    "gradient descent",
    "backpropagation",
    "transformer architecture",
]

# =============================================================================
# SECTION 1: SEQUENTIAL — THE SLOW WAY
# =============================================================================
# Each call blocks until complete before the next starts.
# Total time ≈ sum of all individual call times.

print("\n--- Section 1: Sequential Execution ---")
print(f"Processing {len(CONCEPTS)} concepts one by one...")

start = time.time()
sequential_results = []
for concept in CONCEPTS:
    result = chain.invoke({"concept": concept})
    sequential_results.append(result)
    print(f"  ✓ {concept}")

sequential_time = time.time() - start
print(f"\nSequential time: {sequential_time:.2f}s")
print(f"Average per call: {sequential_time / len(CONCEPTS):.2f}s")

# =============================================================================
# SECTION 2: chain.batch() — THE FAST WAY
# =============================================================================
# batch() sends multiple requests and collects all results.
# max_concurrency limits how many run simultaneously (respect API rate limits).

print("\n--- Section 2: chain.batch() ---")
print(f"Processing {len(CONCEPTS)} concepts in parallel...")

start = time.time()
batch_results = chain.batch(
    [{"concept": c} for c in CONCEPTS],
    config={"max_concurrency": 5},  # Up to 5 concurrent requests
)
batch_time = time.time() - start

print(f"\nBatch time: {batch_time:.2f}s")
print(f"Speedup: {sequential_time / batch_time:.1f}x faster than sequential")

for concept, result in zip(CONCEPTS, batch_results):
    print(f"  {concept}: {result[:60]}...")

# =============================================================================
# SECTION 3: ASYNC WITH astream()
# =============================================================================
# astream() is non-blocking — the event loop can handle other tasks
# while waiting for the LLM to generate tokens.

print("\n--- Section 3: Async astream() ---")

async def async_stream_concept(concept: str) -> str:
    """Stream a single concept definition asynchronously."""
    result = ""
    async for token in chain.astream({"concept": concept}):
        result += token
    return result

async def async_sequential():
    """Process concepts one by one, but async (non-blocking)."""
    print("\nAsync sequential (awaiting each one):")
    start = time.time()
    results = []
    for concept in CONCEPTS:
        result = await async_stream_concept(concept)
        results.append(result)
        print(f"  ✓ {concept}")
    elapsed = time.time() - start
    print(f"  Async sequential time: {elapsed:.2f}s")
    return results

asyncio.run(async_sequential())

# =============================================================================
# SECTION 4: asyncio.gather() — TRUE PARALLELISM
# =============================================================================
# asyncio.gather() launches ALL coroutines at the same time.
# They run concurrently — total time ≈ slowest individual call, not sum.
# This is the gold standard for LLM-intensive web applications.

print("\n--- Section 4: asyncio.gather() — Concurrent Async ---")

async def concurrent_async():
    """Run all concept lookups concurrently with asyncio.gather()."""
    print(f"\nLaunching all {len(CONCEPTS)} requests simultaneously...")
    start = time.time()

    # All coroutines start at the same time
    results = await asyncio.gather(
        *[async_stream_concept(c) for c in CONCEPTS]
    )

    elapsed = time.time() - start
    print(f"asyncio.gather() time: {elapsed:.2f}s")
    print(f"Speedup vs sequential: {sequential_time / elapsed:.1f}x")

    for concept, result in zip(CONCEPTS, results):
        print(f"  {concept}: {result[:60]}...")

    return results

asyncio.run(concurrent_async())

# =============================================================================
# SECTION 5: RATE-LIMITED CONCURRENT REQUESTS
# =============================================================================
# asyncio.Semaphore limits how many requests run at once.
# Critical for respecting API rate limits (OpenRouter, OpenAI both have limits).
# Without this, 1000 concurrent requests = rate limit error.

print("\n--- Section 5: Semaphore Rate Limiting ---")

MORE_CONCEPTS = [
    "overfitting", "underfitting", "regularization",
    "cross-validation", "feature engineering",
    "dimensionality reduction", "transfer learning",
    "reinforcement learning", "attention mechanism",
    "tokenization",
]

async def rate_limited_batch(concepts: list[str], max_concurrent: int = 3) -> list[str]:
    """
    Process concepts with a semaphore to limit concurrency.
    max_concurrent=3 means at most 3 API calls at once.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_call(concept: str) -> str:
        async with semaphore:  # Blocks if max_concurrent slots are full
            result = await async_stream_concept(concept)
            print(f"  ✓ {concept}")
            return result

    print(f"\nProcessing {len(concepts)} concepts with max_concurrent={max_concurrent}...")
    start = time.time()
    results = await asyncio.gather(*[limited_call(c) for c in concepts])
    elapsed = time.time() - start
    print(f"Rate-limited batch time: {elapsed:.2f}s for {len(concepts)} calls")
    return results

asyncio.run(rate_limited_batch(MORE_CONCEPTS, max_concurrent=3))

# =============================================================================
# SECTION 6: BATCH WITH ERROR HANDLING
# =============================================================================
# In production, some calls fail. batch() can handle this gracefully.

print("\n--- Section 6: Batch with Error Handling ---")

print("""
  For error-resilient batch processing:

  results = chain.batch(
      inputs,
      config={"max_concurrency": 5},
      return_exceptions=True,   # Don't crash on one failure
  )

  for i, result in enumerate(results):
      if isinstance(result, Exception):
          print(f"Input {i} failed: {result}")
      else:
          process(result)

  This lets partial failures not block successful ones.
""")

# =============================================================================
# SECTION 7: CHOOSING THE RIGHT PATTERN
# =============================================================================
print("--- Section 7: Choosing the Right Pattern ---")
print("""
  ┌──────────────────────────┬────────────────────────────────────────┐
  │ Pattern                  │ Use When                               │
  ├──────────────────────────┼────────────────────────────────────────┤
  │ chain.invoke()           │ Single call, CLI scripts               │
  │ sequential loop          │ Debugging, development                 │
  │ chain.batch()            │ Known list of inputs, simple parallel  │
  │ asyncio.gather()         │ Web apps, max throughput needed        │
  │ Semaphore + gather()     │ API rate limits must be respected      │
  │ chain.astream()          │ Streaming UI, real-time display        │
  └──────────────────────────┴────────────────────────────────────────┘

  GOLDEN RULES:
  → Development/debugging   → invoke() (simple, visible)
  → Batch processing        → chain.batch(max_concurrency=5)
  → Web application         → asyncio.gather() + Semaphore
  → Real-time streaming UI  → astream() or astream_events()
  → Rate-limited APIs       → asyncio.Semaphore(3-5)
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 31")
print("=" * 60)
print(f"""
  1. Sequential: simple but slow — total time = sum of all calls
  2. chain.batch(): easiest speedup — built-in, max_concurrency controls parallelism
  3. astream(): async streaming — non-blocking, great for web handlers
  4. asyncio.gather(): max throughput — all requests start simultaneously
  5. asyncio.Semaphore(N): rate limiting — at most N concurrent requests
  6. return_exceptions=True: resilient batching — partial failures don't crash

  TIMING SUMMARY (approximate, depends on network/API):
  Sequential (~{sequential_time:.1f}s) >> batch() >> asyncio.gather()

  Next up: 32_caching_and_optimization.py
  Make repeated calls instant with caching — LLM and embeddings.
""")
