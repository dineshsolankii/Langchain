# =============================================================================
# FILE: 14_text_splitters.py
# PART: 4 - Documents & Embeddings  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   Imagine trying to find a specific sentence in a 500-page book
#   that was photocopied as one giant image.
#   You can't search it. You can't index it. You're stuck.
#
#   Splitting is how you break that into searchable index cards.
#   Each card (chunk) is small enough to search precisely,
#   but large enough to contain meaningful context.
#
#   The art is in choosing the right card size — too small and
#   you lose context; too large and retrieval is imprecise.
#
# WHAT YOU WILL LEARN:
#   1. Why splitting is necessary (context windows + retrieval precision)
#   2. RecursiveCharacterTextSplitter — the go-to choice (explained in depth)
#   3. chunk_size and chunk_overlap — the key parameters visualized
#   4. TokenTextSplitter — split by tokens instead of characters
#   5. MarkdownHeaderTextSplitter — semantic splitting by document structure
#   6. Chunk size experiment — 100 vs 1000, see the difference
#
# HOW THIS CONNECTS:
#   Previous: 13_document_loaders.py — loading documents from any source
#   Next:     15_embeddings.py — turning text chunks into numeric vectors
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
)

load_dotenv()

print("=" * 60)
print("  CHAPTER 14: Text Splitters")
print("=" * 60)

# Sample document — a longer piece of text to split
SAMPLE_TEXT = """
LangChain is a powerful framework designed to simplify the development of applications
powered by large language models (LLMs). It provides a set of tools and abstractions
that make it easy to build complex AI workflows.

One of LangChain's core features is the LangChain Expression Language (LCEL). LCEL
uses a pipe operator (|) to chain components together. This makes it easy to compose
prompts, LLMs, and output parsers into a single pipeline.

Memory is another critical component. LangChain provides InMemoryChatMessageHistory
for storing conversation history within a single session. For persistence across
sessions, FileChatMessageHistory or database-backed solutions can be used.

The Retrieval-Augmented Generation (RAG) pattern is one of LangChain's most powerful
use cases. RAG allows LLMs to access external knowledge bases, making their answers
more accurate and up-to-date. The pipeline involves loading documents, splitting them
into chunks, creating embeddings, storing in a vector database, and retrieving
relevant chunks to answer questions.

Agents are the most advanced LangChain concept. An agent uses an LLM as a reasoning
engine to decide which tools to use and in what order. Tools can be anything from
web search to database queries to custom Python functions.

LangGraph extends LangChain with graph-based workflows. Unlike linear chains,
LangGraph supports cycles, branching, and stateful multi-agent systems. It is the
recommended approach for building production-grade AI agents.
"""

# =============================================================================
# SECTION 1: WHY SPLITTING IS NECESSARY
# =============================================================================
print("\n--- Section 1: Why Splitting is Necessary ---")
print(f"\nSample document: {len(SAMPLE_TEXT)} characters")
print("""
  PROBLEM 1 — Context Window Limits:
    Most LLMs have a context limit (e.g., 128k tokens for GPT-4o).
    A 10MB PDF would exceed this limit — you can't fit it all in one call.

  PROBLEM 2 — Retrieval Precision:
    If you embed the entire document as one chunk, similarity search finds
    the whole document — not the specific paragraph that answers the question.
    Smaller chunks = more precise retrieval.

  SOLUTION — Split into chunks:
    Split the document into overlapping chunks.
    Embed each chunk. Store in vector DB.
    At query time: find the 3-5 most relevant chunks.
    Feed ONLY those chunks to the LLM.
""")

# =============================================================================
# SECTION 2: RecursiveCharacterTextSplitter — The Default Choice
# =============================================================================
# This splitter tries to split on natural boundaries:
#   1st try: split on "\n\n" (paragraphs)
#   2nd try: split on "\n" (lines)
#   3rd try: split on ". " (sentences)
#   4th try: split on " " (words)
#   5th try: split character by character (last resort)
#
# It uses the FIRST separator that creates chunks small enough.
# This respects document structure as much as possible.

print("--- Section 2: RecursiveCharacterTextSplitter ---")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,        # Maximum characters per chunk
    chunk_overlap=80,      # Characters shared between consecutive chunks
    length_function=len,   # How to measure chunk size (characters here)
    separators=["\n\n", "\n", ". ", " ", ""],  # Try these in order
)

chunks = splitter.create_documents([SAMPLE_TEXT])

print(f"\nOriginal document: {len(SAMPLE_TEXT)} characters")
print(f"Chunks created   : {len(chunks)}")
print(f"Avg chunk size   : {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")

print(f"\nFirst chunk (index 0):")
print(f"  Length : {len(chunks[0].page_content)} chars")
print(f"  Content: {chunks[0].page_content[:200]}...")

print(f"\nSecond chunk (index 1):")
print(f"  Length : {len(chunks[1].page_content)} chars")
print(f"  Content: {chunks[1].page_content[:200]}...")

# =============================================================================
# SECTION 3: VISUALIZING chunk_overlap
# =============================================================================
# Chunk overlap ensures that information at chunk boundaries isn't lost.
# Chunk N and Chunk N+1 share `chunk_overlap` characters.
# This prevents a sentence from being cut in half between two chunks.

print("\n--- Section 3: Visualizing chunk_overlap ---")

# Use a simple splitter with visible overlap
small_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

small_chunks = small_splitter.create_documents([SAMPLE_TEXT])

if len(small_chunks) >= 2:
    chunk0_end = small_chunks[0].page_content[-50:]    # Last 50 chars of chunk 0
    chunk1_start = small_chunks[1].page_content[:50]   # First 50 chars of chunk 1

    print(f"\nChunk 0 ends with  : '...{chunk0_end}'")
    print(f"Chunk 1 starts with: '{chunk1_start}...'")

    # Find the overlap
    overlap_text = ""
    for i in range(50, 0, -1):
        if small_chunks[0].page_content.endswith(small_chunks[1].page_content[:i]):
            overlap_text = small_chunks[1].page_content[:i]
            break

    if overlap_text:
        print(f"\nOverlapping text   : '{overlap_text}'")
        print(f"(This text appears in BOTH chunk 0 and chunk 1)")

# =============================================================================
# SECTION 4: CHUNK SIZE EXPERIMENT
# =============================================================================
# See how different chunk_size values affect the number and size of chunks.

print("\n--- Section 4: Chunk Size Experiment ---")

for size in [100, 300, 500, 1000]:
    exp_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=20)
    exp_chunks = exp_splitter.create_documents([SAMPLE_TEXT])
    avg_len = sum(len(c.page_content) for c in exp_chunks) // len(exp_chunks)
    print(f"  chunk_size={size:4}: {len(exp_chunks):2} chunks, avg {avg_len:3} chars/chunk")

print("\nInsight: Smaller chunks = more precise retrieval but less context per chunk")
print("Typical production choice: chunk_size=500-1000, chunk_overlap=50-200")

# =============================================================================
# SECTION 5: SPLITTING DOCUMENTS (WITH METADATA)
# =============================================================================
# When you split Documents (not plain text), metadata is INHERITED.
# Each chunk gets the original document's metadata, plus a chunk index.

print("\n--- Section 5: Splitting Documents with Metadata ---")

from langchain_community.document_loaders import TextLoader

try:
    raw_docs = TextLoader("Langchain/data/knowledge.txt").load()
except FileNotFoundError:
    from langchain_core.documents import Document
    raw_docs = [Document(
        page_content=SAMPLE_TEXT,
        metadata={"source": "Langchain/data/knowledge.txt"}
    )]

doc_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
split_docs = doc_splitter.split_documents(raw_docs)

print(f"\nOriginal documents : {len(raw_docs)}")
print(f"After splitting    : {len(split_docs)} chunks")
print(f"\nChunk 0 metadata   : {split_docs[0].metadata}")
print(f"Chunk 0 content    : {split_docs[0].page_content[:100]}...")
# Note: metadata is inherited from the original document!

# =============================================================================
# SECTION 6: MarkdownHeaderTextSplitter — Semantic Splitting
# =============================================================================
# Instead of splitting by character count, split at Markdown headers.
# Each chunk respects the document's logical structure.
# Metadata contains the section headers — great for hierarchical retrieval.

print("\n--- Section 6: MarkdownHeaderTextSplitter ---")

markdown_text = """# LangChain Overview

LangChain is a framework for building LLM applications.
It provides tools for prompt management, chain composition, and memory.

## Core Components

### LCEL (LangChain Expression Language)
LCEL uses the pipe operator to chain Runnables together.
It supports streaming, batching, and async execution.

### Memory
Memory allows LLMs to remember past conversations.
Options include InMemory, File-based, and Database-backed storage.

## Advanced Features

### RAG (Retrieval-Augmented Generation)
RAG connects LLMs to external knowledge bases.
It improves accuracy and reduces hallucinations.

### Agents
Agents use LLMs to decide which tools to call and when.
"""

headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,  # Keep the headers in the content
)

md_chunks = md_splitter.split_text(markdown_text)

print(f"\nMarkdown split into {len(md_chunks)} semantic chunks")
for i, chunk in enumerate(md_chunks):
    print(f"\nChunk {i+1}:")
    print(f"  Metadata: {chunk.metadata}")
    print(f"  Content : {chunk.page_content[:100]}...")

# =============================================================================
# SECTION 7: TokenTextSplitter — Split by Token Count
# =============================================================================
print("\n--- Section 7: TokenTextSplitter ---")
print("""
  # Usage:
  from langchain_text_splitters import TokenTextSplitter

  token_splitter = TokenTextSplitter(
      chunk_size=100,      # Max tokens per chunk (not characters!)
      chunk_overlap=20,    # Tokens of overlap
  )
  token_chunks = token_splitter.create_documents([long_text])

  # Why use this?
  # LLM pricing is per TOKEN, not per character.
  # A token is roughly 4 characters in English.
  # Use TokenTextSplitter when you need to respect exact token budgets.
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 14")
print("=" * 60)
print("""
  1. Splitting is necessary because LLMs have context limits and retrieval needs precision
  2. RecursiveCharacterTextSplitter is the default — respects paragraph/line boundaries
  3. chunk_size: max characters per chunk; chunk_overlap: shared chars between chunks
  4. Overlap prevents information loss at chunk boundaries
  5. .split_documents() inherits metadata from original Documents
  6. MarkdownHeaderTextSplitter splits semantically at headers — metadata has section info
  7. Rule of thumb: chunk_size=500-1000, chunk_overlap=10-20% of chunk_size

  Next up: 15_embeddings.py
  Converting text chunks into numeric vectors for semantic search.
""")
