# =============================================================================
# FILE: 13_document_loaders.py
# PART: 4 - Documents & Embeddings  |  LEVEL: Intermediate
# =============================================================================
#
# THE STORY:
#   LangChain can read almost any document format you throw at it.
#   Text files. PDFs. Spreadsheets. Web pages. Entire folders.
#
#   Each loader is a specialist reader. They look different on the outside
#   but they all hand you back the same simple object: a Document.
#
#   The Document has two fields:
#     page_content : The actual text (what the LLM reads)
#     metadata     : Source, page number, author, etc. (context for YOU)
#
# WHAT YOU WILL LEARN:
#   1. The Document object — page_content + metadata
#   2. TextLoader — plain .txt files
#   3. PyPDFLoader — PDF documents (each page = one Document)
#   4. CSVLoader — CSV rows as individual Documents
#   5. JSONLoader — JSON with jq schema
#   6. WebBaseLoader — scrape any webpage
#   7. DirectoryLoader — load an entire folder
#   8. Combining documents from multiple sources
#
# HOW THIS CONNECTS:
#   Previous: 12_memory_persistent.py — persistent conversation history
#   Next:     14_text_splitters.py — splitting documents into chunks
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

print("=" * 60)
print("  CHAPTER 13: Document Loaders")
print("=" * 60)

# =============================================================================
# SECTION 1: THE Document OBJECT — THE UNIVERSAL UNIT
# =============================================================================
# Everything in LangChain's document pipeline works with Document objects.
# A Document is simply: (text_content, metadata_dict)
# You can create them manually or use loaders to create them from files.

print("\n--- Section 1: The Document Object ---")

# You can create Documents manually
manual_doc = Document(
    page_content="LangChain is a framework for building LLM-powered applications.",
    metadata={"source": "manual", "author": "Dinesh", "topic": "AI"}
)

print(f"\nManual Document:")
print(f"  page_content : {manual_doc.page_content}")
print(f"  metadata     : {manual_doc.metadata}")

# =============================================================================
# SECTION 2: TextLoader — The Simplest Loader
# =============================================================================
# TextLoader reads a plain text file and returns ONE Document.
# The whole file becomes page_content. Metadata has the source path.

print("\n--- Section 2: TextLoader ---")

from langchain_community.document_loaders import TextLoader

# Load our knowledge base file
try:
    loader = TextLoader("Langchain/data/knowledge.txt", encoding="utf-8")
    docs = loader.load()

    print(f"\nLoaded {len(docs)} document(s) from knowledge.txt")
    print(f"\nDocument content: {docs[0].page_content}")
    print(f"Metadata         : {docs[0].metadata}")
    print(f"Content length   : {len(docs[0].page_content)} characters")
except FileNotFoundError:
    print("\nNote: data/knowledge.txt not found. Using a manual document instead.")
    docs = [Document(
        page_content="Dinesh is learning LangChain to build AI agents.\nHe works with n8n and automation.",
        metadata={"source": "Langchain/data/knowledge.txt"}
    )]
    print(f"Using fallback: {docs[0].page_content}")

# =============================================================================
# SECTION 3: PyPDFLoader — PDF Documents
# =============================================================================
# PyPDFLoader splits a PDF into one Document PER PAGE.
# Each Document's metadata includes: source, page number.
# Install: pip install pypdf

print("\n--- Section 3: PyPDFLoader (PDF Files) ---")
print("""
  # Usage:
  from langchain_community.document_loaders import PyPDFLoader

  loader = PyPDFLoader("data/sample.pdf")
  docs = loader.load()

  print(f"PDF has {{len(docs)}} pages")
  for doc in docs:
      print(f"  Page {{doc.metadata['page']}}: {{doc.page_content[:100]}}...")

  # Each page becomes one Document:
  # doc.page_content = text of that page
  # doc.metadata = {'source': 'data/sample.pdf', 'page': 0}
""")

# =============================================================================
# SECTION 4: CSVLoader — CSV Files
# =============================================================================
# CSVLoader turns each ROW of a CSV file into one Document.
# All columns are included in page_content as "key: value" pairs.
# Perfect for Q&A over tabular data.

print("--- Section 4: CSVLoader (CSV Files) ---")

# Create a sample CSV to demonstrate
sample_csv_path = "Langchain/data/sample.csv"
os.makedirs("Langchain/data", exist_ok=True)

with open(sample_csv_path, "w") as f:
    f.write("name,role,company,skill\n")
    f.write("Dinesh,AI Engineer,Freelance,LangChain\n")
    f.write("Alice,Data Scientist,TechCorp,Python\n")
    f.write("Bob,ML Engineer,StartupAI,PyTorch\n")

from langchain_community.document_loaders import CSVLoader

csv_loader = CSVLoader(file_path=sample_csv_path)
csv_docs = csv_loader.load()

print(f"\nLoaded {len(csv_docs)} documents from CSV (one per row)")
print(f"\nFirst document:")
print(f"  page_content: {csv_docs[0].page_content}")
print(f"  metadata    : {csv_docs[0].metadata}")

# =============================================================================
# SECTION 5: JSONLoader — JSON Files
# =============================================================================
# JSONLoader uses jq syntax to select which parts of JSON to load.
# Each selected item becomes one Document.
# Install: pip install jq

print("\n--- Section 5: JSONLoader (JSON Files) ---")

# Create a sample JSON file
sample_json_path = "Langchain/data/sample.json"
import json as json_module

sample_data = [
    {"id": 1, "topic": "LangChain", "description": "Framework for LLM apps"},
    {"id": 2, "topic": "FAISS", "description": "Fast local vector store"},
    {"id": 3, "topic": "RAG", "description": "Retrieval Augmented Generation"},
]

with open(sample_json_path, "w") as f:
    json_module.dump(sample_data, f, indent=2)

try:
    from langchain_community.document_loaders import JSONLoader

    json_loader = JSONLoader(
        file_path=sample_json_path,
        jq_schema=".[]",           # Select each item in the array
        text_content=False,        # Convert the whole item to text (not just one field)
    )
    json_docs = json_loader.load()

    print(f"\nLoaded {len(json_docs)} documents from JSON")
    for doc in json_docs:
        print(f"  Content: {doc.page_content[:80]}")
        print(f"  Source : {doc.metadata.get('source')}")

except (ImportError, Exception) as e:
    print(f"\nNote: JSONLoader requires 'jq' package. Install: pip install jq")
    print(f"Error: {e}")
    # Manual fallback
    print("Creating Documents manually from JSON:")
    json_docs = [
        Document(page_content=str(item), metadata={"source": sample_json_path, "id": item["id"]})
        for item in sample_data
    ]
    for doc in json_docs:
        print(f"  {doc.page_content}")

# =============================================================================
# SECTION 6: WebBaseLoader — Scrape a Webpage
# =============================================================================
# WebBaseLoader fetches a webpage and extracts its text content.
# Uses BeautifulSoup4 under the hood.
# Install: pip install beautifulsoup4 requests

print("\n--- Section 6: WebBaseLoader (Webpage Scraping) ---")
print("""
  # Usage:
  from langchain_community.document_loaders import WebBaseLoader

  loader = WebBaseLoader("https://python.langchain.com/docs/introduction/")
  docs = loader.load()

  print(f"Scraped {{len(docs[0].page_content)}} characters")
  print(docs[0].page_content[:500])
  print(docs[0].metadata)  # {'source': 'https://...', 'title': '...'}

  # Load multiple URLs at once:
  loader = WebBaseLoader([
      "https://python.langchain.com/docs/introduction/",
      "https://python.langchain.com/docs/concepts/",
  ])
  docs = loader.load()  # One Document per URL
""")

# =============================================================================
# SECTION 7: DirectoryLoader — Load an Entire Folder
# =============================================================================
# DirectoryLoader loads all files matching a glob pattern from a directory.
# It dispatches each file to the right loader class automatically.

print("--- Section 7: DirectoryLoader (Bulk Load a Folder) ---")

from langchain_community.document_loaders import DirectoryLoader

try:
    # Load all .csv and .json files from our data folder
    dir_loader = DirectoryLoader(
        "Langchain/data/",
        glob="**/*.csv",       # Pattern: all CSV files recursively
        loader_cls=CSVLoader,  # Use CSVLoader for each file
    )
    dir_docs = dir_loader.load()
    print(f"\nDirectoryLoader found {len(dir_docs)} documents from all CSV files in data/")
except Exception as e:
    print(f"\nDirectoryLoader example: {e}")

# =============================================================================
# SECTION 8: COMBINING DOCUMENTS FROM MULTIPLE SOURCES
# =============================================================================
# In real RAG systems, your knowledge base has many sources.
# Just concatenate the lists!

print("\n--- Section 8: Combining Multiple Sources ---")

# Combine all our loaded documents
all_docs = docs + csv_docs + json_docs  # List concatenation

print(f"\nCombined knowledge base:")
print(f"  TextLoader  docs: {len(docs)}")
print(f"  CSVLoader   docs: {len(csv_docs)}")
print(f"  JSONLoader  docs: {len(json_docs)}")
print(f"  TOTAL           : {len(all_docs)}")

# Inspect source diversity
print("\nSources in combined knowledge base:")
sources = set(doc.metadata.get("source", "unknown") for doc in all_docs)
for source in sources:
    print(f"  {source}")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 13")
print("=" * 60)
print("""
  1. Document = (page_content, metadata) — the universal unit
  2. TextLoader: whole file → 1 Document
  3. PyPDFLoader: each page → 1 Document (with page number in metadata)
  4. CSVLoader: each row → 1 Document (all columns in page_content)
  5. JSONLoader: uses jq syntax to select items from JSON
  6. WebBaseLoader: fetches and parses webpage HTML → Document(s)
  7. DirectoryLoader: bulk-loads all matching files from a folder
  8. Combine sources: all_docs = docs1 + docs2 + docs3

  Next up: 14_text_splitters.py
  Splitting large documents into small, searchable chunks.
""")
