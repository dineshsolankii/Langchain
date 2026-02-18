# =============================================================================
# FILE: 20_rag_conversational.py
# PART: 5 - RAG  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   The user asks: "What did he say about that?"
#   "He" and "that" refer to the previous message.
#   Basic RAG would search for "he" and "that" — useless.
#
#   Conversational RAG reads the history, rephrases the follow-up question
#   to be standalone: "What did Einstein say about quantum mechanics?"
#   THEN retrieves and answers. This closes the loop between memory and RAG.
#
# WHAT YOU WILL LEARN:
#   1. The history-aware retriever — rephrase queries using conversation context
#   2. create_history_aware_retriever() — the LCEL helper
#   3. create_stuff_documents_chain() — format retrieved docs into a prompt
#   4. create_retrieval_chain() — combine retrieval + QA into one chain
#   5. Full conversational RAG with source attribution
#
# HOW THIS CONNECTS:
#   Previous: 19_rag_advanced_retrievers.py — advanced retrieval strategies
#   Next:     21_tools_builtin.py — giving the LLM tools to use
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

print("=" * 60)
print("  CHAPTER 20: Conversational RAG")
print("=" * 60)
print("""
  Combines RAG + Memory:
  User asks a follow-up → History-aware retriever rephrases it
  → Retrieves relevant docs → LLM answers with context
""")

# =============================================================================
# SETUP: Knowledge base and models
# =============================================================================
# A comprehensive LangChain knowledge base
knowledge_docs = [
    Document(page_content="LCEL (LangChain Expression Language) uses the pipe | operator to chain Runnables. It supports streaming, batching, and async execution natively.", metadata={"source": "lcel_guide"}),
    Document(page_content="RunnablePassthrough passes the input unchanged. RunnableLambda wraps a Python function. RunnableParallel runs multiple chains on the same input simultaneously.", metadata={"source": "lcel_guide"}),
    Document(page_content="LangChain memory stores conversation history. InMemoryChatMessageHistory stores in RAM. FileChatMessageHistory stores to a JSON file on disk.", metadata={"source": "memory_guide"}),
    Document(page_content="RunnableWithMessageHistory wraps any chain to automatically manage reading and writing conversation history.", metadata={"source": "memory_guide"}),
    Document(page_content="RAG (Retrieval-Augmented Generation) improves LLM accuracy by retrieving relevant documents before answering. It prevents hallucination by grounding answers in retrieved facts.", metadata={"source": "rag_guide"}),
    Document(page_content="FAISS is an in-memory vector store. similarity_search() returns top-k Documents. as_retriever() converts it to a Retriever for use in chains.", metadata={"source": "vector_store_guide"}),
    Document(page_content="LangChain agents use tools. Tools have a name, description, and function. The @tool decorator creates a tool from any Python function.", metadata={"source": "agents_guide"}),
    Document(page_content="LangGraph builds graph-based workflows with StateGraph. Nodes are functions. Edges define the flow. add_conditional_edges() enables branching.", metadata={"source": "langgraph_guide"}),
    Document(page_content="Output parsers transform LLM text output into Python objects. StrOutputParser returns a string. PydanticOutputParser returns a typed Pydantic model.", metadata={"source": "parsers_guide"}),
    Document(page_content="Streaming: llm.stream() yields tokens one by one. chain.stream() streams through entire chains. astream_events() provides fine-grained async streaming.", metadata={"source": "streaming_guide"}),
]

print(f"\nKnowledge base: {len(knowledge_docs)} documents")

embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

vectorstore = FAISS.from_documents(knowledge_docs, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

# =============================================================================
# SECTION 1: THE CONTEXTUALIZATION PROMPT
# =============================================================================
# This prompt rephrases the follow-up question based on chat history.
# Goal: Convert "What about that?" → "What is LCEL used for?"
# The LLM reads the history and figures out what "that" refers to.

print("\n--- Section 1: History-Aware Retriever ---")

contextualize_q_system = (
    "Given a chat history and the latest user question, "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system),
    MessagesPlaceholder("chat_history"),     # The conversation so far
    ("human", "{input}"),                   # The latest (possibly ambiguous) question
])

# create_history_aware_retriever combines:
# 1. The contextualization prompt
# 2. The LLM (to rephrase the question)
# 3. The base retriever (to retrieve with rephrased question)
history_aware_retriever = create_history_aware_retriever(
    llm,
    base_retriever,
    contextualize_q_prompt,
)

print("  History-aware retriever ready.")
print("  It rephrases ambiguous follow-up questions using conversation history.")

# =============================================================================
# SECTION 2: THE QA CHAIN
# =============================================================================
# The QA chain takes the retrieved documents and answers the question.
# {context} is filled with the retrieved document text.
# chat_history is included so the model can maintain conversational tone.

print("\n--- Section 2: QA Chain with Context ---")

qa_system_prompt = (
    "You are an expert assistant for LangChain questions. "
    "Use the following retrieved context to answer the question. "
    "If you don't know the answer from the context, say 'I don't have this information.' "
    "Keep answers concise (2-3 sentences).\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),   # Keep conversational context
    ("human", "{input}"),
])

# create_stuff_documents_chain: formats retrieved docs into {context}
# "stuff" means: put all docs into the prompt (vs. map-reduce for very large sets)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

print("  QA chain ready.")
print("  It 'stuffs' all retrieved docs into the {context} variable.")

# =============================================================================
# SECTION 3: THE FULL CONVERSATIONAL RAG CHAIN
# =============================================================================
# create_retrieval_chain combines:
# 1. history_aware_retriever → retrieves relevant docs (with rephrasing)
# 2. question_answer_chain → answers using retrieved docs
#
# The chain output is a dict:
# {
#   "input": "user's question",
#   "chat_history": [...messages...],
#   "context": [...retrieved_documents...],
#   "answer": "the AI's answer"
# }

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

print("\n--- Section 3: Full Conversational RAG Chain ---")
print("  Conversational RAG chain assembled:")
print("  query → rephrase with history → retrieve → answer with context")

# =============================================================================
# SECTION 4: DEMO CONVERSATION
# =============================================================================
# Watch how the chain handles follow-up questions using history.

print("\n--- Section 4: Demo Conversation ---")

chat_history = []  # Starts empty; accumulates as we talk

def chat_with_rag(question: str, history: list) -> dict:
    """
    Ask a question using conversational RAG.
    Automatically handles follow-up questions using history.
    """
    result = rag_chain.invoke({
        "input": question,
        "chat_history": history,
    })

    # Update history with this exchange
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=result["answer"]))

    return {
        "answer": result["answer"],
        "sources": list(set(doc.metadata.get("source", "unknown") for doc in result.get("context", []))),
        "num_docs": len(result.get("context", [])),
    }

# Turn 1: Initial question
q1 = "What is LCEL and how does it work?"
result1 = chat_with_rag(q1, chat_history)
print(f"\nYou: {q1}")
print(f"AI : {result1['answer']}")
print(f"   (Sources: {result1['sources']}, {result1['num_docs']} chunks retrieved)")

# Turn 2: Follow-up using "it" — tests history-aware retrieval
q2 = "Can it do streaming? How?"
result2 = chat_with_rag(q2, chat_history)
print(f"\nYou: {q2}")
print(f"AI : {result2['answer']}")
print(f"   (The chain rephrased 'it' to refer to LCEL using history)")
print(f"   (Sources: {result2['sources']})")

# Turn 3: Switching topics
q3 = "What about memory in LangChain? Is it different from RAG?"
result3 = chat_with_rag(q3, chat_history)
print(f"\nYou: {q3}")
print(f"AI : {result3['answer']}")
print(f"   (Sources: {result3['sources']})")

# Turn 4: Another follow-up
q4 = "Which of the two options is better for a chatbot?"
result4 = chat_with_rag(q4, chat_history)
print(f"\nYou: {q4}")
print(f"AI : {result4['answer']}")
print(f"   (Rephrased 'the two options' using history)")

# Show the conversation history
print(f"\nTotal messages in history: {len(chat_history)}")

# =============================================================================
# SECTION 5: INTERACTIVE CONVERSATIONAL RAG
# =============================================================================
print()
print("=" * 50)
print("  Interactive Conversational RAG Chat")
print("  (The AI remembers your conversation AND retrieves from docs)")
print("=" * 50)
print("\nType 'clear' to reset conversation, 'exit' to stop.\n")

live_history = []

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue
    elif user_input.lower() == "exit":
        print("Goodbye!")
        break
    elif user_input.lower() == "clear":
        live_history = []
        print("  (Conversation history cleared)\n")
        continue

    result = chat_with_rag(user_input, live_history)
    print(f"\nAI: {result['answer']}")
    print(f"   [Sources: {', '.join(result['sources'])} | {result['num_docs']} chunks]\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 20")
print("=" * 60)
print("""
  1. Conversational RAG = RAG + Memory in one unified system
  2. create_history_aware_retriever() rephrases follow-up questions using history
  3. create_stuff_documents_chain() stuffs retrieved docs into {context}
  4. create_retrieval_chain() binds retriever + QA chain together
  5. Output dict has: input, chat_history, context (docs), answer
  6. Update chat_history with HumanMessage + AIMessage after each turn

  The full flow:
    user query + history → rephrase → retrieve → answer with context

  PART 5 (RAG) COMPLETE — you can build any RAG system!

  Next up: 21_tools_builtin.py
  Giving the LLM tools — search, calculation, databases, and more.
""")
