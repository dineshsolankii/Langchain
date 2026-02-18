# =============================================================================
# FILE: 25_agents_multi_tool.py
# PART: 6 - Tools & Agents  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   Let's build something REAL — a customer support agent.
#   It can search the knowledge base, look up customer records,
#   calculate billing, create tickets, and send emails.
#   All from one natural language conversation. All with memory.
#
#   This is the culmination of everything we've learned:
#   Tools + Agents + Memory = A production-ready AI assistant
#
# WHAT YOU WILL LEARN:
#   1. Building an agent with 5 specialized tools
#   2. Adding persistent session memory to an agent
#   3. RunnableWithMessageHistory wrapping AgentExecutor
#   4. Multi-turn conversation with a stateful agent
#   5. How to structure a complete AI assistant
#
# HOW THIS CONNECTS:
#   Previous: 24_agents_openai_functions.py — tool-calling agents
#   Next:     26_callbacks_and_streaming.py — monitoring and logging
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.tools.base import ToolException
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from datetime import datetime

load_dotenv()

print("=" * 60)
print("  CHAPTER 25: Multi-Tool Agent with Memory")
print("=" * 60)
print("""
  Building a complete customer support agent:
  Tools: search_kb + lookup_customer + calculate_refund + create_ticket + send_email
  Memory: RunnableWithMessageHistory for multi-turn conversations
""")

# =============================================================================
# SETUP: Knowledge Base for Tool 1
# =============================================================================
# The agent will use this as a retrieval-augmented knowledge base

kb_docs = [
    Document(page_content="To reset your password: go to Settings > Security > Reset Password. You'll receive an email within 5 minutes."),
    Document(page_content="Subscription plans: Starter ($49/mo, 5 users), Professional ($299/mo, 25 users), Enterprise (custom pricing)."),
    Document(page_content="To cancel your subscription: go to Settings > Billing > Cancel Plan. Effective at end of billing period."),
    Document(page_content="Refunds are available within 30 days of charge if requested. Contact support with order ID for processing."),
    Document(page_content="Two-factor authentication (2FA) can be enabled in Settings > Security > Enable 2FA. Supports authenticator apps and SMS."),
    Document(page_content="API rate limits: Starter plan 100 req/min, Professional 1000 req/min, Enterprise unlimited."),
    Document(page_content="Data export: Go to Settings > Data > Export All. Downloads a ZIP with all your data in JSON format."),
]

embeddings = OpenAIEmbeddings(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/text-embedding-3-small",
)

vectorstore = FAISS.from_documents(kb_docs, embeddings)
kb_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# =============================================================================
# MOCK DATA
# =============================================================================
CUSTOMERS = {
    "C001": {"name": "Dinesh Solanki", "email": "dinesh@example.com", "plan": "Professional", "active": True, "billing_day": 1},
    "C002": {"name": "Alice Johnson", "email": "alice@example.com", "plan": "Starter", "active": True, "billing_day": 15},
    "C003": {"name": "Bob Williams", "email": "bob@example.com", "plan": "Enterprise", "active": False, "billing_day": 1},
}

ORDERS = {
    "ORD-001": {"customer_id": "C001", "amount": 299.0, "date": "2025-01-01", "description": "Professional Plan - January"},
    "ORD-002": {"customer_id": "C002", "amount": 49.0, "date": "2025-01-15", "description": "Starter Plan - January"},
}

TICKETS = []
SENT_EMAILS = []
ticket_num = 5000

# =============================================================================
# THE FIVE TOOLS
# =============================================================================

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the company knowledge base for answers to customer questions.
    Use this FIRST for any product, feature, billing, or policy question.
    Returns the most relevant help articles.
    """
    docs = kb_retriever.invoke(query)
    if not docs:
        return "No relevant information found in the knowledge base."
    return "\n\n".join(f"- {doc.page_content}" for doc in docs)

@tool
def lookup_customer(customer_id: str) -> str:
    """
    Look up a customer's account details by their customer ID.
    Use this to verify account status, plan, and billing info.
    Customer IDs are in format C001, C002, etc.
    """
    customer = CUSTOMERS.get(customer_id.upper().strip())
    if not customer:
        raise ToolException(f"Customer '{customer_id}' not found. Valid IDs: {', '.join(CUSTOMERS.keys())}")

    status = "Active" if customer["active"] else "Cancelled"
    return (
        f"Name: {customer['name']}\n"
        f"Email: {customer['email']}\n"
        f"Plan: {customer['plan']}\n"
        f"Status: {status}\n"
        f"Billing day: {customer['billing_day']}th of each month"
    )

@tool
def check_refund_eligibility(order_id: str) -> str:
    """
    Check if an order is eligible for a refund.
    Refunds are allowed within 30 days of purchase.
    Use this before processing a refund request.
    """
    order = ORDERS.get(order_id.upper().strip())
    if not order:
        raise ToolException(f"Order '{order_id}' not found. Valid IDs: {', '.join(ORDERS.keys())}")

    order_date = datetime.strptime(order["date"], "%Y-%m-%d")
    days_ago = (datetime.now() - order_date).days

    if days_ago <= 30:
        return (
            f"Order {order_id}: ${order['amount']:.2f} ({order['description']})\n"
            f"Placed {days_ago} days ago — ELIGIBLE for refund (within 30-day window)."
        )
    else:
        return (
            f"Order {order_id}: ${order['amount']:.2f} ({order['description']})\n"
            f"Placed {days_ago} days ago — NOT eligible for refund (past 30-day window)."
        )

@tool
def create_support_ticket(customer_id: str, issue_summary: str, priority: str = "medium") -> str:
    """
    Create a support ticket for a customer issue that requires follow-up.
    Use this when an issue can't be resolved immediately in the conversation.
    Priority: 'low', 'medium', 'high', 'critical'
    """
    global ticket_num
    ticket_num += 1
    ticket_id = f"TKT-{ticket_num:04d}"

    TICKETS.append({
        "id": ticket_id,
        "customer_id": customer_id.upper(),
        "issue": issue_summary,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })

    return (
        f"Support ticket created!\n"
        f"Ticket ID: {ticket_id}\n"
        f"Priority: {priority.upper()}\n"
        f"Our team will respond within "
        f"{'1 hour' if priority in ['high', 'critical'] else '24 hours'}."
    )

@tool
def send_email(to_email: str, subject: str, message: str) -> str:
    """
    Send an email to the customer.
    Use this to send confirmations, follow-ups, or resolution summaries.
    Only send when explicitly requested or when confirming important actions.
    """
    if "@" not in to_email:
        raise ToolException(f"Invalid email address: '{to_email}'")

    SENT_EMAILS.append({
        "to": to_email, "subject": subject,
        "message": message, "sent_at": datetime.now().strftime("%H:%M")
    })

    print(f"\n  [EMAIL] To: {to_email} | Subject: {subject}")
    return f"Email sent to {to_email} with subject '{subject}'."

tools = [search_knowledge_base, lookup_customer, check_refund_eligibility, create_support_ticket, send_email]

# =============================================================================
# BUILD THE AGENT WITH MEMORY
# =============================================================================
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

# System prompt for customer support role
system_prompt = """You are a helpful and professional customer support agent for TechCorp.

You have access to these tools:
- search_knowledge_base: Search help articles for product/policy questions
- lookup_customer: Get customer account details by ID (format: C001, C002, etc.)
- check_refund_eligibility: Check if an order qualifies for a refund
- create_support_ticket: Create a ticket for issues needing follow-up
- send_email: Send emails to customers

Guidelines:
- Always search the knowledge base before escalating to a ticket
- Look up the customer when they provide their ID
- Be empathetic, professional, and solution-focused
- Summarize what you did at the end of each interaction
"""

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),  # Memory slot
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
])

agent = create_tool_calling_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
    handle_tool_error=True,
)

# Add memory via RunnableWithMessageHistory
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap the agent executor with memory
agent_with_memory = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# =============================================================================
# DEMO CONVERSATION
# =============================================================================
print()
print("=" * 50)
print("  DEMO: Multi-Turn Customer Support Conversation")
print("=" * 50)

session = {"configurable": {"session_id": "support_session_001"}}

# Turn 1: Introduction
print("\n[Turn 1] Customer introduces themselves")
r1 = agent_with_memory.invoke({"input": "Hi, I'm customer C001 and I'm having trouble resetting my password."}, config=session)
print(f"\nAgent: {r1['output']}")

# Turn 2: Follow-up (agent should remember customer C001)
print("\n[Turn 2] Customer asks about their plan")
r2 = agent_with_memory.invoke({"input": "Also, can you remind me what subscription plan I'm on?"}, config=session)
print(f"\nAgent: {r2['output']}")

# Turn 3: Refund request
print("\n[Turn 3] Customer requests a refund")
r3 = agent_with_memory.invoke({"input": "I was charged for order ORD-001. Can I get a refund?"}, config=session)
print(f"\nAgent: {r3['output']}")

# Turn 4: Request a ticket and email
print("\n[Turn 4] Customer wants a ticket created")
r4 = agent_with_memory.invoke(
    {"input": "Please create a support ticket for my password issue and email me a summary at dinesh@example.com"},
    config=session
)
print(f"\nAgent: {r4['output']}")

# Show what was created
print(f"\n--- Session Summary ---")
print(f"Tickets created : {len(TICKETS)}")
for t in TICKETS:
    print(f"  {t['id']}: {t['issue'][:50]}... ({t['priority'].upper()})")
print(f"Emails sent     : {len(SENT_EMAILS)}")
for e in SENT_EMAILS:
    print(f"  To: {e['to']} | {e['subject']}")

# =============================================================================
# INTERACTIVE MODE
# =============================================================================
print()
print("=" * 50)
print("  Interactive Customer Support Agent")
print("  (The agent has memory — it remembers the whole conversation)")
print("  Type 'exit' to stop, 'new' to start fresh session")
print("=" * 50 + "\n")

session_id = "interactive_session"

while True:
    user_input = input("Customer: ").strip()
    if not user_input:
        continue
    if user_input.lower() == "exit":
        print("Thank you for contacting TechCorp Support. Goodbye!")
        break
    if user_input.lower() == "new":
        session_id = f"session_{len(store) + 1}"
        print(f"  (New session started: {session_id})\n")
        continue

    try:
        result = agent_with_memory.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        print(f"\nAgent: {result['output']}\n")
    except Exception as e:
        print(f"\nError: {e}\n")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 25")
print("=" * 60)
print("""
  1. Multi-tool agents combine specialized tools into one assistant
  2. RunnableWithMessageHistory wraps AgentExecutor for session memory
  3. The agent_prompt has both {chat_history} and {agent_scratchpad} slots
  4. Tool docstrings are critical — they tell the agent when to use each tool
  5. The agent decides tool order based on the conversation context
  6. Memory + tools = conversational AI that takes action

  PART 6 (TOOLS & AGENTS) COMPLETE — you can build full AI assistants!

  Next up: 26_callbacks_and_streaming.py
  Monitoring, logging, and observability for your chains and agents.
""")
