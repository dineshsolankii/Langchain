# =============================================================================
# FILE: 22_tools_custom.py
# PART: 6 - Tools & Agents  |  LEVEL: Advanced
# =============================================================================
#
# THE STORY:
#   Built-in tools are great — but YOUR application has unique needs.
#   Your CRM database. Your product catalog. Your billing system.
#   Your company's internal APIs.
#
#   Custom tools turn YOUR business logic into LLM superpowers.
#   Build once, use in any agent, forever.
#
# WHAT YOU WILL LEARN:
#   1. Customer lookup tool (mock CRM database)
#   2. Safe calculator tool (no eval() dangers)
#   3. Date/time arithmetic tool
#   4. Mock email/notification sender
#   5. ToolException — proper error handling in tools
#   6. StructuredTool — alternative to @tool for reuse
#
# HOW THIS CONNECTS:
#   Previous: 21_tools_builtin.py — built-in tools and @tool decorator
#   Next:     23_agents_react.py — wiring tools to a ReAct agent
# =============================================================================

import os
from dotenv import load_dotenv
from langchain_core.tools import tool, StructuredTool
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional

load_dotenv()

print("=" * 60)
print("  CHAPTER 22: Building Custom Tools")
print("=" * 60)

# =============================================================================
# TOOL 1: CUSTOMER LOOKUP (Mock CRM Database)
# =============================================================================
# Simulates a CRM database. In production, this would query a real database.
# The LLM can look up customer info by ID to answer support questions.

# Mock database — in production, replace with an actual DB query
CUSTOMER_DB = {
    "C001": {
        "name": "Dinesh Solanki",
        "email": "dinesh@example.com",
        "plan": "Professional",
        "joined": "2023-01-15",
        "active": True,
        "monthly_spend": 299.0,
    },
    "C002": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "plan": "Starter",
        "joined": "2024-03-20",
        "active": True,
        "monthly_spend": 49.0,
    },
    "C003": {
        "name": "Bob Williams",
        "email": "bob@example.com",
        "plan": "Enterprise",
        "joined": "2022-11-01",
        "active": False,  # Cancelled account
        "monthly_spend": 0.0,
    },
}

@tool
def lookup_customer(customer_id: str) -> str:
    """
    Look up customer information from the CRM database by customer ID.
    Use this to find customer name, email, subscription plan, and account status.
    Customer IDs follow the format: C001, C002, etc.
    """
    customer_id = customer_id.upper().strip()

    if customer_id not in CUSTOMER_DB:
        raise ToolException(
            f"Customer {customer_id} not found in the database. "
            f"Valid IDs are: {', '.join(CUSTOMER_DB.keys())}"
        )

    customer = CUSTOMER_DB[customer_id]
    status = "Active" if customer["active"] else "Cancelled"

    return (
        f"Customer: {customer['name']}\n"
        f"Email: {customer['email']}\n"
        f"Plan: {customer['plan']}\n"
        f"Status: {status}\n"
        f"Member since: {customer['joined']}\n"
        f"Monthly spend: ${customer['monthly_spend']:.2f}"
    )

# =============================================================================
# TOOL 2: SAFE CALCULATOR
# =============================================================================
# Never use eval() in production — it's a security risk.
# This calculator supports common operations safely using Python's ast module.

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.
    Supports: +, -, *, /, **, %, sqrt, abs, round.
    Examples: '2 + 2', '15 * 8', '100 / 7', '2 ** 10', 'sqrt(144)'

    Args:
        expression: A mathematical expression as a string
    """
    import ast
    import math

    # Map of safe functions and constants the calculator can use
    safe_env = {
        "sqrt": math.sqrt,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "max": max,
        "min": min,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        # Parse the expression first (without evaluating) for safety
        tree = ast.parse(expression, mode='eval')

        # Compile and evaluate in a restricted namespace
        code = compile(tree, '<string>', 'eval')
        result = eval(code, {"__builtins__": {}}, safe_env)

        return f"{expression} = {result}"

    except ZeroDivisionError:
        raise ToolException("Division by zero is not allowed.")
    except SyntaxError:
        raise ToolException(f"Invalid expression syntax: '{expression}'")
    except Exception as e:
        raise ToolException(f"Calculation failed: {str(e)}")

# =============================================================================
# TOOL 3: DATE AND TIME ARITHMETIC
# =============================================================================

@tool
def days_until(target_date: str) -> str:
    """
    Calculate how many days until a future date, or how many days since a past date.
    Use this for deadline calculations, event countdowns, or age questions.

    Args:
        target_date: Date in YYYY-MM-DD format (e.g., '2025-12-31')
    """
    try:
        target = datetime.strptime(target_date.strip(), "%Y-%m-%d")
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        delta = (target - today).days

        if delta > 0:
            return f"{delta} days until {target_date} ({target.strftime('%A, %B %d, %Y')})"
        elif delta == 0:
            return f"Today is {target_date}!"
        else:
            return f"{abs(delta)} days since {target_date} ({target.strftime('%A, %B %d, %Y')})"

    except ValueError:
        raise ToolException(
            f"Invalid date format: '{target_date}'. "
            "Please use YYYY-MM-DD format (e.g., '2025-12-31')"
        )

# =============================================================================
# TOOL 4: SEND EMAIL (Mock)
# =============================================================================
# In production, this would use smtplib, SendGrid, or AWS SES.
# For learning: it just prints what would be sent.

SENT_EMAILS = []  # Track sent emails for demonstration

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    Send an email to a specified recipient.
    Use this to notify customers, send confirmations, or communicate updates.
    Only use when the user explicitly asks to send an email.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body text (plain text)
    """
    if "@" not in to:
        raise ToolException(f"Invalid email address: '{to}'")

    # Log the email (in production: actually send it)
    email_record = {
        "to": to,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    SENT_EMAILS.append(email_record)

    print(f"\n  [EMAIL SYSTEM] Email sent!")
    print(f"  To     : {to}")
    print(f"  Subject: {subject}")
    print(f"  Body   : {body[:100]}...")

    return f"Email successfully sent to {to} with subject '{subject}'."

# =============================================================================
# TOOL 5: CREATE SUPPORT TICKET
# =============================================================================
TICKETS = []
ticket_counter = 1000

@tool
def create_support_ticket(customer_id: str, issue: str, priority: str = "medium") -> str:
    """
    Create a support ticket in the ticketing system for a customer issue.
    Use this when a customer reports a problem that needs follow-up.

    Args:
        customer_id: The customer's ID (e.g., C001)
        issue: Description of the problem or issue
        priority: Ticket priority - 'low', 'medium', or 'high'
    """
    global ticket_counter

    valid_priorities = ["low", "medium", "high", "critical"]
    if priority.lower() not in valid_priorities:
        priority = "medium"

    ticket_id = f"TKT-{ticket_counter:04d}"
    ticket_counter += 1

    ticket = {
        "id": ticket_id,
        "customer_id": customer_id.upper(),
        "issue": issue,
        "priority": priority.lower(),
        "status": "open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    TICKETS.append(ticket)

    return (
        f"Support ticket created successfully!\n"
        f"Ticket ID : {ticket_id}\n"
        f"Customer  : {customer_id.upper()}\n"
        f"Priority  : {priority.upper()}\n"
        f"Status    : Open\n"
        f"Issue     : {issue[:100]}"
    )

# =============================================================================
# TEST ALL TOOLS INDIVIDUALLY
# =============================================================================
print("\n--- Testing All Custom Tools ---")

print("\n[1] Customer Lookup:")
try:
    print(lookup_customer.invoke({"customer_id": "C001"}))
except Exception as e:
    print(f"Error: {e}")

print("\n[2] Customer Not Found (Error Handling):")
try:
    print(lookup_customer.invoke({"customer_id": "C999"}))
except ToolException as e:
    print(f"ToolException caught: {e}")

print("\n[3] Calculator:")
print(calculator.invoke({"expression": "2 ** 10"}))
print(calculator.invoke({"expression": "sqrt(144) + 3 * 5"}))
print(calculator.invoke({"expression": "15 * 8 / 100"}))

print("\n[4] Days Until:")
print(days_until.invoke({"target_date": "2025-12-31"}))

print("\n[5] Send Email:")
print(send_email.invoke({
    "to": "dinesh@example.com",
    "subject": "Your account update",
    "body": "Your subscription has been renewed successfully. Thank you!"
}))

print("\n[6] Create Support Ticket:")
print(create_support_ticket.invoke({
    "customer_id": "C001",
    "issue": "Cannot access dashboard after recent update",
    "priority": "high"
}))

# =============================================================================
# TOOL COLLECTION
# =============================================================================
print("\n--- Custom Tool Collection Ready for Agents ---")

custom_tools = [
    lookup_customer,
    calculator,
    days_until,
    send_email,
    create_support_ticket,
]

print(f"\n{len(custom_tools)} custom tools ready:")
for t in custom_tools:
    print(f"  - {t.name}")

# =============================================================================
# SECTION: ToolException — PROPER ERROR HANDLING
# =============================================================================
print("\n--- ToolException Pattern ---")
print("""
  When a tool encounters an error (invalid input, resource not found, etc.):
  1. raise ToolException("descriptive error message")
  2. The agent sees this error message and can:
     - Try a different approach
     - Ask the user for correct information
     - Report the error gracefully

  WRONG way:
    return "Error: customer not found"  # LLM might not recognize this as an error

  RIGHT way:
    raise ToolException("Customer C999 not found. Valid IDs: C001, C002, C003")
    # Agent clearly understands this is an error and handles it accordingly

  In AgentExecutor, set handle_tool_error=True to prevent crashes:
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        handle_tool_error=True,  # Catches ToolException and shows error to agent
    )
""")

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 22")
print("=" * 60)
print("""
  1. Custom tools: @tool decorator + clear docstring = LLM-ready tool
  2. The docstring tells the LLM WHEN to use the tool — make it specific!
  3. Type hints define the input schema — use descriptive names
  4. raise ToolException() for errors — the agent handles these gracefully
  5. Mock external services first, replace with real API calls in production
  6. Test tools with tool.invoke({"arg": value}) BEFORE adding to an agent
  7. Collect all tools in a list: tools = [tool1, tool2, ...]

  Next up: 23_agents_react.py
  Wiring tools to a ReAct agent — watch the AI reason and act step by step.
""")
