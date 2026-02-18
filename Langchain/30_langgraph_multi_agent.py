# =============================================================================
# FILE: 30_langgraph_multi_agent.py
# PART: 8 - LangGraph  |  LEVEL: Expert
# =============================================================================
#
# THE STORY:
#   One agent trying to research, write, AND critique a report is like
#   asking one person to be a journalist, editor, and fact-checker
#   simultaneously. They'll get confused and the quality will suffer.
#
#   Multi-agent systems split the work:
#   - Researcher: gathers information
#   - Writer: turns it into polished prose
#   - Critic: reviews and requests improvements
#   - Supervisor: coordinates who does what next
#
#   LangGraph makes this clean: each agent is just a node.
#   State flows between them. The supervisor decides routing.
#   This is how production-grade AI pipelines are built.
#
# WHAT YOU WILL LEARN:
#   1. Multi-agent architecture with a Supervisor pattern
#   2. Shared TypedDict state flowing through all agents
#   3. Each specialist agent as a simple node function
#   4. Supervisor routing: decides who acts next based on state
#   5. Revision cycles: Critic → Writer loop until quality passes
#   6. How to build report-generation pipelines
#
# HOW THIS CONNECTS:
#   Previous: 29_langgraph_stateful_agent.py — stateful ReAct agent
#   Next:     31_async_and_batching.py — concurrent execution patterns
# =============================================================================

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

print("=" * 60)
print("  CHAPTER 30: Multi-Agent System with LangGraph")
print("=" * 60)

# =============================================================================
# SECTION 1: THE SHARED STATE
# =============================================================================
# All agents read from and write to this shared state.
# It flows through the graph like a baton in a relay race.
# Each agent adds its contribution — the state accumulates context.

print("\n--- Section 1: Shared Report State ---")

class ReportState(TypedDict):
    """
    The shared state flowing through all agents.

    topic:          The user's original research topic
    research_notes: Raw facts gathered by the Researcher
    draft:          The Writer's prose draft of the report
    critique:       The Critic's review and suggestions
    final_report:   The polished final output
    revision_count: How many revision cycles have occurred
    next_agent:     Which agent the Supervisor routes to next
    status:         "in_progress" | "complete"
    """
    topic: str
    research_notes: str
    draft: str
    critique: str
    final_report: str
    revision_count: int
    next_agent: str
    status: str

print("""
  ReportState flows through:
  START → Supervisor → Researcher → Supervisor
                    → Writer     → Supervisor
                    → Critic     → Supervisor
                    → (revise?)  → Supervisor → END
""")

# =============================================================================
# SECTION 2: THE LLM
# =============================================================================

llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=0.3,
)

# =============================================================================
# SECTION 3: THE SUPERVISOR AGENT
# =============================================================================
# The Supervisor reads the current state and decides who acts next.
# It's the director of the whole production.
# It routes to: researcher → writer → critic → (done or revise)

print("--- Section 3: Supervisor Agent ---")

def supervisor(state: ReportState) -> ReportState:
    """
    The coordinator. Reads state and decides who should act next.
    Uses simple logic (could also use an LLM for dynamic routing).
    """
    topic = state.get("topic", "")
    research_notes = state.get("research_notes", "")
    draft = state.get("draft", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # Decision logic: sequential pipeline with one optional revision cycle
    if not research_notes:
        # Step 1: Research hasn't happened yet
        next_agent = "researcher"
    elif not draft:
        # Step 2: We have research but no draft
        next_agent = "writer"
    elif not critique:
        # Step 3: We have a draft but no critique
        next_agent = "critic"
    elif revision_count == 0 and "improve" in critique.lower():
        # Step 4: First critique requested improvements — revise once
        next_agent = "writer"
    else:
        # Step 5: Either critique approved it, or we've revised once already
        next_agent = "done"

    print(f"\n  [SUPERVISOR] State assessment:")
    print(f"    Has research: {'Yes' if research_notes else 'No'}")
    print(f"    Has draft   : {'Yes' if draft else 'No'}")
    print(f"    Has critique: {'Yes' if critique else 'No'}")
    print(f"    Revisions   : {revision_count}")
    print(f"    → Routing to: {next_agent}")

    return {**state, "next_agent": next_agent}

# =============================================================================
# SECTION 4: THE RESEARCHER AGENT
# =============================================================================
# Gathers facts and key points about the topic.
# Returns structured research notes.

print("\n--- Section 4: Researcher Agent ---")

def researcher(state: ReportState) -> ReportState:
    """
    Researches the topic and returns structured notes.
    In production, this would call search tools, databases, APIs.
    Here, the LLM generates realistic research notes.
    """
    topic = state["topic"]
    print(f"\n  [RESEARCHER] Researching: {topic}")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a thorough research analyst. "
            "Gather key facts, statistics, and insights about the given topic. "
            "Format as bullet points with clear, factual statements. "
            "Cover: definition, key facts, current state, challenges, future outlook. "
            "Be specific and data-driven. 8-10 bullet points."
        )),
        HumanMessage(content=f"Research topic: {topic}")
    ])

    research_notes = response.content
    print(f"  [RESEARCHER] Generated {len(research_notes.split(chr(10)))} research points")

    return {**state, "research_notes": research_notes}

# =============================================================================
# SECTION 5: THE WRITER AGENT
# =============================================================================
# Takes the research notes and writes a polished report.
# On revision: also reads the critique and improves based on feedback.

print("\n--- Section 5: Writer Agent ---")

def writer(state: ReportState) -> ReportState:
    """
    Writes (or rewrites) the report based on research notes.
    On revision cycles, incorporates the Critic's feedback.
    """
    topic = state["topic"]
    research_notes = state["research_notes"]
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    is_revision = bool(critique) and revision_count == 0
    print(f"\n  [WRITER] {'Revising draft' if is_revision else 'Writing initial draft'} for: {topic}")

    system_msg = (
        "You are a professional technical writer. "
        "Write a clear, well-structured report using the provided research notes. "
        "Format: Title, Executive Summary (2 sentences), 3-4 body paragraphs, Conclusion. "
        "Use professional tone. Be concise but comprehensive."
    )

    user_content = f"Topic: {topic}\n\nResearch Notes:\n{research_notes}"

    if is_revision and critique:
        user_content += f"\n\nCritic's Feedback (incorporate this):\n{critique}"
        system_msg += " You are revising an existing draft based on critic feedback — address each point raised."

    response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=user_content)
    ])

    new_draft = response.content
    new_revision_count = revision_count + 1 if is_revision else revision_count

    print(f"  [WRITER] Draft written ({len(new_draft.split())} words)")

    return {**state, "draft": new_draft, "revision_count": new_revision_count, "critique": ""}

# =============================================================================
# SECTION 6: THE CRITIC AGENT
# =============================================================================
# Reviews the draft for quality, accuracy, and clarity.
# Either approves it or requests specific improvements.

print("\n--- Section 6: Critic Agent ---")

def critic(state: ReportState) -> ReportState:
    """
    Reviews the draft critically.
    Returns either approval or specific improvement requests.
    """
    topic = state["topic"]
    draft = state["draft"]
    print(f"\n  [CRITIC] Reviewing draft on: {topic}")

    response = llm.invoke([
        SystemMessage(content=(
            "You are a rigorous editorial critic reviewing a research report. "
            "Evaluate the report on: clarity, completeness, accuracy, structure, and impact. "
            "If the report is excellent, start your response with 'APPROVED:' "
            "If it needs improvement, start with 'IMPROVE:' and list specific changes needed. "
            "Be constructive and specific."
        )),
        HumanMessage(content=f"Review this report:\n\n{draft}")
    ])

    critique = response.content
    verdict = "APPROVED" if critique.startswith("APPROVED") else "NEEDS REVISION"
    print(f"  [CRITIC] Verdict: {verdict}")

    return {**state, "critique": critique}

# =============================================================================
# SECTION 7: THE FINALIZER
# =============================================================================
# Prepares the final output and marks the pipeline complete.

def finalizer(state: ReportState) -> ReportState:
    """Marks the report complete and sets the final output."""
    print("\n  [FINALIZER] Report complete! Packaging final output.")

    final = state["draft"]
    topic = state["topic"]
    revisions = state["revision_count"]

    # Add a header with metadata
    header = f"# Research Report: {topic}\n"
    header += f"_Generated by Multi-Agent Pipeline | Revisions: {revisions}_\n\n"
    header += "---\n\n"

    return {**state, "final_report": header + final, "status": "complete"}

# =============================================================================
# SECTION 8: BUILD THE MULTI-AGENT GRAPH
# =============================================================================
print("\n--- Section 8: Building the Multi-Agent Graph ---")

# The routing function reads next_agent from state and routes accordingly
def route_from_supervisor(state: ReportState) -> Literal["researcher", "writer", "critic", "finalizer"]:
    """Routes from the supervisor node to the appropriate specialist."""
    return state["next_agent"] if state["next_agent"] != "done" else "finalizer"

# Build the graph
builder = StateGraph(ReportState)

# Add all agent nodes
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)
builder.add_node("critic", critic)
builder.add_node("finalizer", finalizer)

# Start with the supervisor (it decides what's needed)
builder.add_edge(START, "supervisor")

# Supervisor uses conditional routing to dispatch to specialists
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "researcher": "researcher",
        "writer":     "writer",
        "critic":     "critic",
        "finalizer":  "finalizer",
    }
)

# All specialists report back to the supervisor after completing
builder.add_edge("researcher", "supervisor")
builder.add_edge("writer",     "supervisor")
builder.add_edge("critic",     "supervisor")

# Finalizer is the terminal node
builder.add_edge("finalizer", END)

# Compile the graph
multi_agent_graph = builder.compile()

print("  Graph structure:")
print("  START → Supervisor → Researcher → Supervisor")
print("                    → Writer     → Supervisor")
print("                    → Critic     → Supervisor")
print("                    → Finalizer  → END")

# Visualize
try:
    print("\n  Mermaid Diagram:")
    print(multi_agent_graph.get_graph().draw_mermaid())
except Exception:
    print("  (visualization not available)")

# =============================================================================
# SECTION 9: RUN THE PIPELINE
# =============================================================================
print("\n--- Section 9: Running the Multi-Agent Pipeline ---")
print("\nTopic: 'The Impact of Large Language Models on Software Development'")
print("=" * 60)

initial_state: ReportState = {
    "topic": "The Impact of Large Language Models on Software Development",
    "research_notes": "",
    "draft": "",
    "critique": "",
    "final_report": "",
    "revision_count": 0,
    "next_agent": "",
    "status": "in_progress",
}

# Run the full pipeline
final_state = multi_agent_graph.invoke(initial_state)

# Display results
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print(f"\nStatus      : {final_state['status']}")
print(f"Revisions   : {final_state['revision_count']}")
print(f"\nResearch Notes (first 200 chars):")
print(f"  {final_state['research_notes'][:200]}...")
print(f"\nCritic Verdict:")
print(f"  {final_state['critique'][:150]}..." if final_state['critique'] else "  (Cleared after revision)")
print(f"\n{'=' * 60}")
print("  FINAL REPORT:")
print("=" * 60)
print(final_state["final_report"][:1500])
if len(final_state["final_report"]) > 1500:
    print(f"\n  ... [{len(final_state['final_report']) - 1500} more characters]")

# =============================================================================
# SECTION 10: STREAMING THE PIPELINE
# =============================================================================
print("\n\n--- Section 10: Streaming a Shorter Pipeline ---")
print("Streaming events for: 'Benefits of Test-Driven Development'")
print()

short_initial: ReportState = {
    "topic": "Benefits of Test-Driven Development (TDD)",
    "research_notes": "",
    "draft": "",
    "critique": "",
    "final_report": "",
    "revision_count": 0,
    "next_agent": "",
    "status": "in_progress",
}

for event in multi_agent_graph.stream(short_initial, stream_mode="updates"):
    for node_name, node_output in event.items():
        next_a = node_output.get("next_agent", "")
        status = node_output.get("status", "")
        if next_a:
            print(f"  [{node_name}] → next: {next_a}")
        elif status == "complete":
            print(f"  [{node_name}] → Pipeline complete!")
        else:
            print(f"  [{node_name}] → completed its work")

# =============================================================================
# SUMMARY
# =============================================================================
print()
print("=" * 60)
print("  KEY INSIGHTS FROM CHAPTER 30")
print("=" * 60)
print("""
  1. Multi-agent = each specialist is a node; Supervisor decides routing
  2. Shared TypedDict state flows through all nodes — everyone reads/writes it
  3. Supervisor reads state and sets next_agent; conditional edge routes it
  4. Revision cycles are just loops: Critic → Writer → Critic (until APPROVED)
  5. add_conditional_edges() with a dict maps return values to node names
  6. Each agent is a pure Python function: state → updated state
  7. Finalizer is a terminal node (edges to END) — clean pipeline close

  THE MULTI-AGENT PATTERN:
    supervisor → specialist → supervisor → specialist → ... → END
    State accumulates as the baton passes from node to node.

  PART 8 (LANGGRAPH) COMPLETE!

  Next up: 31_async_and_batching.py
  Concurrent execution: batch processing and async patterns for scale.
""")
