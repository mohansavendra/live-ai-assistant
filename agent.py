from typing import TypedDict, Annotated, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator

from tools import ALL_TOOLS
from config import GROQ_API_KEY, GROQ_MODEL

# ─── STATE DEFINITION ─────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    persistent_context: str   # injected from ChromaDB if enabled


# ─── LLM SETUP ────────────────────────────────────────────────────────────────

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.3,
    max_tokens=2048,
)

llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_PROMPT = """You are a highly capable AI assistant with real-time internet access.

Your capabilities:
- web_search: search the internet for current, real-time information
- verify_fact: verify specific claims or facts using live web data

DECISION RULES:
1. If the question involves current events, recent news, live data, prices, scores, 
   or anything that changes over time → ALWAYS use web_search first.
2. If the user asks you to verify or fact-check something → use verify_fact.
3. If the question is purely conceptual, mathematical, or clearly within your training 
   knowledge and time-insensitive → answer directly without searching.
4. After searching, synthesize the results clearly. Cite sources when relevant.
5. If past context is provided below, use it to personalize your response.

Always be direct, precise, and honest about uncertainty. If search results conflict,
surface both perspectives and note the discrepancy.
"""


# ─── AGENT NODES ──────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    """Main reasoning node — decides whether to call tools or answer directly."""
    messages = state["messages"]
    persistent_context = state.get("persistent_context", "")

    # Inject system prompt + optional persistent memory context
    system_content = SYSTEM_PROMPT
    if persistent_context and persistent_context != "No past memories found.":
        system_content += f"\n\nRELEVANT PAST CONTEXT FROM MEMORY:\n{persistent_context}"

    full_messages = [SystemMessage(content=system_content)] + messages

    response = llm_with_tools.invoke(full_messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """Router: if last message has tool calls, go to tools; else end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# ─── GRAPH ASSEMBLY ───────────────────────────────────────────────────────────

tool_node = ToolNode(ALL_TOOLS)

graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph_builder.add_edge("tools", "agent")  # after tools, always return to agent

agent_graph = graph_builder.compile()


# ─── PUBLIC INTERFACE ─────────────────────────────────────────────────────────

def run_agent(
    user_input: str,
    session_history: List[dict],
    persistent_context: str = ""
) -> str:
    """
    Run the agent with full session history and optional persistent memory context.
    Returns the final assistant response as a string.
    """
    # Convert session history dicts to LangChain message objects
    lc_messages = []
    for msg in session_history:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Add current user input
    lc_messages.append(HumanMessage(content=user_input))

    initial_state: AgentState = {
        "messages": lc_messages,
        "persistent_context": persistent_context,
    }

    final_state = agent_graph.invoke(initial_state)

    # Extract the last AI message
    for msg in reversed(final_state["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content

    return "I was unable to generate a response. Please try again."
