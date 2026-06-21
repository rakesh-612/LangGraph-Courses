from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage, trim_messages
from langgraph.graph import END, START, StateGraph, MessagesState

from chains import revisor, first_responder
from tool_executor import execute_tools

MAX_ITERATIONS = 2
MAX_TOKENS = 5000


def _count_tokens(messages) -> int:
    """Count tokens using tiktoken — works without needing the raw llm."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        total += len(enc.encode(content))
    return total


def _trim(messages):
    """Trim message history to stay within Groq TPM limits."""
    return trim_messages(
        messages,
        strategy="last",
        max_tokens=MAX_TOKENS,
        token_counter=_count_tokens,  # plain function, no raw llm needed
        include_system=True,
        allow_partial=False,
    )


def draft_node(state: MessagesState):
    """Draft the initial response."""
    messages = _trim(state["messages"])
    response = first_responder.invoke({"messages": messages})
    return {"messages": [response]}


def revise_node(state: MessagesState):
    """Revise the answer based on tool results."""
    messages = _trim(state["messages"])
    response = revisor.invoke({"messages": messages})
    return {"messages": [response]}


def event_loop(state: MessagesState) -> Literal["execute_tools", "__end__"]:
    """Continue looping until MAX_ITERATIONS tool calls have been made."""
    count_tool_visits = sum(
        isinstance(item, ToolMessage) for item in state["messages"]
    )
    if count_tool_visits >= MAX_ITERATIONS:
        return END
    return "execute_tools"


builder = StateGraph(MessagesState)
builder.add_node("draft", draft_node)
builder.add_node("execute_tools", execute_tools)
builder.add_node("revise", revise_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "execute_tools")
builder.add_edge("execute_tools", "revise")
builder.add_conditional_edges("revise", event_loop, ["execute_tools", END])
graph = builder.compile()

print(graph.get_graph().draw_mermaid())

res = graph.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Write about AI-Powered SOC / autonomous SOC problem domain,"
                " list startups that do that and raised capital.",
            }
        ]
    }
)

last_message = res["messages"][-1]
if isinstance(last_message, AIMessage) and last_message.tool_calls:
    print("\n=== FINAL ANSWER ===")
    print(last_message.tool_calls[0]["args"]["answer"])
    if "references" in last_message.tool_calls[0]["args"]:
        print("\n=== REFERENCES ===")
        for ref in last_message.tool_calls[0]["args"]["references"]:
            print(ref)
