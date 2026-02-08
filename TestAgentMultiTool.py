from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from IPython.display import Image, display
# ----------------------------------
# LLM
# ----------------------------------
model = "gpt-oss:20b"
model1 = "qwen3-vl:32b"
llm = ChatOllama(model=model, temperature=0)

# ----------------------------------
# State
# ----------------------------------
class AgentState(TypedDict):
    messages: List
    result: int
    phase: Literal["ask", "add", "spell", "log"]
    initialized: bool

# ----------------------------------10
# Tool 1: ADD
# ----------------------------------
@tool
def add_numbers(current: int, new: int) -> int:
    """This tool adds two numbers together."""
    print(f"\n[TOOL CALL] add_numbers(current={current}, new={new})")
    result = current + new
    print(f"[TOOL OUTPUT] result = {result}")
    return result

# ----------------------------------
# Tool 2: SPELL (LLM)
# ----------------------------------
@tool
def spell_number(value: int) -> str:
    """This tool spells a number in English words."""
    
    print(f"\n[TOOL CALL] spell_number(value={value})")

    prompt = f"Spell the number {value} in English words only."
    response = llm.invoke(prompt)

    print(f"[TOOL OUTPUT] spelled = {response.content}")
    return response.content

# ----------------------------------
# Tool 3: LOG METADATA
# ----------------------------------
@tool
def log_metadata(metadata: dict) -> str:
    """This tool logs metadata from the last response."""
    print(f"\n[TOOL CALL] log_metadata(metadata={metadata})")
    print("[TOOL OUTPUT] metadata logged")
    return "logged"

# ----------------------------------
# Tool-scoped LLMs (enforcement)
# ----------------------------------
llm_add = llm.bind_tools([add_numbers])
llm_spell = llm.bind_tools([spell_number])
llm_log = llm.bind_tools([log_metadata])

# ----------------------------------
# ASK NODE
# ----------------------------------
def ask(state: AgentState) -> AgentState:
    if not state["initialized"]:
        a = int(input("\nEnter first number: "))
        b = int(input("Enter second number: "))
        msg = HumanMessage(
            content=f"Add {a} and {b}. Current total is {state['result']}."
        )
        print(f"\nOUTPUT: {msg}")
        return {
            **state,
            "messages": state["messages"] + [msg],
            "phase": "add",
            "initialized": True,
        }

    else:
        n = int(input("\nEnter another number to add: "))

        if n == 0:
            print("\nNo more numbers to add. Exiting.")
            exit(0)
            
        msg = HumanMessage(
            content=f"Add {n} to {state['result']}."
        )
        print(f"\nOUTPUT: {msg}")
        return {
            **state,
            "messages": state["messages"] + [msg],
            "phase": "add",
        }

# ----------------------------------
# ADD NODE
# ----------------------------------
def add_node(state: AgentState) -> AgentState:
    ai_msg = llm_add.invoke(state["messages"])

    tool_call = ai_msg.tool_calls[0]
    args = tool_call["args"]

    # 🔥 EXECUTE TOOL
    result = add_numbers.invoke(args)

    # 🔥 UPDATE STATE
    state["result"] = result

    tool_msg = ToolMessage(
        tool_call_id=tool_call["id"],
        content=str(result),
    )

    print(f"\n✅ CURRENT TOTAL UPDATED → {result}")

    return {
        **state,
        "messages": state["messages"] + [ai_msg, tool_msg],
        "phase": "spell",
    }







# ----------------------------------
# SPELL NODE
# ----------------------------------
def spell_node(state: AgentState) -> AgentState:
    response = llm_spell.invoke(state["messages"])
    return {
        **state,
        "messages": state["messages"] + [response],
        "phase": "log",
    }

# ----------------------------------
# LOG NODE
# ----------------------------------
def log_node(state: AgentState) -> AgentState:
    metadata = state["messages"][-1].response_metadata
    tool_msg = ToolMessage(
        tool_call_id="log",
        content=str(metadata),
    )

    log_metadata.invoke({"metadata": metadata})

    return {
        **state,
        "messages": state["messages"] + [tool_msg],
        "phase": "ask",
    }

# ----------------------------------
# ROUTER (hard enforcement)
# ----------------------------------
def router(state: AgentState):
    return state["phase"]

# ----------------------------------
# GRAPH
# ----------------------------------
graph = StateGraph(AgentState)

graph.add_node("ask", ask)
graph.add_node("add", add_node)
graph.add_node("spell", spell_node)
graph.add_node("log", log_node)

graph.set_entry_point("ask")

graph.add_conditional_edges(
    "ask", router, {"add": "add"}
)
graph.add_conditional_edges(
    "add", router, {"spell": "spell"}
)
graph.add_conditional_edges(
    "spell", router, {"log": "log"}
)
graph.add_conditional_edges(
    "log", router, {"ask": "ask"}
)

app = graph.compile()


png_data = app.get_graph().draw_mermaid_png()
with open("multitool_graph.png", "wb") as f:
    f.write(png_data)

#display(Image(app.get_graph().draw_mermaid_png()))


# ----------------------------------
# RUN
# ----------------------------------
state = {
    "messages": [
        HumanMessage(
            content="You are a calculator agent. Follow the tool order strictly."
        )
    ],
    "result": 0,
    "phase": "ask",
    "initialized": False,
}

try:
    app.invoke(state)
except KeyboardInterrupt:
    print("\nStopped by user.")
