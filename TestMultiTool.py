from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage, HumanMessage, SystemMessage
from typing import TypedDict, Sequence, Annotated, Union, Iterable
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import ToolNode
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def format_response_metadata(metadata: dict) -> None:
    """
    Converts LangChain/Ollama response_metadata into a Json‑formatted string with human‑friendly labels,
    focusing on timing and performance information.
    """

    if not metadata:
        return "No response metadata available."

    def ns_to_ms(ns):
        return round(ns / 1_000_000, 2)

    def ns_to_s(ns):
        return round(ns / 1_000_000_000, 3)

    lines = []
    lines.append("🧠 Model Response Metadata")
    lines.append("=" * 30)

    # Model info
    model = metadata.get("model")
    if model:
        lines.append(f"• Model: {model}")

    # Token counts
    prompt_tokens = metadata.get("prompt_eval_count")
    completion_tokens = metadata.get("eval_count")
    if prompt_tokens is not None or completion_tokens is not None:
        lines.append("• Tokens:")
        if prompt_tokens is not None:
            lines.append(f"  - Prompt tokens: {prompt_tokens}")
        if completion_tokens is not None:
            lines.append(f"  - Completion tokens: {completion_tokens}")

    # Timing info (nanoseconds)
    lines.append("• Timing:")
    timing_fields = {
        "Total generation time": "total_duration",
        "Prompt evaluation time": "prompt_eval_duration",
        "Response generation time": "eval_duration",
        "Model load time": "load_duration"
    }

    for label, key in timing_fields.items():
        value = metadata.get(key)
        if value is not None:
            lines.append(
                f"  - {label}: {ns_to_ms(value)} ms ({ns_to_s(value)} s)"
            )

    tok_per_sec = round(completion_tokens / ns_to_s(metadata.get('eval_duration', 1)), 2) if completion_tokens is not None and metadata.get('eval_duration', 1) > 0 else 'N/A'
    lines.append(f". Token/Second: {tok_per_sec}")
    
    utc_dt = datetime.fromisoformat(metadata.get("created_at", "").replace("Z", "+00:00"))
    eastern_dt = utc_dt.astimezone(ZoneInfo("America/New_York"))
    
    # Finish reason
    finish_reason = metadata.get("done_reason")
    if finish_reason:
        lines.append(f"• Finish reason: {finish_reason}")

    print("\n".join(lines))

    # ---------- Structured JSON ----------
    
    json_data = {
        "model": model,
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "tokens_per_second": tok_per_sec
        },
        "timing_seconds": {
            "total_duration": ns_to_s(metadata.get('total_duration', 0)),
            "prompt_eval_duration": ns_to_s(metadata.get('prompt_eval_duration', 0)),
            "eval_duration": ns_to_s(metadata.get('eval_duration', 0)),
            "load_duration": ns_to_s(metadata.get('load_duration', 0))
        },
        "finish_reason": metadata.get("done_reason"),
        "created_at": eastern_dt.isoformat()
        
    }
    
    #print(json.dumps(json_data, indent=2))
    
    logging_info(json_data)

    

def logging_info(json_data: dict) -> None:
    """
    Append the supplied ``json_data`` to a JSON‑array log.

    The function will:
      * create (and keep) a single `activity.log` file,
      * load the current list of entries if the file already exists,
      * append the new entry (with an optional timestamp field),
      * and write the whole list back as one JSON document.
    """

    
    # Eastern Time (America/New_York) – automatically uses EST/EDT
    eastern_now = datetime.now(tz=ZoneInfo("America/New_York"))

    # Format only the date part
    today_date = eastern_now.strftime("%Y-%m-%d")

    # 1. Resolve the folder where the log lives.
    log_dir: Path = Path("~/Documents/MyWorkspace/ProjectLogs").expanduser()

    # 2. Make sure the folder exists (no error if it already does).
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. Full path to the file. Create a new file each day, so you can easily manage log size and keep things organized.
    
    file_path: Path = log_dir / f"activity_{today_date}.log" 

    
    # ------------------------------------------------------------------
    # 4. Read the existing file (if any) and build a list of entries.
    # ------------------------------------------------------------------
    if file_path.exists() and file_path.stat().st_size > 0:
        # The file already contains something – try to load it.
        try:
            with file_path.open("r", encoding="utf-8") as fp:
                logs = json.load(fp)
            if not isinstance(logs, list):
                raise ValueError("activity.log does not contain a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            # Either the file was corrupted or was empty – start fresh.
            print(f"⚠️  Warning: could not read {file_path}, starting a new log. ({exc})")
            logs = []
    else:
        # Fresh file: start with an empty list.
        logs = []

    # ------------------------------------------------------------------
    # 5. Add a timestamp *inside* the entry so you can sort/search
    # ------------------------------------------------------------------
    json_data = json_data.copy()          # avoid mutating the caller’s dict
    json_data["logged_at"] = datetime.now(tz=ZoneInfo("America/New_York")).isoformat()

    # ------------------------------------------------------------------
    # 6. Append the new entry and write the whole list back as one JSON doc.
    # ------------------------------------------------------------------
    logs.append(json_data)

    with file_path.open("w", encoding="utf-8") as fp:
        json.dump(logs, fp, indent=2, ensure_ascii=False)

    print(f"Entry appended to: {file_path}")

    

@tool
def convert_temp_to_celcius(temp_f: Union[float, int]) -> Union[float, int]:
    """This tool converts a temperature from Fahrenheit to Celsius."""
    temp_c = (temp_f - 32) * 5.0/9.0
    print(f"\n[TOOL CALL] convert_temp_to_celcius(temp_f={temp_f})")
    print(f"[TOOL OUTPUT] temp_c = {temp_c}")
    return temp_c

@tool
def convert_temp_to_kelvin(temp_f: Union[float, int]) -> Union[float, int]:
    """This tool converts a temperature from Fahrenheit to Kelvin."""
    temp_k = (temp_f - 32) * 5.0/9.0 + 273.15
    print(f"\n[TOOL CALL] convert_temp_to_kelvin(temp_f={temp_f})")
    print(f"[TOOL OUTPUT] temp_k = {temp_k}")
    return temp_k




def model_call(state: AgentState) -> AgentState:
    """This function calls the llm"""
    
    system_prompt = """
        You are an expert meteorologist assistant that helps classify given temperatures from User as 
        freezing, cold, warm, or hot. You can use the provided tools to convert temperatures from Fahrenheit to Celsius and Kelvin."""
        
    messages = [
        SystemMessage(content=system_prompt)
    ]
    
    response = llm_tool.invoke(messages + state["messages"])
    print(f"\n[MODEL CALL RESPONSE] {response.content}")
    return {"messages":[response]}


def should_continue(state: AgentState) -> str:
    """This function determines whether the agent should continue or end the conversation."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "continue"
    else:
        return "end"
                

def all_ai_messages_from_state(state: dict) -> Iterable[BaseMessage]:
    """
    Yields every ``AIMessage`` that sits inside the ``state["messages"]`` list.
    """
    if not state:
        return
    for msg in state.get("messages", []):
        if isinstance(msg, AIMessage):
            yield msg
            
def main() -> None: 
    print("Enter a temperature in °F (e.g. 15). Type 'finish' to exit.")
    while True:
        user_input = input("User: Temperature in °F: ").strip()
        if user_input.lower().strip() == "finish":
            print("Good-bye!")
            break

        # Create the initial state (only the user message)
        user_message = HumanMessage(content=user_input)
        
        # Run the agent with streaming output
        last_state = None
        for chunk in app.stream({"messages": [user_message]}, stream_mode="values"):
            last_state = chunk
            message = chunk["messages"][-1]
            if isinstance(message, ToolMessage):
                print(f"\n[TOOL RESPONSE] {message}")
            else:
                message.pretty_print()
                
        #print_stream(app.stream({"messages": [user_message]}, stream_mode="values"))
        #state = graph.invoke({"messages": [user_message]})

        # The LLM’s reply is the last message in the conversation
        answer = last_state["messages"]
        print(f"Final Agent: {answer}")
        
        for ai_msg in all_ai_messages_from_state(last_state):
        
            ai_msg.pretty_print()

            # 4b – get the metadata dict – depends on the LLM you used.
            #   With LangChain / Ollama the reply object typically stores
            #   it in a `response_metadata` attribute.
            metadata = getattr(ai_msg, "response_metadata", None)
            if metadata:
                format_response_metadata(metadata)      # the function you posted
            else:
                print("⚠️  AI reply has no `response_metadata` – nothing logged.")


if __name__ == "__main__":
    model = "gpt-oss:20b"
    llm = ChatOllama(model= model)
    Tools = [convert_temp_to_celcius, convert_temp_to_kelvin]

    llm_tool = llm.bind_tools(Tools)
    graph = StateGraph(AgentState)
    graph.add_node("our_agent", model_call)

    tool_node = ToolNode(tools=Tools)
    graph.add_node("tool_node", tool_node)

    graph.set_entry_point("our_agent")
    graph.add_conditional_edges("our_agent", should_continue, {"continue": "tool_node", "end": END})
    graph.add_edge("tool_node", "our_agent")

    app = graph.compile()
    
    main()

