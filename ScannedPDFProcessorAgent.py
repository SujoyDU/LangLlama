from typing import TypedDict, List, Sequence, Annotated
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pdf2image import convert_from_path
import base64
import os
from io import BytesIO
from datetime import datetime
from zoneinfo import ZoneInfo
import json


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]
    
model_name = "qwen3-vl:8b"
llm = ChatOllama(model=model_name)

@tool
def pdf_to_str(pdf_file:str) -> str:
    """This tool converts a scanned pdf file to base64 encoded image strings"""
    if not os.path.exists(pdf_file):
        return f"File {pdf_file} does not exist."
    
    if not pdf_file.lower().endswith('.pdf'):
        return "The provided file is not a PDF."
    
    pages = convert_from_path(pdf_file)
    encoded_images = []
    for page in pages:
        buffered = BytesIO()
        page.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        encoded_images.append(img_str)
    
    return encoded_images

@tool
def image_to_text(image_str:str) -> str:
    """This tool takes a base64 encoded image string and extracts text using the ChatOllama model."""
    messages = [
        SystemMessage(content="You are a helpful assistant that extracts text from images."),
        HumanMessage(
            content=[
                {"type": "text", "text": "Extract all text and table data from this image. Please keep the original formatting."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_str}"}
                },
            ]
        )
    ]
    response = llm.invoke(messages)
    print("\n\nExtracted Text: ")
    print(response.content)
    return response.content


@tool
def save_text_to_file(text: str, filename: str) -> str:
    """This tool saves the extracted text to a file."""
    
    filepath = Path(filename)    
    with filepath.open("a", encoding="utf-8") as f:
        f.write(text)
    
    return f"Text saved to {filename}."



def format_response_metadata(metadata: dict) -> None:
    """
    Converts LangChain/Ollama response_metadata into a Json-formatted string with human-friendly labels,
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
    Append the supplied ``json_data`` to a JSON-array log.

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


def save_response_metadata(state: AgentState) -> AgentState:
    """This function extracts the response metadata from AI messages in the agent's state, formats it, and saves it to a log file."""
   
    def _save_metadata(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
        new_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.response_metadata:
                metadata_str = json.dumps(msg.response_metadata, indent=2)
                print(f"\n\n**Response Metadata:**\n{metadata_str}\n\n")
            new_messages.append(msg)
        return new_messages
    
    state['messages'] = _save_metadata(state['messages'])
    return state
    

llm = llm.bind_tools(pdf_to_str, image_to_text, save_text_to_file)


def assistant(state: AgentState):
    """The main brain that decides whether to use a tool or reply."""
    return {"messages": [llm.invoke(state["messages"])]}

# --- 4. Build the Graph ---
builder = StateGraph(AgentState)

# Add processing nodes
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode([pdf_to_base64]))

# Define the flow
builder.add_edge(START, "assistant")
# If the assistant wants to call a tool, go to 'tools', else end
builder.add_conditional_edges("assistant", tools_condition)
# After tool finishes, go back to assistant to summarize/analyze the result
builder.add_edge("tools", "assistant")

graph = builder.compile()

# --- 5. Run the Agent ---
# The agent will see the path, call the tool, and then analyze the base64 output
inputs = {"messages": [HumanMessage(content="Extract the text from 'invoice.pdf'")]}
for chunk in graph.stream(inputs, stream_mode="values"):
    chunk["messages"][-1].pretty_print()