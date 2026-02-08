from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from pdf2image import convert_from_path
from io import BytesIO
import base64
from pathlib import Path
import json
from datetime import datetime
from zoneinfo import ZoneInfo

@tool
def pdf_to_images(pdf_path: str) -> list[str]:
    """Convert a scanned PDF into base64 PNG images (one per page)."""

    path = Path(pdf_path)
    if not path.exists() or path.suffix.lower() != ".pdf":
        raise ValueError("Valid PDF path required")

    pages = convert_from_path(path)
    images = []

    for page in pages:
        buf = BytesIO()
        page.save(buf, format="PNG")
        images.append(base64.b64encode(buf.getvalue()).decode())

    return images


@tool
def image_to_text(image_base64: str, page: int) -> str:
    """Extract text from a base64-encoded image and label it with a page number."""

    ocr_llm = ChatOllama(model="qwen3-vl:8b")

    msg = HumanMessage(
        content=[
            {"type": "text", "text": f"Extract all text from page {page}. Keep layout."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )

    resp = ocr_llm.invoke([msg])
    return f"\n\n--- Page {page} ---\n{resp.content}"

@tool
def save_text(text: str, filename: str = "output.txt") -> str:
    """Append extracted text to a file."""

    Path(filename).write_text("", encoding="utf-8") if not Path(filename).exists() else None

    with open(filename, "a", encoding="utf-8") as f:
        f.write(text)

    return f"Saved page text to {filename}"

SYSTEM_PROMPT = """
You are a ReAct agent that extracts text from scanned PDFs.

Rules:
1. ALWAYS use tools.
2. First call pdf_to_images.
3. Then call image_to_text for each page.
4. Save each page using save_text.
5. Stop when all pages are processed.
6. Never fabricate content.
"""

llm = ChatOllama(
    model="gpt-oss:20b",
    temperature=0
).bind_tools([pdf_to_images, image_to_text, save_text])

messages = [
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content="Extract all text from 2025-26-westchester-rgb-explanatory-statement-pages-1.pdf")
]



def format_response_metadata(metadata: dict) -> None:
    """
    Converts LangChain/Ollama response_metadata into a Json‑formatted string with human‑friendly labels,
    focusing on timing and performance information.
    """

    if not metadata:
        print("No response metadata available.")
        return

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

    


tool_map = {
    "pdf_to_images": pdf_to_images,
    "image_to_text": image_to_text,
    "save_text": save_text
}

while True:
    response = llm.invoke(messages)
    messages.append(response)

    if not response.tool_calls:
        print("✅ Done")
        print(response.content)
        format_response_metadata(response.response_metadata)
        break

    for call in response.tool_calls:
        tool = tool_map[call["name"]]
        result = tool.invoke(call["args"])  # ✅ correct
        messages.append(
            HumanMessage(
                content=f"Tool result ({call['name']}): {result}"
            )
        )

