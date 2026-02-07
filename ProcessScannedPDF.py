import base64
from datetime import datetime, timezone
from importlib.metadata import metadata
import os
from io import BytesIO
from zoneinfo import ZoneInfo
from pdf2image import convert_from_path
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import json
from pathlib import Path

# Convert PDF to images
def pdf_to_images(pdf_path):
    """Converts pdf files to base64 image strings"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    
    # Convert first page to image
    pages = convert_from_path(pdf_path, 300)
    images =[]
    for image in pages:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images.append(img_str)

    return images

def read_image(image_str):
    """Reads the image using ChatOllama and returns the text"""
    # Initialize the ChatOllama model
    llm = ChatOllama(model="qwen3-vl:32b", temperature=0)

    # Define messages for the multimodal model
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

    # Invoke the model
    response = llm.invoke(messages)
    
    print("\n\nExtracted Text: ")
    print(response.content)
    
    filepath = Path("extracted_text.txt")    
    with filepath.open("a", encoding="utf-8") as f:
        f.write(response.content)
    
    
    print(f"\n\n**Response Metadata:**\n")
    format_response_metadata(response.response_metadata)


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

    


pdf_path = "2025-26-westchester-rgb-explanatory-statement.pdf"
all_img_data = pdf_to_images(pdf_path)

for img_data in all_img_data[5:6]:
    read_image(img_data)