"""
Shared Gemini analysis logic — used by both drive_watcher and gemini_vision CLI.
"""

import mimetypes
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import logfire
from google import genai
from google.genai import types
from pydantic import BaseModel

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")

IMAGE_MIMES = {"image/png", "image/jpeg", "image/webp", "image/gif", "image/heic", "image/heif"}
VIDEO_MIMES = {"video/mp4", "video/avi", "video/quicktime", "video/x-matroska", "video/webm", "video/x-flv", "video/mpeg", "video/3gpp"}
AUDIO_MIMES = {"audio/wav", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac", "audio/x-flac", "audio/mp4", "audio/m4a"}

EXTENSION_MIME_MAP = {
    ".m4a": "audio/mp4",
    ".m4v": "video/mp4",
    ".mkv": "video/x-matroska",
    ".flac": "audio/flac",
    ".heic": "image/heic",
    ".heif": "image/heif",
}


class TimeBlock(BaseModel):
    start: str
    end: str
    text: str


class VideoEvent(BaseModel):
    start: str
    end: str
    description: str


class MediaAnalysis(BaseModel):
    transcription: list[TimeBlock]
    events: list[VideoEvent]
    summary: str
    action_items: list[str]
    insights: list[str]


def get_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        mime = EXTENSION_MIME_MAP.get(path.suffix.lower())
    if not mime:
        raise ValueError(f"Cannot determine MIME type for: {path.name}")
    return mime


def get_media_category(mime: str) -> str:
    if mime in IMAGE_MIMES:
        return "image"
    if mime in VIDEO_MIMES:
        return "video"
    if mime in AUDIO_MIMES:
        return "audio"
    raise ValueError(f"Unsupported MIME type: {mime}")


def safe_upload_path(path: Path) -> Path:
    safe_name = path.name.encode("ascii", errors="ignore").decode()
    if safe_name == path.name:
        return path
    tmp = Path(tempfile.mkdtemp()) / (safe_name or "upload" + path.suffix)
    shutil.copy2(path, tmp)
    return tmp


def build_prompt(category: str, user_prompt: str | None) -> str:
    base = {
        "video": (
            "Analyze this video thoroughly. "
            "Produce a word-for-word transcription with timestamps, "
            "a step-by-step breakdown of what is happening on screen with timestamps, "
            "a concise summary, concrete action items, and key insights."
        ),
        "audio": (
            "Analyze this audio thoroughly. "
            "Produce a word-for-word transcription with timestamps, "
            "a step-by-step breakdown of what is discussed with timestamps, "
            "a concise summary, concrete action items, and key insights. "
            "Leave the events list empty."
        ),
        "image": (
            "Analyze this image thoroughly. "
            "Describe everything visible in detail. "
            "Provide a summary, any action items implied by the content, and key insights. "
            "Leave transcription and events empty."
        ),
    }[category]
    return f"{user_prompt}\n\n{base}" if user_prompt else base


def print_analysis(analysis: MediaAnalysis) -> None:
    if analysis.transcription:
        print("\n=== TRANSCRIPTION ===")
        for block in analysis.transcription:
            print(f"[{block.start} → {block.end}] {block.text}")

    if analysis.events:
        print("\n=== VIDEO EVENTS ===")
        for event in analysis.events:
            print(f"[{event.start} → {event.end}] {event.description}")

    print("\n=== SUMMARY ===")
    print(analysis.summary)

    if analysis.action_items:
        print("\n=== ACTION ITEMS ===")
        for item in analysis.action_items:
            print(f"• {item}")

    if analysis.insights:
        print("\n=== INSIGHTS ===")
        for insight in analysis.insights:
            print(f"• {insight}")


def analyze(file_path: str, user_prompt: str | None) -> MediaAnalysis:
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    mime = get_mime_type(path)
    category = get_media_category(mime)
    prompt = build_prompt(category, user_prompt)

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
        response_mime_type="application/json",
        response_schema=MediaAnalysis,
    )

    with logfire.span("gemini.analyze", file=path.name, mime=mime, category=category, model=MODEL):
        if category == "image" and path.stat().st_size < 20 * 1024 * 1024:
            data = path.read_bytes()
            response = client.models.generate_content(
                model=MODEL,
                contents=[types.Part.from_bytes(data=data, mime_type=mime), prompt],
                config=config,
            )
            return MediaAnalysis.model_validate_json(response.text)

        upload_path = safe_upload_path(path)
        print(f"[uploading {path.name} ({path.stat().st_size // 1024}KB)...]", flush=True)
        with logfire.span("file_api.upload"):
            uploaded = client.files.upload(file=str(upload_path))

        if category == "video":
            print("[waiting for processing...]", flush=True)
            with logfire.span("file_api.wait"):
                while uploaded.state == "PROCESSING":
                    time.sleep(2)
                    uploaded = client.files.get(name=uploaded.name)

        with logfire.span("generate_content"):
            response = client.models.generate_content(
                model=MODEL,
                contents=[uploaded, prompt],
                config=config,
            )
        return MediaAnalysis.model_validate_json(response.text)
