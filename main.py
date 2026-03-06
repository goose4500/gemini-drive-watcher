"""
Gemini Drive Watcher — polls Google Drive "GeminiInbox" folder for new media,
analyzes with Gemini Vision, and uploads results to "GeminiResults" folder.

Deploy on Railway. Required env vars:
    GOOGLE_API_KEY              — AI Studio key
    GOOGLE_SERVICE_ACCOUNT_JSON — full JSON string of service account credentials

Share your "GeminiInbox" Drive folder with the service account email.
Results appear in "GeminiResults" alongside the original file.
"""

import io
import json
import os
import tempfile
import time
from pathlib import Path

import logfire
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

from analysis import analyze, MediaAnalysis, print_analysis

logfire.configure(send_to_logfire="if-token-present")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "30"))
INBOX_FOLDER_NAME = os.getenv("INBOX_FOLDER", "GeminiInbox")
RESULTS_FOLDER_NAME = os.getenv("RESULTS_FOLDER", "GeminiResults")
PROCESSED_PROPERTY = "gemini_processed"

MEDIA_MIME_PREFIXES = ("image/", "video/", "audio/")
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv", ".mpeg", ".mpg",
    ".mp3", ".m4a", ".wav", ".aac", ".ogg", ".flac", ".aiff",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif",
}


def build_drive_service():
    creds_json = os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]
    creds_info = json.loads(creds_json)
    creds = service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/drive"],
    )
    return build("drive", "v3", credentials=creds)


def get_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]

    folder = service.files().create(body=body, fields="id").execute()
    print(f"[created] Drive folder: {name}", flush=True)
    return folder["id"]


def is_processed(service, file_id: str) -> bool:
    try:
        props = service.files().get(
            fileId=file_id,
            fields="properties"
        ).execute().get("properties", {})
        return props.get(PROCESSED_PROPERTY) == "true"
    except Exception:
        return False


def mark_processed(service, file_id: str) -> None:
    service.files().update(
        fileId=file_id,
        body={"properties": {PROCESSED_PROPERTY: "true"}},
    ).execute()


def download_file(service, file_id: str, dest: Path) -> None:
    request = service.files().get_media(fileId=file_id)
    with open(dest, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def upload_result(service, text: str, filename: str, parent_id: str) -> None:
    content = io.BytesIO(text.encode("utf-8"))
    media = MediaFileUpload.__new__(MediaFileUpload)

    # Upload as plain text
    service.files().create(
        body={"name": filename, "parents": [parent_id]},
        media_body=MediaIoBaseDownload.__new__(MediaIoBaseDownload),
    ).execute()

    # Use a temp file for upload
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
        tmp_path = tmp.name

    media = MediaFileUpload(tmp_path, mimetype="text/plain")
    service.files().create(
        body={"name": filename, "parents": [parent_id]},
        media_body=media,
        fields="id",
    ).execute()
    Path(tmp_path).unlink(missing_ok=True)


def is_media_file(name: str, mime: str) -> bool:
    ext = Path(name).suffix.lower()
    return ext in MEDIA_EXTENSIONS or any(mime.startswith(p) for p in MEDIA_MIME_PREFIXES)


def process_file(service, file_meta: dict, results_folder_id: str) -> None:
    file_id = file_meta["id"]
    name = file_meta["name"]
    mime = file_meta.get("mimeType", "")

    print(f"[processing] {name}", flush=True)

    with logfire.span("drive_watcher.process", file=name, mime=mime):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dest = Path(tmp_dir) / name
            with logfire.span("drive.download", file=name):
                download_file(service, file_id, dest)

            analysis: MediaAnalysis = analyze(str(dest), None)

        result_text = format_analysis(analysis)
        result_filename = f"{Path(name).stem}.gemini.txt"

        with logfire.span("drive.upload_result", filename=result_filename):
            upload_result(service, result_text, result_filename, results_folder_id)

        mark_processed(service, file_id)
        print(f"[done] {name} → {result_filename}", flush=True)


def format_analysis(analysis: MediaAnalysis) -> str:
    lines = []

    if analysis.transcription:
        lines.append("=== TRANSCRIPTION ===")
        for block in analysis.transcription:
            lines.append(f"[{block.start} → {block.end}] {block.text}")
        lines.append("")

    if analysis.events:
        lines.append("=== VIDEO EVENTS ===")
        for event in analysis.events:
            lines.append(f"[{event.start} → {event.end}] {event.description}")
        lines.append("")

    lines.append("=== SUMMARY ===")
    lines.append(analysis.summary)
    lines.append("")

    if analysis.action_items:
        lines.append("=== ACTION ITEMS ===")
        for item in analysis.action_items:
            lines.append(f"• {item}")
        lines.append("")

    if analysis.insights:
        lines.append("=== INSIGHTS ===")
        for insight in analysis.insights:
            lines.append(f"• {insight}")

    return "\n".join(lines)


def poll_loop(service) -> None:
    inbox_id = get_or_create_folder(service, INBOX_FOLDER_NAME)
    results_id = get_or_create_folder(service, RESULTS_FOLDER_NAME)

    print(f"[ready] Watching '{INBOX_FOLDER_NAME}' (id: {inbox_id})", flush=True)
    print(f"[ready] Results → '{RESULTS_FOLDER_NAME}' (id: {results_id})", flush=True)
    print(f"[polling] Every {POLL_INTERVAL}s\n", flush=True)

    while True:
        try:
            results = service.files().list(
                q=f"'{inbox_id}' in parents and trashed=false",
                fields="files(id, name, mimeType)",
                orderBy="createdTime",
            ).execute()

            files = results.get("files", [])

            for f in files:
                if not is_media_file(f["name"], f.get("mimeType", "")):
                    continue
                if is_processed(service, f["id"]):
                    continue
                process_file(service, f, results_id)

        except Exception as e:
            logfire.error("poll error", error=str(e))
            print(f"[error] {e}", flush=True)

        time.sleep(POLL_INTERVAL)


def main() -> None:
    required = ["GOOGLE_API_KEY", "GOOGLE_SERVICE_ACCOUNT_JSON"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")

    service = build_drive_service()
    poll_loop(service)


if __name__ == "__main__":
    main()
