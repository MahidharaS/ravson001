from __future__ import annotations

import mimetypes
from pathlib import Path

from .services import BotService


class CliRunner:
    def __init__(self, service: BotService) -> None:
        self.service = service

    def run(self) -> None:
        print(self.service.help_text())
        print("Type 'exit' to quit.")
        while True:
            try:
                raw = input("\n> ").strip()
            except EOFError:
                print()
                break
            if not raw:
                continue
            if raw.lower() in {"exit", "quit"}:
                break
            if raw.startswith("/help"):
                print(self.service.help_text())
                continue
            if raw.startswith("/ask"):
                response = self.service.ask("cli-user", "cli-chat", raw[4:].strip())
                print(response.answer)
                continue
            if raw.startswith("/summarize"):
                response = self.service.summarize("cli-user", "cli-chat")
                print(response.summary)
                continue
            if raw.startswith("/image"):
                image_path = raw[6:].strip()
                if not image_path:
                    print("Usage: /image <path-to-image>")
                    continue
                path = Path(image_path)
                if not path.exists():
                    print(f"File not found: {path}")
                    continue
                mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                response = self.service.describe_image(
                    user_id="cli-user",
                    chat_id="cli-chat",
                    image_bytes=path.read_bytes(),
                    mime_type=mime_type,
                )
                print(self.service.format_image_response(response))
                continue
            print("Use /help to see the available commands.")
