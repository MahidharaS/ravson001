from __future__ import annotations

import logging
import time
from typing import Any
from urllib import error, request

from .config import Settings
from .services import BotService


LOGGER = logging.getLogger(__name__)


PLACEHOLDER_TELEGRAM_TOKENS = {
    "PASTE_YOUR_BOTFATHER_TOKEN_HERE",
    "PASTE_YOUR_REAL_TOKEN_HERE",
    "your-token",
}

MAIN_MENU_COMPANY = "Company"
MAIN_MENU_PROFILE = "Profile"
MAIN_MENU_BACK = "Back to Menu"

COMPANY_SUGGESTED_QUESTIONS = [
    "What does this company do?",
    "What products and solutions does this company offer?",
    "Who are this company's customers and clients?",
    "What is the strategic outlook for this company?",
]

PROFILE_SUGGESTED_QUESTIONS = [
    "Tell me about this profile.",
    "What are this candidate's core strengths?",
    "What projects or achievements stand out?",
    "Why is this candidate a strong fit for the company?",
]


def validate_telegram_token(token: str | None) -> str:
    normalized = (token or "").strip()
    if not normalized:
        raise ValueError("TELEGRAM_BOT_TOKEN is required for Telegram mode")
    if normalized in PLACEHOLDER_TELEGRAM_TOKENS:
        raise ValueError(
            "Set TELEGRAM_BOT_TOKEN in .env to the real token from BotFather before starting Telegram mode"
        )
    return normalized


class TelegramHttpClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.token = validate_telegram_token(settings.telegram_bot_token)
        self.api_base = settings.telegram_api_base.rstrip("/")

    def _method_url(self, method: str) -> str:
        return f"{self.api_base}/bot{self.token}/{method}"

    def _file_url(self, file_path: str) -> str:
        return f"{self.api_base}/file/bot{self.token}/{file_path}"

    def _request_json(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = request.Request(
            self._method_url(method),
            data=json_bytes(payload),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.settings.request_timeout_seconds) as response:
                return parse_telegram_response(response.read().decode("utf-8"))
        except error.URLError as exc:  # pragma: no cover - depends on network runtime
            raise RuntimeError(f"Telegram API error for {method}: {exc}") from exc

    def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": self.settings.telegram_poll_timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        response = self._request_json("getUpdates", payload)
        return response.get("result", [])

    def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to_message_id: int | None = None,
        message_thread_id: int | None = None,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self._request_json("sendMessage", payload)

    def send_chat_action(
        self,
        chat_id: str,
        action: str,
        message_thread_id: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "action": action}
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        self._request_json("sendChatAction", payload)

    def get_file(self, file_id: str) -> dict[str, Any]:
        response = self._request_json("getFile", {"file_id": file_id})
        return response["result"]

    def download_file(self, file_path: str) -> bytes:
        req = request.Request(self._file_url(file_path), method="GET")
        try:
            with request.urlopen(req, timeout=self.settings.request_timeout_seconds) as response:
                return response.read()
        except error.URLError as exc:  # pragma: no cover - depends on network runtime
            raise RuntimeError(f"Telegram file download failed: {exc}") from exc


def json_bytes(payload: dict[str, Any]) -> bytes:
    import json

    return json.dumps(payload).encode("utf-8")


def parse_telegram_response(raw: str) -> dict[str, Any]:
    import json

    data = json.loads(raw)
    if not data.get("ok"):
        description = data.get("description", "Unknown Telegram error")
        raise RuntimeError(description)
    return data


def extract_command(text: str | None, entities: list[dict[str, Any]] | None) -> tuple[str | None, str]:
    if not text or not entities:
        return None, ""
    first = entities[0]
    if first.get("type") != "bot_command" or first.get("offset") != 0:
        return None, ""
    length = int(first.get("length", 0))
    token = text[:length]
    command = token.split("@", 1)[0]
    return command, text[length:].strip()


def extract_image_attachment(message: dict[str, Any]) -> tuple[str, str | None, int | None]:
    if message.get("photo"):
        photo = message["photo"][-1]
        return photo["file_id"], "image/jpeg", photo.get("file_size")
    document = message.get("document")
    if document and str(document.get("mime_type", "")).startswith("image/"):
        return document["file_id"], document.get("mime_type"), document.get("file_size")
    raise ValueError("No image attachment found")


def has_non_image_document(message: dict[str, Any]) -> bool:
    document = message.get("document")
    if not document:
        return False
    return not str(document.get("mime_type", "")).startswith("image/")


def normalize_text(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def scope_from_menu_text(text: str | None) -> str | None:
    normalized = normalize_text(text)
    if normalized == normalize_text(MAIN_MENU_COMPANY):
        return "company"
    if normalized == normalize_text(MAIN_MENU_PROFILE):
        return "profile"
    return None


def source_prefixes_for_scope(scope: str | None) -> list[str]:
    if scope == "company":
        return ["company/"]
    if scope == "profile":
        return [
            "my_profile/",
            "my profile/",
            "profile/",
            "candidate_profiles/",
            "company/candidate_profiles/",
        ]
    return []


def suggested_questions_for_scope(scope: str | None) -> list[str]:
    if scope == "company":
        return COMPANY_SUGGESTED_QUESTIONS
    if scope == "profile":
        return PROFILE_SUGGESTED_QUESTIONS
    return []


def reply_keyboard(rows: list[list[str]]) -> dict[str, Any]:
    return {
        "keyboard": [[{"text": label} for label in row] for row in rows],
        "resize_keyboard": True,
        "is_persistent": True,
    }


def menu_keyboard() -> dict[str, Any]:
    return reply_keyboard([[MAIN_MENU_COMPANY, MAIN_MENU_PROFILE]])


def scope_keyboard(scope: str | None) -> dict[str, Any]:
    suggestions = suggested_questions_for_scope(scope)
    rows = [[question] for question in suggestions]
    rows.append([MAIN_MENU_BACK])
    return reply_keyboard(rows)


def welcome_text() -> str:
    return (
        "Choose a focus area to begin.\n\n"
        "Company: ask about the target company.\n"
        "Profile: ask about your background, strengths, and fit.\n\n"
        "After you choose one, I will show suggested questions and keep answers focused on that area."
    )


def scope_intro_text(scope: str) -> str:
    questions = suggested_questions_for_scope(scope)
    heading = "Company mode is active." if scope == "company" else "Profile mode is active."
    lines = [heading, "", "Suggested questions:"]
    lines.extend(f"- {question}" for question in questions)
    lines.extend(["", "You can tap one of these or type your own question directly."])
    return "\n".join(lines)


class TelegramBotRunner:
    def __init__(self, settings: Settings, service: BotService) -> None:
        self.settings = settings
        self.service = service
        self.client = TelegramHttpClient(settings)
        self.offset: int | None = None

    def run_forever(self) -> None:
        LOGGER.info("Starting Telegram polling bot")
        while True:
            try:
                updates = self.client.get_updates(self.offset)
                for update in updates:
                    self.handle_update(update)
                    self.offset = int(update["update_id"]) + 1
            except Exception as exc:  # pragma: no cover - depends on runtime network
                LOGGER.exception("Telegram polling error: %s", exc)
                time.sleep(2)

    def handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not message:
            return
        chat_id = str(message["chat"]["id"])
        user_id = str(message.get("from", {}).get("id", chat_id))
        message_id = message.get("message_id")
        thread_id = message.get("message_thread_id")
        text = (message.get("text") or "").strip()

        command, args = extract_command(message.get("text"), message.get("entities"))
        if command is None and message.get("caption"):
            command, args = extract_command(message.get("caption"), message.get("caption_entities"))

        has_image = bool(message.get("photo")) or bool(
            message.get("document")
            and str(message["document"].get("mime_type", "")).startswith("image/")
        )
        has_document = has_non_image_document(message)
        selected_scope = self.service.get_query_scope(user_id, chat_id)
        scope_choice = scope_from_menu_text(text)

        try:
            if command in {"/help", "/start"}:
                self.service.set_query_scope(user_id, chat_id, None)
                self.client.send_message(
                    chat_id=chat_id,
                    text=f"{welcome_text()}\n\n{self.service.help_text()}",
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=menu_keyboard(),
                )
                return

            if normalize_text(text) == normalize_text(MAIN_MENU_BACK):
                self.service.set_query_scope(user_id, chat_id, None)
                self.client.send_message(
                    chat_id=chat_id,
                    text=welcome_text(),
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=menu_keyboard(),
                )
                return

            if scope_choice is not None:
                self.service.set_query_scope(user_id, chat_id, scope_choice)
                self.client.send_message(
                    chat_id=chat_id,
                    text=scope_intro_text(scope_choice),
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(scope_choice),
                )
                return

            if command == "/ask":
                self.client.send_chat_action(chat_id, "typing", thread_id)
                selected_scope = self.service.get_query_scope(user_id, chat_id)
                response = self.service.ask(
                    user_id=user_id,
                    chat_id=chat_id,
                    query=args,
                    source_prefixes=source_prefixes_for_scope(selected_scope),
                )
                self.client.send_message(
                    chat_id=chat_id,
                    text=response.answer,
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
                )
                return

            if command == "/summarize":
                self.client.send_chat_action(chat_id, "typing", thread_id)
                response = self.service.summarize(user_id=user_id, chat_id=chat_id)
                self.client.send_message(
                    chat_id=chat_id,
                    text=response.summary,
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
                )
                return

            if command == "/image" and not has_image:
                self.service.mark_waiting_for_image(user_id=user_id, chat_id=chat_id)
                self.client.send_message(
                    chat_id=chat_id,
                    text="Upload an image next and I will describe it with a caption and 3 tags.",
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
                )
                return

            if has_image and (
                command == "/image"
                or self.service.is_waiting_for_image(user_id=user_id, chat_id=chat_id)
                or self.settings.auto_caption_photos
            ):
                file_id, mime_type, file_size = extract_image_attachment(message)
                if file_size and file_size > self.settings.max_image_bytes:
                    self.client.send_message(
                        chat_id=chat_id,
                        text="The image is too large for this bot. Try a file under 20 MB.",
                        reply_to_message_id=message_id,
                        message_thread_id=thread_id,
                    )
                    return
                self.client.send_chat_action(chat_id, "typing", thread_id)
                file_info = self.client.get_file(file_id)
                image_bytes = self.client.download_file(file_info["file_path"])
                response = self.service.describe_image(
                    user_id=user_id,
                    chat_id=chat_id,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                )
                self.client.send_message(
                    chat_id=chat_id,
                    text=self.service.format_image_response(response),
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
                )
                return

            if has_document:
                self.client.send_message(
                    chat_id=chat_id,
                    text=(
                        "Document uploads are not added to RAG automatically yet. "
                        "To ask about a document, place its text in the local knowledge_base folder "
                        "as a .md, .txt, or .json file, then restart the bot. "
                        "PDF and DOCX Telegram uploads are not indexed by this build."
                    ),
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
                )
                return

            if text and selected_scope is not None:
                self.client.send_chat_action(chat_id, "typing", thread_id)
                response = self.service.ask(
                    user_id=user_id,
                    chat_id=chat_id,
                    query=text,
                    source_prefixes=source_prefixes_for_scope(selected_scope),
                )
                self.client.send_message(
                    chat_id=chat_id,
                    text=response.answer,
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=scope_keyboard(selected_scope),
                )
                return

            if message.get("text"):
                self.client.send_message(
                    chat_id=chat_id,
                    text="Choose Company or Profile to begin, or use /help to see the available commands.",
                    reply_to_message_id=message_id,
                    message_thread_id=thread_id,
                    reply_markup=menu_keyboard(),
                )
        except Exception as exc:
            LOGGER.exception("Failed to handle Telegram update: %s", exc)
            self.client.send_message(
                chat_id=chat_id,
                text="Something went wrong while processing that message.",
                reply_to_message_id=message_id,
                message_thread_id=thread_id,
                reply_markup=scope_keyboard(selected_scope) if selected_scope else menu_keyboard(),
            )
