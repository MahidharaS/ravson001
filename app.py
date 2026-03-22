from __future__ import annotations

import argparse
import json
import logging

from rag_vision_bot.cli import CliRunner
from rag_vision_bot.config import load_settings
from rag_vision_bot.services import BotService
from rag_vision_bot.telegram_bot import TelegramBotRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid Telegram RAG + vision bot")
    parser.add_argument("--cli", action="store_true", help="Run the local CLI instead of Telegram polling")
    parser.add_argument("--doctor", action="store_true", help="Print runtime health information and exit")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    args = parse_args()
    settings = load_settings()
    if args.cli:
        settings.transport_mode = "cli"

    service = BotService(settings)
    service.prepare()

    if args.doctor:
        print(json.dumps(service.health_report(), indent=2))
        return

    if settings.transport_mode == "telegram":
        runner = TelegramBotRunner(settings, service)
        runner.run_forever()
        return

    runner = CliRunner(service)
    runner.run()


if __name__ == "__main__":
    main()
