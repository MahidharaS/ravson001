# Data Folder Notes

`data/bot.db` is the live runtime SQLite database created by the bot on this machine.

It is not committed to GitHub because it can contain:

- cached answers
- Telegram conversation history
- session state from local testing

`data/bot_public.db` is the public-safe database snapshot for GitHub.

That file is sanitized before publishing so it keeps the indexed knowledge-base structure while removing private runtime tables such as query cache, turn history, and session state.
