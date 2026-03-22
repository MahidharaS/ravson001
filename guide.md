# Beginner Guide

This file is for someone who does **not** know coding.

It explains exactly what to do next, in the simplest possible order.

## What This Project Does

This project is a bot.

It can do 2 main things:

1. Answer questions from a small set of documents
2. Describe an image and give 3 tags

It supports these commands:

- `/ask <your question>`
- `/image`
- `/help`
- `/summarize`

## Before You Start

You are on a Mac if you are using this folder as-is.

You will use the **Terminal** app.

If you have never used Terminal before:

1. Press `Command + Space`
2. Type `Terminal`
3. Press `Enter`

## The Folder You Should Be In

This project is in this folder:

`/Users/mahidharas/Desktop/project-rag-vision-bot`

In Terminal, go into the project folder with this command:

```bash
cd /Users/mahidharas/Desktop/project-rag-vision-bot
```

## The Safest First Step

Do **not** start with Telegram.

Start with the local test mode first. It is much easier.

This lets you check that the bot works before connecting it to Telegram.

## Step 1: Check Python

Copy and run this:

```bash
python3 --version
```

If you see something like `Python 3.x.x`, that is good.

If you get an error, install Python 3 first from:

`https://www.python.org/downloads/`

Then come back and run the command again.

## Step 2: Run the Bot in Local Test Mode

Inside the project folder, run:

```bash
python3 app.py --cli
```

You should see help text and a prompt where you can type commands.

## Step 3: Test the Question Feature

After the bot starts, type this:

```text
/ask What is the leave carry-forward policy?
```

You should get an answer from the local documents.

Then try:

```text
/ask Which expenses need manager approval?
```

Then try:

```text
/summarize
```

If that works, the project is working in local mode.

## Step 4: Stop the Local Test Mode

Type:

```text
exit
```

or press:

`Control + C`

## What You Have Proven So Far

If the steps above worked, then:

1. Python is installed
2. The project files are okay
3. The question-answering part is working
4. The summary command is working

That is the best first checkpoint.

## Step 5: Decide What You Want Next

You now have 2 choices:

1. Use the project only in local test mode
2. Connect it to Telegram so it works like a real chat bot

If you want the full assignment result, continue with Telegram.

## Step 6: Create a Telegram Bot

Open the Telegram app on your phone or desktop.

Then do this:

1. Search for `BotFather`
2. Open the official `BotFather` chat
3. Type:

```text
/newbot
```

4. Follow the instructions
5. Choose a bot name
6. Choose a bot username
   The username usually ends with `bot`
7. BotFather will give you a **bot token**

That token is very important.

It looks something like a long secret password.

Do **not** share it with anyone.

## Step 7: Put Your Telegram Bot Token Into The `.env` File

You do **not** need to type long `export` commands.

This project now has a file named `.env` in the project folder.

That file already contains the correct settings.

You only need to replace this line:

```text
TELEGRAM_BOT_TOKEN=PASTE_YOUR_BOTFATHER_TOKEN_HERE
```

with your real token from BotFather.

Simple way to do that on Mac:

1. Open the project folder in Finder
2. If you do not see `.env`, press `Command + Shift + .` to show hidden files
3. Open the `.env` file in a text editor
4. Find the line that starts with `TELEGRAM_BOT_TOKEN=`
5. Paste your real token after the `=`
6. Save the file

Important:

- Do not add extra spaces
- Do not share your token with anyone
- After you save `.env`, the project will remember it for the next run

## Step 8: Start the Telegram Bot

Run:

```bash
python3 app.py
```

If it starts correctly, it will wait for Telegram messages.

## Step 9: Open Your Bot in Telegram

In Telegram:

1. Search for the bot username you created
2. Open the chat
3. Press `Start` or send:

```text
/help
```

Then test:

```text
/ask What is the leave carry-forward policy?
```

Then test:

```text
/ask What purchases need manager approval?
```

## Step 10: Use the Image Feature

There are 2 ways image mode can work:

1. You send `/image`, then upload a picture
2. You upload a picture directly if auto-image mode is active

But image description needs an image model backend.

Right now, the easiest built-in path is **Ollama**.

## Step 10.5: Add Your Own Documents For RAG

If you want the bot to answer questions about your own materials, place those files inside the `knowledge_base` folder.

This build supports:

- `.md`
- `.txt`
- `.json`

You can now organize them inside subfolders too.

Example:

```text
knowledge_base/
company/
company/about.md
company/team_values.md
company/candidate_profiles/your_resume.md
sales/customer_notes.txt
```

Important:

- Telegram document uploads are still not added into RAG automatically
- If your file is a PDF or DOCX, convert it to `.md` or `.txt` first
- After adding or editing files, stop the bot and start it again

## Important Truth About Image Mode

The project code is ready for image description.

But image description will only work if you also install and run a model backend.

The recommended backend in this project is **Ollama**.

Without Ollama, the bot can still run, but image description will say that the vision backend is unavailable.

That is normal.

## Step 11: Ollama Is Already Installed

Ollama has already been installed on this Mac for this project.

The required models are also already downloaded:

- `llama3.2:3b` for text answers
- `nomic-embed-text` for document search
- `llava:7b` for image description

So you do **not** need to install or pull them again right now.

## Step 12: Start The Bot

After your token is saved in `.env`, run:

```bash
python3 app.py
```

That starts the Telegram bot using the saved token and the local Ollama models.

## Step 13: Test the Full Bot

In Telegram, test these:

```text
/help
```

```text
/ask What is the leave carry-forward policy?
```

```text
/summarize
```

Then test image mode:

```text
/image
```

After that, upload a photo.

If everything is working, the bot should send back:

1. A caption
2. 3 tags

## Step 14: If You Want a Quick Health Check

In Terminal, inside the project folder, run:

```bash
python3 app.py --doctor
```

This prints a status report.

Use this when you want to check:

1. Which knowledge-base files were loaded
2. Which providers are active
3. Whether the project sees Ollama or not

## Step 15: If Something Fails

Here is the simple troubleshooting list.

### Problem: `python3: command not found`

Fix:

Install Python 3 from:

`https://www.python.org/downloads/`

### Problem: The bot works in CLI mode but not in Telegram

Fix:

1. Open the `.env` file
2. Make sure `TELEGRAM_BOT_TOKEN=` contains your real token
3. Make sure you saved the file
4. Run:

```bash
python3 app.py
```

### Problem: `/ask` works badly or gives fallback-style answers

Fix:

That usually means the project is using the built-in fallback mode instead of Ollama.

Run:

```bash
python3 app.py --doctor
```

Check the `.env` file and make sure these lines exist:

```text
EMBEDDING_PROVIDER=ollama
LLM_PROVIDER=ollama
VISION_PROVIDER=ollama
```

If those lines are correct, open the Ollama app and then run the doctor command again.

### Problem: `/image` says vision backend is unavailable

Fix:

That means Ollama is not installed, not running, or the `llava` model is missing.

Run these:

```bash
python3 app.py --doctor
```

If the doctor report says Ollama is unavailable, open the Ollama app and then start the bot again.

### Problem: Nothing happens in Telegram

Fix:

1. Make sure the bot process is still running in Terminal
2. Make sure you used the correct token
3. Send `/help` to your bot again
4. If needed, stop the bot with `Control + C` and start it again

## The Easiest Exact Path To Follow

If you want the shortest possible version, do exactly this:

1. Open Terminal
2. Run:

```bash
cd /Users/mahidharas/Desktop/project-rag-vision-bot
python3 app.py --cli
```

3. Test:

```text
/ask What is the leave carry-forward policy?
```

4. Exit
5. Create a Telegram bot with BotFather
6. Put your BotFather token into the `.env` file
7. In Terminal run:

```bash
python3 app.py
```

8. Test `/help` in Telegram
9. Test `/ask`, `/summarize`, and `/image`

## If You Want To Know Which File Matters Most

If you only care about running the project, the most important file is:

`app.py`

The most important guide files are:

- `guide.md`
- `README.md`
- `.env.example`

## Final Advice

Do this in order:

1. First make local CLI mode work
2. Then add your Telegram token to `.env`
3. Then start Telegram mode
4. Then test the image feature

That is the cleanest path with the fewest mistakes.
