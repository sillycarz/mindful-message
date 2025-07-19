README.md
+39
-0

# MindfulMessage

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/github.com/sillycarz/mindful-message)

A Discord bot that encourages mindful communication by prompting users to reflect before sending messages that may be harmful.

## Installation

1. Clone this repository.
2. Install the required packages:

```bash
pip install -r requirements-light.txt
```

## Configuration

Create a `.env` file (you can copy `hm.env.example`) or set the required environment variables. At minimum you must provide your Discord bot token:

```bash
DISCORD_TOKEN=your_token_here
```

Additional settings such as `REFLECTION_DELAY_SECONDS` or `DEFAULT_LANGUAGE` can also be configured in the `.env` file.

## Usage

Start the bot with:

```bash
python bot.py
```

### Available Commands

- `!settings` – view or change personal preferences (`!settings reflection on/off`, `!settings language <code>`)
- `!stats` – display weekly statistics
- `!report` – generate an analytics report

Enjoy a kinder Discord community!
