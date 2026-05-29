# Mr Prompter — batch watermark / overlay removal bot

A functional Telegram bot. Send it a batch of images and it returns the same
images with watermarks, logos, text overlays, captions and labels removed —
each image processed independently. No prompt or instructions required.

**Current path for friends (recommended):** Use the standalone Gemini 2.5 Flash Image tool below. No user keys, no 25-image limit, full fidelity.

```mermaid
flowchart LR
    U([User]) -- sends images --> B[Bot]
    B -- free tier? --> Q{free_used < 25?}
    Q -- yes --> H[Host OpenRouter key]
    Q -- no --> K[User's own key /setup]
    H & K -- image + fixed instruction --> OR[OpenRouter image model]
    OR -- cleaned image --> B
    B -- one cleaned image per input --> U
```

## How it works

1. A user sends one or more images (as photos, or as files for best quality).
2. Each image is processed independently and the cleaned version is sent back —
   so a batch of N images yields N cleaned images.
3. The first **25 images per user** are processed on the host's shared
   OpenRouter key (`HOST_OPENROUTER_KEY`). After that, the user runs `/setup`
   to add their own OpenRouter key and continues with no limit.

A single fixed instruction drives every call (see `REMOVAL_INSTRUCTION` in
`src/engine.py`); the user never writes a prompt. Any image caption is ignored.

## Why an image-output model

Removing a watermark means *returning a new, edited image* — not describing one.
A plain text/vision LLM can only read an image, so this bot uses OpenRouter
**image-output (editing) models**.

Current shortlist (in `src/config.py` and `MODEL_SHORTLIST` env var) contains
the models we are actively evaluating for watermark/overlay removal quality.

**Cost-effective recommendations** (as of late 2026 research):
- `black-forest-labs/flux.2-klein-4b` — Currently one of the cheapest strong options.
- `sourceful/riverflow-v2-fast` — Excellent speed/price balance.
- `recraft/recraft-v4.1-utility` — Best control via `image_config.strength` for conservative edits that don't destroy real signage.

The more expensive models (Gemini Pro previews, GPT-5.4 Image, Grok Imagine, FLUX.2 Pro/Max) can be used when higher quality is required.

**Important**: Real usage requires empirical testing. Only image-output / image-editing models are valid in the shortlist.

A future lower-cost architecture may use a two-stage pipeline (cheap detection model → precise masked inpainting) instead of full image-to-image on every request.

## Free tier

- Counted **per user, for the lifetime of their record** (`free_used` column).
- A slot is reserved atomically *before* processing (`claim_free_slot`), so a
  batch processed concurrently can never exceed the cap.
- If an image fails, the slot is **refunded** (`release_free_slot`) — failures
  don't cost the user.
- If `HOST_OPENROUTER_KEY` is unset, the free tier is disabled and users must
  add their own key first.

## Commands

| Command   | What it does                                   |
|-----------|------------------------------------------------|
| `/start`  | What the bot does                              |
| `/status` | Free images remaining / current model          |
| `/setup`  | Add your own OpenRouter key, then pick a model |
| `/model`  | Change the AI model                            |
| `/forget` | Delete your stored key, model, and usage count |
| `/cancel` | Cancel the current operation                   |

## Security

- User API keys are **Fernet-encrypted at rest**. The master key lives in
  `data/secret.key` (mode 0400), separate from `.env`, and the SQLite file is
  restricted to `0600`.
- Pasted keys are deleted from the chat on a best-effort basis, and logs redact
  anything matching an OpenRouter/OpenAI key pattern.

## Quick start (Gemini 2.5 free tool — current recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your GEMINI_API_KEY (get from https://aistudio.google.com/app/apikey)
python scripts/gemini_25_free_watermark_remover.py /path/to/images/
```

The old Telegram bot (OpenRouter-based) is kept as dead code for now.

For the legacy Telegram bot: see "Running in Production" below.

## Running in Production

### Using the run script (recommended)

A convenience script is provided:

```bash
chmod +x run_bot.sh
./run_bot.sh
```

This script:
- Activates the virtual environment (if `.venv` or `venv` exists)
- Loads variables from `.env`
- Starts the bot

### Systemd autostart (recommended for servers)

1. Copy the example service file:

   ```bash
   sudo cp deploy/mr-prompter-bot.service /etc/systemd/system/
   ```

2. Edit it to match your setup (especially `User` and `WorkingDirectory`):

   ```bash
   sudo nano /etc/systemd/system/mr-prompter-bot.service
   ```

3. Enable and start:

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable mr-prompter-bot
   sudo systemctl start mr-prompter-bot
   ```

4. Useful commands:

   ```bash
   sudo systemctl status mr-prompter-bot
   sudo journalctl -u mr-prompter-bot -f
   sudo systemctl restart mr-prompter-bot
   ```

The service is configured to restart automatically on failure.

### Error Telemetry

- Gemini 2.5 free tool: `data/logs/gemini_free_errors.log`
- Legacy Telegram bot: `data/logs/errors.log`

Both use rotating file handlers.

### Running the Gemini 2.5 free tool

```bash
# One-shot
python scripts/gemini_25_free_watermark_remover.py photo.jpg folder/ --prompt-file prompts/conservative-watermark-removal.txt

# With the run script (loads .env)
./run_bot.sh --gemini /path/to/images/
# (edit run_bot.sh if you want it to default to Gemini mode)
```

For systemd, point ExecStart at a small wrapper that calls the Gemini script (or run it manually / via cron).

### Running the legacy Telegram bot (OpenRouter)

Use the sections below (run_bot.sh + systemd still point at `src.main`).

## Stack

- Python 3.12+, `python-telegram-bot` 21
- `httpx` for the OpenRouter image-edit calls
- `aiosqlite` for async SQLite, `cryptography` (Fernet) for key encryption
- Default model: `google/gemini-2.5-flash-image` (overridable via env)

## Tests

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

Covers the database layer (including a concurrency regression proving the free
tier can't be over-granted), the OpenRouter engine (success, missing-image, HTTP
and network errors), config parsing, and the handler routing (free-tier
exhaustion, own-key unlimited, failure refund, no-host-key).

