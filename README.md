# claudeg

Run **Claude Code** through your **ChatGPT subscription**.

`claudeg` is a small, fast Rust binary that presents Anthropic's Messages API on `127.0.0.1` and translates each request to ChatGPT's Codex Responses API, forwarded with your ChatGPT (Plus / Pro / Business / Enterprise) OAuth token. Claude Code itself runs unmodified — `claudeg` just sits on loopback and pretends to be Anthropic.

```
┌──────────────┐  HTTPS  ┌──────────────────┐  HTTPS  ┌────────────────────────┐
│ Claude Code  │ ──────▶ │ claudeg :4000    │ ──────▶ │ chatgpt.com            │
│ (claude CLI) │ ◀────── │ Anthropic ⇄ Codex│ ◀────── │ /backend-api/codex/... │
└──────────────┘   SSE   └──────────────────┘   SSE   └────────────────────────┘
```

> **⚠ ToS caveat.** Routing your consumer ChatGPT subscription as the backend for a third-party tool is **not what OpenAI prices it for**. OpenAI may detect this pattern and suspend or ban the account. Use at your own risk on accounts you can afford to lose.

---

## Install

**macOS / Linux:**

```sh
curl -fsSL https://github.com/tuantong/claudeg/releases/latest/download/install.sh | sh
```

**Windows (PowerShell):**

```powershell
iwr -useb https://github.com/tuantong/claudeg/releases/latest/download/install.ps1 | iex
```

The installer downloads the right binary for your OS/arch, drops it on PATH, and runs `claudeg setup` (OAuth login + sentinel approval + auto-start service).

After install:

```sh
claude              # real Anthropic API (unchanged)
claudeg "..."       # routed through your ChatGPT subscription
```

---

## Usage

```text
claudeg                     interactive Claude Code session, routed through the proxy
claudeg "<prompt>"          one-shot — runs `claude` with your prompt
claudeg login               OAuth PKCE login (opens browser)
claudeg logout              forget cached token + stop running proxy
claudeg whoami              "logged in (expires in …)" / "not logged in"
claudeg serve [--port N]    run the proxy in the foreground (advanced)
claudeg setup               register auto-start service + approve sentinel
claudeg uninstall           reverse setup
claudeg --help              full help
```

`claudeg` (alone) auto-starts the background proxy on first call, sets `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`, and execs `claude` — same UX as plain `claude`, but every request goes through your ChatGPT subscription. Anything you pass after `claudeg` (a prompt or any other `claude` flag) is forwarded verbatim. Subsequent calls are instant — the proxy stays warm.

Your normal `claude` is unchanged — it still hits real Anthropic.

---

## Configuration

`claudeg` reads `~/Library/Application Support/claudeg/config.toml` on macOS (`~/.config/claudeg/config.toml` on Linux, `%APPDATA%\claudeg\config.toml` on Windows). All fields optional.

```toml
listen = "127.0.0.1:4000"
default_model = "gpt-5.3-codex"

[models]
# Override the default Anthropic → ChatGPT model mapping.
"claude-opus-4-7"           = "gpt-5.5"          # built-in: gpt-5.5
"claude-sonnet-4-6"         = "gpt-5.3-codex"    # built-in: gpt-5.3-codex
"claude-haiku-4-5-20251001" = "gpt-5.4-mini"     # built-in: gpt-5.4-mini
"claude-haiku-4-5"          = "gpt-5.4-mini"     # built-in: gpt-5.4-mini
```

Slugs accepted by the ChatGPT-Codex backend (as of May 2026):
`gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.2`.
`gpt-5.3-codex-spark` is Pro-tier-only and not API-accessible.

---

## Switching ChatGPT accounts

```sh
claudeg logout                                     # forget token + kill proxy
# Sign out of chatgpt.com in your browser (or use a private window)
claudeg login                                      # OAuth with the other account
claudeg whoami                                     # confirm
```

---

## How do I know it's actually using ChatGPT?

Tail the proxy log in a side terminal:

```sh
# the auto-started service writes here
tail -f /tmp/claudeg.out.log

# or run the proxy in the foreground yourself:
claudeg serve
```

Every `claudeg "..."` invocation prints one line:

```
INFO claudeg: request method=POST path=/v1/messages model_in=claude-opus-4-7 model_out=gpt-5.5 status=200 …
```

Plain `claude` (no env vars) leaves no trace — its traffic goes straight to `api.anthropic.com`.

---

## Architecture

Single binary, all logic in `src/main.rs`:

| Section | Purpose |
|---|---|
| Types | Anthropic + Codex wire formats |
| Config | TOML loader, model-name mapping |
| Auth | OAuth PKCE login (loopback `:1455/auth/callback`), refresh on `<60s` |
| Translation | `to_codex(req)` — Anthropic Messages → Codex `responses` |
| Translation | `Translator` — Codex SSE → Anthropic SSE (or JSON if `stream=false`) |
| Handlers | `axum` routes: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health` |
| Service | per-OS auto-start (launchd / systemd-user / Windows Startup) |

The proxy listens loopback-only (`127.0.0.1:4000`). It is single-user; no multi-tenant code.

OAuth: the ChatGPT-Codex public client (`app_EMoamEEZ73f0CkXaXp7hrann`) + PKCE (S256) + localhost-redirect on port 1455 / 1457, same as OpenAI's own Codex CLI. Tokens cached at `~/Library/Application Support/claudeg/auth.json` (mode `0600`).

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Error: HTTP status client error (404 Not Found) for url (https://auth.openai.com/oauth/device/code)` | You're on an old build that used device-code. Reinstall via the one-liner. |
| `upstream 400: "Instructions are required"` | Anthropic request had no `system` field — claudeg sends a default. If you still see this, file an issue. |
| `upstream 400: "Store must be set to false"` / `"Unsupported parameter: max_output_tokens"` | Codex backend protocol drift. Open an issue with the proxy log. |
| `upstream 400: "<model> is not supported when using Codex with a ChatGPT account"` | The mapped model isn't on your tier (e.g. `gpt-5.5` requires Plus+). Add an override in `config.toml`. |
| Claude Code shows "Detected a custom API key" prompt every run | Run `claudeg setup` once — it pre-approves the sentinel value in `~/.claude.json`. |
| Claude Code shows "Auth conflict" warning | Harmless. Both `claude.ai` token and `ANTHROPIC_API_KEY` are set; the env var wins for `claudeg`, the token wins for plain `claude`. |
| Port 4000 busy | `claudeg serve --port 5000` (foreground), or set `[listen]` in `config.toml`. |

---

## Build from source

```sh
git clone https://github.com/tuantong/claudeg.git
cd claudeg
cargo build --release
./target/release/claudeg --help
```

Requires Rust 1.85+ (edition 2024). Pure Rust dependencies, rustls — no OpenSSL.

---

## License

MIT.

## Acknowledgements

OAuth flow and Codex Responses API contract reverse-engineered from the public [openai/codex](https://github.com/openai/codex) CLI. Inspired by [caixiaoshun/claudex](https://github.com/caixiaoshun/claudex) and [raine/claude-code-proxy](https://github.com/raine/claude-code-proxy).
