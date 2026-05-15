# claudeg

Run **Claude Code** through your **ChatGPT subscription**.

`claudeg` is a small Rust binary that presents Anthropic's Messages API on `127.0.0.1` and translates each request to ChatGPT's Codex Responses API, signed with your ChatGPT OAuth token. Claude Code itself runs unmodified — `claudeg` just sits on loopback and pretends to be Anthropic.

```
┌──────────────┐  HTTPS  ┌──────────────────┐  HTTPS  ┌────────────────────────┐
│ Claude Code  │ ──────▶ │ claudeg :4000    │ ──────▶ │ chatgpt.com            │
│ (claude CLI) │ ◀────── │ Anthropic ⇄ Codex│ ◀────── │ /backend-api/codex/... │
└──────────────┘   SSE   └──────────────────┘   SSE   └────────────────────────┘
```

`claudeg` reads your ChatGPT tier from the OAuth JWT (`chatgpt_plan_type`) and picks the best ChatGPT model for each Claude model automatically — nothing to configure.

> **⚠ ToS caveat.** Routing your consumer ChatGPT subscription as the backend for a third-party tool is **not what OpenAI prices it for**. OpenAI may detect this pattern and suspend or ban the account. Use at your own risk on accounts you can afford to lose.

---

## Install

**macOS / Linux:**

```sh
curl -fsSL https://claudeg.org/install.sh | sh
```

**Windows (PowerShell):**

```powershell
iwr -useb https://claudeg.org/install.ps1 | iex
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
claudeg whoami              show detected tier + active model mapping
claudeg serve [--port N]    run the proxy in the foreground (advanced)
claudeg setup               register auto-start service + approve sentinel
claudeg uninstall           reverse setup
claudeg --help              full help
```

`claudeg` (alone) auto-starts the background proxy on first call, sets `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`, and execs `claude` — same UX as plain `claude`, but every request goes through your ChatGPT subscription. Anything you pass after `claudeg` (a prompt or any other `claude` flag) is forwarded verbatim. Subsequent calls are instant — the proxy stays warm.

Your normal `claude` is unchanged — it still hits real Anthropic.

---

## Model mapping (auto)

`claudeg` reads `chatgpt_plan_type` from your ChatGPT OAuth JWT at login and after every token refresh, then picks a ChatGPT model that fits each Claude model's role:

| Claude → ChatGPT | Go | Plus / Edu | Pro / Business / Enterprise |
|---|---|---|---|
| `claude-opus-4-7` (reasoning) | `gpt-5.5` | `gpt-5.5` | `gpt-5.5-pro` |
| `claude-sonnet-4-6` (coding) | `gpt-5.5` | `gpt-5.3-codex` | `gpt-5.3-codex-spark` |
| `claude-haiku-4-5` (fast/cheap) | `gpt-5.4-mini` | `gpt-5.4-mini` | `gpt-5.4-mini` |

Run `claudeg whoami` to see your detected tier and the live mapping:

```text
$ claudeg whoami
logged in as Pro (expires in 23h 14m)

auto-mapping (detected from chatgpt_plan_type):
  claude-opus-4-7      → gpt-5.5-pro
  claude-sonnet-4-6    → gpt-5.3-codex-spark
  claude-haiku-4-5     → gpt-5.4-mini
  (unmapped Claude model → gpt-5.3-codex-spark)
```

**Free tier is not supported** — claudeg returns `HTTP 403 subscription_required`. You need Go ($8/mo) or higher.

If the JWT lacks `chatgpt_plan_type` (rare), `claudeg` falls back to the Plus mapping and logs a warning once. Tier upgrades (e.g. Plus → Pro) take effect on the next token refresh, typically within an hour.

The mapping is hard-coded — there is no `default_model` or `[models]` to override. Legacy keys from v0.1 are parsed but ignored with a startup warning; delete them at your leisure.

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
INFO claudeg: request method=POST path=/v1/messages model_in=claude-opus-4-7 model_out=gpt-5.5-pro tier=pro status=200 …
```

Plain `claude` (no env vars) leaves no trace — its traffic goes straight to `api.anthropic.com`.

---

## Switching ChatGPT accounts

```sh
claudeg logout                                     # forget token + kill proxy
# Sign out of chatgpt.com in your browser (or use a private window)
claudeg login                                      # OAuth with the other account
claudeg whoami                                     # confirm tier + mapping
```

---

## Architecture

Single binary, all logic in `src/main.rs`:

| Section | Purpose |
|---|---|
| Types | Anthropic + Codex wire formats |
| Tier | `ChatGptTier` enum + JWT `chatgpt_plan_type` extraction + per-tier mapping table |
| Auth | OAuth PKCE login (loopback `:1455/auth/callback`), token refresh, tier persisted in `auth.json` |
| Translation | `to_codex(req, model)` — Anthropic Messages → Codex `responses` |
| Translation | `Translator` — Codex SSE → Anthropic SSE (or JSON if `stream=false`) |
| Handlers | `axum` routes: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health` |
| Service | per-OS auto-start (launchd / systemd-user / Windows Startup) |

The proxy listens loopback-only (`127.0.0.1:4000`). It is single-user; no multi-tenant code.

OAuth: the ChatGPT-Codex public client (`app_EMoamEEZ73f0CkXaXp7hrann`) + PKCE (S256) + localhost-redirect on port 1455 / 1457, same as OpenAI's own Codex CLI. Tokens cached at `~/Library/Application Support/claudeg/auth.json` (mode `0600`), including the detected tier.

The only configurable field is `listen` in `~/Library/Application Support/claudeg/config.toml` (or `~/.config/claudeg/config.toml` on Linux, `%APPDATA%\claudeg\config.toml` on Windows):

```toml
listen = "127.0.0.1:4000"
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `Error: HTTP status client error (404 Not Found) for url (https://auth.openai.com/oauth/device/code)` | You're on an old build that used device-code. Reinstall via the one-liner. |
| `HTTP 403 subscription_required` from claudeg | You're on the Free tier. claudeg requires Go ($8/mo) or higher. Run `claudeg whoami` to confirm. |
| `claudeg: jwt missing chatgpt_plan_type — assuming plus tier` warning | OpenAI's OAuth backend occasionally omits the tier claim. claudeg uses the Plus mapping as a safe default; usually clears on next token refresh. |
| `upstream 400: "<model> is not supported when using Codex with a ChatGPT account"` | The Codex backend rejected the mapped slug for your tier. Run `claudeg whoami` to see the auto-detected tier + mapping. If the mapping looks wrong, file an issue. |
| `upstream 400: "Instructions are required"` | Anthropic request had no `system` field — claudeg sends a default. If you still see this, file an issue. |
| `upstream 400: "Store must be set to false"` / `"Unsupported parameter: max_output_tokens"` | Codex backend protocol drift. Open an issue with the proxy log. |
| Claude Code shows "Detected a custom API key" prompt every run | Run `claudeg setup` once — it pre-approves the sentinel value in `~/.claude.json`. |
| Claude Code shows "Auth conflict" warning | Harmless. Both `claude.ai` token and `ANTHROPIC_API_KEY` are set; the env var wins for `claudeg`, the token wins for plain `claude`. |
| Port 4000 busy | `claudeg serve --port 5000` (foreground), or set `listen = "127.0.0.1:5000"` in `config.toml`. |

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

OAuth flow and Codex Responses API contract reverse-engineered from the public [openai/codex](https://github.com/openai/codex) CLI.
