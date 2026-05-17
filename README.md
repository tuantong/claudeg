# claudeg

Run **Claude Code** through your **ChatGPT subscription** or a **free Gemini 3 Flash key**.

`claudeg` is a small Rust binary that presents Anthropic's Messages API on `127.0.0.1` and translates each request to one of two backends:

- **ChatGPT subscription** — your Plus / Pro / Business / Enterprise plan, signed with an OAuth PKCE token.
- **Gemini 3 Flash** (free) — Google AI Studio API key against the native `generateContent` endpoint.

Claude Code itself runs unmodified — `claudeg` sits on loopback and pretends to be Anthropic. You pick a backend once at `claudeg login` time; everything else (`claudeg "..."`, `whoami`, `logout`, `setup`) adapts automatically.

```
                                ┌─▶ chatgpt.com /backend-api/codex/responses
┌──────────────┐  HTTPS  ┌──────────────────┐
│ Claude Code  │ ──────▶ │ claudeg :4000    │
│ (claude CLI) │ ◀────── │ Anthropic ⇄ ...  │
└──────────────┘   SSE   └──────────────────┘
                                └─▶ generativelanguage.googleapis.com /v1beta/models/...
```

> **⚠ ChatGPT-backend ToS caveat.** Routing your consumer ChatGPT subscription as the backend for a third-party tool is **not what OpenAI prices it for**. OpenAI may detect this pattern and suspend or ban the account. Use at your own risk on accounts you can afford to lose. The Gemini backend uses a normal, supported Google AI Studio API key and has no such caveat — only the standard free-tier rate limits.

---

## Install

> **Prerequisite:** Install [Claude Code](https://github.com/anthropics/claude-code) first — `npm install -g @anthropic-ai/claude-code` (requires Node 18+). `claudeg` invokes the `claude` CLI from PATH; it does not bundle it.

**macOS / Linux:**

```sh
curl -fsSL https://claudeg.org/install.sh | sh
```

**Windows (PowerShell):**

```powershell
iwr -useb https://claudeg.org/install.ps1 | iex
```

The installer downloads the right binary for your OS/arch, drops it on PATH, and runs `claudeg setup`.

After install:

```sh
claude              # real Anthropic API (unchanged)
claudeg "..."       # routed through the backend you picked
```

---

## Choose a backend

`claudeg login` prompts you to pick:

```text
$ claudeg login

How do you want to back claudeg?

  1) ChatGPT subscription   Plus/Pro/Business/Enterprise · browser OAuth
  2) Gemini 3 Flash         Free · paste a Google AI Studio API key

Choose [1-2] (default 1): _
```

- **ChatGPT** opens your browser for OAuth PKCE.
- **Gemini** prompts for an API key — paste one from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Input is hidden; the key is validated against Google before being saved.

Skip the picker:

```sh
claudeg login --chatgpt                # force ChatGPT OAuth
claudeg login --gemini                 # force Gemini (interactive key prompt)
claudeg login --gemini-key AIza...     # non-interactive (CI / scripts)
```

Switch backends at any time: `claudeg logout` then `claudeg login` again.

---

## Usage

```text
claudeg                              interactive Claude Code session via the active backend
claudeg "<prompt>"                   one-shot — runs `claude` with your prompt
claudeg login                        pick a backend (interactive picker)
claudeg logout                       forget cached credentials, stop running proxy
claudeg whoami                       show active backend + model mapping
claudeg serve [--port N]             run the proxy in the foreground (advanced)
claudeg setup                        register auto-start service + approve sentinel
claudeg uninstall                    reverse setup
claudeg --help                       full help
```

`claudeg` (alone) auto-starts the background proxy on first call, sets `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`, and execs `claude` — same UX as plain `claude`, but every request goes through your chosen backend. Subsequent calls are instant — the proxy stays warm.

Your normal `claude` is unchanged — it still hits real Anthropic.

---

## Model mapping

### ChatGPT backend (auto, tier-aware)

`claudeg` reads `chatgpt_plan_type` from your ChatGPT OAuth JWT at login and after every token refresh, then picks a ChatGPT model that fits each Claude model's role:

| Claude → ChatGPT | Go | Plus / Edu | Pro / Business / Enterprise |
|---|---|---|---|
| `claude-opus-4-7` (reasoning) | `gpt-5.5` | `gpt-5.5` | `gpt-5.5-pro` |
| `claude-sonnet-4-6` (coding) | `gpt-5.5` | `gpt-5.3-codex` | `gpt-5.3-codex-spark` |
| `claude-haiku-4-5` (fast/cheap) | `gpt-5.4-mini` | `gpt-5.4-mini` | `gpt-5.4-mini` |

**Free tier is not supported** — claudeg returns `HTTP 403 subscription_required`. You need Go ($8/mo) or higher.

If the JWT lacks `chatgpt_plan_type` (rare), claudeg falls back to the Plus mapping and logs a warning once. Tier upgrades (e.g. Plus → Pro) take effect on the next token refresh.

### Gemini backend (fixed)

All Claude models route to `gemini-flash-latest`. Gemini 3 Flash is the only free model and serves general reasoning, coding, and fast/cheap roles with the same weights:

| Claude → Gemini | All tiers |
|---|---|
| `claude-opus-4-7` | `gemini-flash-latest` |
| `claude-sonnet-4-6` | `gemini-flash-latest` |
| `claude-haiku-4-5` | `gemini-flash-latest` |

Free-tier limits (as of May 2026): ~5–15 requests/minute and 100–1000 requests/day, both per project. See [Google's pricing docs](https://ai.google.dev/gemini-api/docs/pricing) for current numbers. Multimodal image input is **not supported** on the Gemini backend in this release — switch to the ChatGPT backend for image prompts.

Run `claudeg whoami` to see which backend is active and the live mapping:

```text
$ claudeg whoami
backend: ChatGPT subscription
logged in as Pro (expires in 23h 14m)

auto-mapping (detected from chatgpt_plan_type):
  claude-opus-4-7      → gpt-5.5-pro
  claude-sonnet-4-6    → gpt-5.3-codex-spark
  claude-haiku-4-5     → gpt-5.4-mini
  (unmapped Claude model → gpt-5.3-codex-spark)
```

```text
$ claudeg whoami
backend: Gemini 3 Flash (free)
key cached at ~/Library/Application Support/claudeg/auth.json (mode 0600)

mapping (all Claude models route to the same free model):
  claude-opus-4-7      → gemini-flash-latest
  claude-sonnet-4-6    → gemini-flash-latest
  claude-haiku-4-5     → gemini-flash-latest

Free tier limits apply: ~5–15 requests/minute, 100–1000 requests/day.
```

---

## How do I know which backend is running?

Tail the proxy log in a side terminal:

```sh
# the auto-started service writes here
tail -f /tmp/claudeg.out.log

# or run the proxy in the foreground yourself:
claudeg serve
```

Every `claudeg "..."` invocation prints one line:

```
INFO claudeg: request method=POST path=/v1/messages provider=chatgpt model_in=claude-opus-4-7 model_out=gpt-5.5-pro tier=pro status=200 …
INFO claudeg: request method=POST path=/v1/messages provider=gemini  model_in=claude-opus-4-7 model_out=gemini-flash-latest tier=free status=200 …
```

Plain `claude` (no env vars) leaves no trace — its traffic goes straight to `api.anthropic.com`.

---

## Switching backends or accounts

```sh
claudeg logout                                     # forget creds + kill proxy
# If switching ChatGPT accounts: sign out of chatgpt.com in your browser
# If switching Gemini keys: (optional) revoke the old key at aistudio.google.com/apikey
claudeg login                                      # pick a backend + authenticate
claudeg whoami                                     # confirm
```

---

## Architecture

Single binary, all logic in `src/main.rs`:

| Section | Purpose |
|---|---|
| Types | Anthropic + Codex + Gemini wire formats |
| Tier | `ChatGptTier` enum + JWT `chatgpt_plan_type` extraction + per-tier mapping table |
| Auth | Provider-tagged `AuthState` enum — OAuth tokens for ChatGPT, API key for Gemini, both persisted in `auth.json` |
| Translation (ChatGPT) | `to_codex(req, model)` — Anthropic Messages → Codex `responses` |
| Translation (ChatGPT) | `Translator` — Codex SSE → Anthropic SSE |
| Translation (Gemini) | `to_gemini(req)` — Anthropic Messages → Gemini `generateContent` |
| Translation (Gemini) | `GeminiTranslator` — Gemini SSE → Anthropic SSE |
| Handlers | `axum` routes: `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/health` |
| Service | per-OS auto-start (launchd / systemd-user / Windows Startup) |

The proxy listens loopback-only (`127.0.0.1:4000`). It is single-user; no multi-tenant code.

### Auth storage

`auth.json` lives at `~/Library/Application Support/claudeg/auth.json` (macOS), `~/.config/claudeg/auth.json` (Linux), or `%APPDATA%\claudeg\auth.json` (Windows). Mode `0600` on Unix. The file carries a `"provider"` discriminator and either OAuth tokens (ChatGPT) or an `api_key` field (Gemini). Pre-v0.3 untagged files are accepted as ChatGPT for backward compatibility.

### Configurable

The only configurable field is `listen` in `~/Library/Application Support/claudeg/config.toml` (or `~/.config/claudeg/config.toml` on Linux, `%APPDATA%\claudeg\config.toml` on Windows):

```toml
listen = "127.0.0.1:4000"
```

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `claude: command not found` / `'claude' is not recognized` | Claude Code isn't installed. `claudeg` invokes the `claude` CLI from PATH — install it with `npm install -g @anthropic-ai/claude-code` ([repo](https://github.com/anthropics/claude-code)), then re-run. |
| `HTTP 403 subscription_required` from claudeg | You're on the ChatGPT Free tier. Either upgrade to Go ($8/mo) or higher, or `claudeg logout && claudeg login --gemini` to switch to the free Gemini backend. |
| `gemini auth rejected` | Gemini API rejected the cached key. `claudeg logout && claudeg login --gemini` and paste a fresh key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey). |
| `gemini upstream 429: rate_limit` | Free-tier limits. Wait a minute (RPM) or a day (RPD). Or switch to the ChatGPT backend. |
| `image inputs are not supported on the Gemini backend` | This release routes only text + tool use on the Gemini path. Switch to the ChatGPT backend for image prompts. |
| `claudeg: jwt missing chatgpt_plan_type — assuming plus tier` warning | OpenAI's OAuth backend occasionally omits the tier claim. claudeg uses the Plus mapping as a safe default; usually clears on next token refresh. |
| `upstream 400: "<model> is not supported when using Codex with a ChatGPT account"` | The Codex backend rejected the mapped slug for your tier. Run `claudeg whoami` to see the auto-detected tier + mapping. If the mapping looks wrong, file an issue. |
| Claude Code shows "Detected a custom API key" prompt every run | Run `claudeg setup` once — it pre-approves the sentinel value in `~/.claude.json`. |
| `Auth conflict: Both a token (claude.ai) and an API key (ANTHROPIC_API_KEY) are set.` | **Harmless** when running through `claudeg` — ignore the warning and the two "Trying to use…?" suggestions. claudeg sets `ANTHROPIC_API_KEY` to its loopback sentinel so Claude Code routes to the proxy; if you've also logged into claude.ai in the same shell, both auth methods coexist. The env var wins for every `claudeg` invocation; the claude.ai token wins for plain `claude`. |
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

ChatGPT OAuth flow and Codex Responses API contract reverse-engineered from the public [openai/codex](https://github.com/openai/codex) CLI. Gemini wire format follows Google's public [Generative Language API](https://ai.google.dev/gemini-api/docs/api-reference) docs.
