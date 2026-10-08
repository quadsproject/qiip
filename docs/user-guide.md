# QIIP user guide

This guide is for people who use the gateway with a coding tool. It walks
through signing in, connecting a tool, and managing your token and usage.
Operators who run the gateway should read the README instead.

## Sign in

Open `/start` in your browser and sign in with your Google account. Google
OAuth must be enabled by the operator; if your domain is restricted, it has
to be on the gateway allowlist. Once signed in you land on `/start`, your
home page. Every operations page (`/dashboard`, `/models`, `/chat`,
`/profile`, the admin pages) sends a normal user back there.

## Connect your tool the first time

If you have no token yet, `/start` walks you through three short questions:

1. give your token a name (it is the key to your coding tool),
2. pick a coding tool (agent harness),
3. pick the models it may use.

The last screen shows one line to paste into a terminal:

```bash
curl -sSL https://gateway.example.com/s/k7m2x9qd4tpa | bash
```

The script writes that tool's config file, already pointed at the gateway
with your token and models. It backs up any existing config first and merges
into JSON and TOML files instead of replacing them (Oh My Pi replaces the
file). The setup link lives for 15 minutes and can be fetched again inside
that window; it dies when the token is replaced.

Used with the tool it configures, it works with one model at a time
(Claude Code, Codex) or many (the rest):

| Coding tool | Config file written | Models |
|-------------|---------------------|--------|
| OpenCode | `~/.config/opencode/opencode.json` | many |
| Pi | `~/.pi/agent/models.json` | many |
| Oh My Pi | `~/.omp/agent/models.yml` | many |
| Claude Code | `~/.claude/settings.json` | one |
| Codex | `~/.codex/config.toml` | one |

A tool is offered only when the gateway serves the API route it speaks.
Models come and go as capacity changes: a model chip marked `(offline)` is
not served right now, and when no model is online the wizard says so and the
models can be picked later.

## Your token

A normal user owns exactly one token. The home screen shows it: name,
prefix, when it was last used, and the models it may use (tap a chip to
change the list). Two actions exist:

- **Set up a tool** reuses the current token and just fetches a new setup
  link for another harness.
- **Create a new token** revokes the old one. Tools configured with the old
  token stop working and need to be set up again.

The raw token value is shown once at creation and never again; the pages
only ever show its prefix. If you lose it, create a new token.

## Leaderboard

The `Leaderboard` link in the header opens `/leaderboard`, a simple
dashboard for signed-in users:

- **Usage leaderboard**: every non-admin user ranked by total tokens.
  Admin users and their usage are never shown. Your own row is marked
  `(you)`.
- **Your tokens**: your token with its status and models, plus a Delete
  button. Deleting revokes the token; tools configured with it stop
  working.
- **Create a token**: a simplified flow. Pick an optional model, or leave
  it on "All models" to cover every model online right now, then Create.
  The token is named `default` and replaces the previous one. Pick an
  optional agent harness and the Download config button becomes active.

**Download config** saves the environment's config file for that harness
(with your token embedded) instead of a setup script. Place it at the
config path from the table above. Unlike the setup script, it does not back
up or merge into an existing config file, and single-model tools take
exactly one model.

## Usage and limits

- Usage is recorded per request and credited to the token that called
  `/v1`; the leaderboard shows your totals in the ranking. Detailed
  per-model breakdowns live on the admin dashboard.
- The token may only request the models chosen for it. Anything else is
  refused with `403 model_not_permitted`, and `/v1/models` lists only your
  models.
- If the operator does not enforce API tokens, requests without a token are
  treated as anonymous, so the scope only constrains token-authenticated
  calls.
