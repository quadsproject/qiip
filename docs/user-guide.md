# QIIP user guide

This guide is for people who use the gateway with a coding tool. It walks
through signing in, connecting a tool, and managing your token and usage.
Operators who run the gateway should read the README instead.

## Sign in

Open `/start` in your browser and sign in with the provider the operator
configured: a Google account (Google OAuth must be enabled by the operator;
if your domain is restricted, it has to be on the gateway allowlist), or a
local account on small installs that run the opt-in local OIDC provider
(the Local Login form; the operator hands out the username/password). Once
signed in you land on `/start`, your home page. Every operations page
(`/dashboard`, `/models`, `/profile`, the admin pages) sends a normal user
back there.

## Chat playground

`/chat` is a browser playground for trying out a model. It runs on your own
token, so the models it offers are the models your token may use (never
admin-only servers) and its usage is recorded with your other usage, which
`/leaderboard` shows for your tokens. If you have no token yet, the playground
stays disabled and links you to `/start`, where you create one.

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

A normal user owns exactly one token. The home screen shows it: name, the
first characters of its value, when it was last used, and the models it may
use (tap a chip to change the list). Three actions exist:

- **The clipboard icon** next to the prefix copies the full token to your
  clipboard, for a client or a script the setup flow does not cover (the
  pages themselves only ever show the prefix). A notification tells you the
  token is on the clipboard.
- **Set up a tool** reuses the current token and just fetches a new setup
  link for another harness.
- **Create a new token** revokes the old one. Tools configured with the old
  token stop working and need to be set up again.

Copying works for tokens created by the current setup flow. A token made
before it shows a note instead of the icon. If a token can no longer be
re-derived after the operator rotates the session secret, the icon stays and
the copy reports that a new token is needed; creating one replaces the old
token.

## Leaderboard

The `Leaderboard` link in the header opens `/leaderboard`, a simple
dashboard for signed-in users:

Admins see the same ranking; the token sections below are for normal users,
and admins manage their tokens on the profile page.

- **Usage leaderboard**: every non-admin user ranked by total tokens.
  Admin users and their usage are never shown. Your own row is marked
  `(you)`. The `Rank` column is that standing, so it stays with each user
  when you sort by another column.
- **Your tokens**: your token with its status and models, the clipboard icon
  next to its prefix (the same copy action as the home screen) and a Delete
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

Both tables sort by column: select a header label to sort ascending, select
it again for descending. The arrow next to the label shows the direction,
and the chosen sort stays in place when the page refreshes.

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
