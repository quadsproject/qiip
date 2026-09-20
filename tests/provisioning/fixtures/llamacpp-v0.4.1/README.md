# Real llama.cpp v0.4.1 startup logs

These are engine logs written by `auto-llamacpp/start-llamacpp.sh` in
profile sizing mode, launching `llama-server` v0.4.1 (build 10964) built with
the flags `auto-llamacpp/setup.sh` uses. Each run loaded the real GGUF files at
the revisions the profile catalog pins, served one chat completion and was then
stopped with `stop-llamacpp.sh`.

The cache prefix `/home/<user>/.cache/qiip-models` was replaced with
`/srv/hf-cache`, and trailing whitespace was removed. All log content and
measurements are otherwise unchanged.

They were captured on an RTX PRO 6000 Blackwell workstation, presented to the
script as a single GPU. They qualify the log parser against the pinned
revision. They are not evidence that a profile fits an L4 or an A30: memory
figures in them describe a 96 GB card.

`gemma-4-31b-assistant.engine.log` was captured the same way on 2026-09-20, on the
workstation's other GPU (so it carries a different GPU UUID). It is the only
fixture with a drafter that allocates no KV cache (Gemma's MTP assistant shares
the target's) and with two target caches (global and sliding-window).
