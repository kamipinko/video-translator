"""
Claude-powered subtitle translation.

Replaces the old per-line Google Translate pass with context-aware chunked
translation through Claude:

  - cues are sent in chunks (default 40) WITH surrounding context lines,
    so pronouns, idioms and register survive across cue boundaries
  - the prompt asks for natural target-language phrasing, register
    preservation, idiom equivalence, and subtitle-length awareness
    (~84 chars per cue; final 2x42 line breaking happens locally)
  - transport is picked automatically:
      * ANTHROPIC_API_KEY set  -> official `anthropic` SDK (Claude API)
      * no key                 -> `claude --print` CLI subprocess
        (Pinko's king-claude-subs pattern; uses the CLI's own login)

Model selection (env CLAUDE_MODEL):
  - default: "claude-sonnet-5"  ($3/$15 per MTok; intro $2/$10 through
    2026-08-31) — best cost/quality for translation
  - max quality: set CLAUDE_MODEL=claude-fable-5  ($10/$50 per MTok)
    (thinking is always-on there; the disabled-thinking param is skipped)
"""

import json
import os
import re
import shutil
import subprocess

DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5")
CHUNK_SIZE = int(os.environ.get("TRANSLATE_CHUNK", "40"))
CONTEXT_LINES = 3

LANG_NAMES = {
    "ar": "Arabic", "da": "Danish", "de": "German", "en": "English",
    "es": "Spanish", "fi": "Finnish", "fr": "French", "hi": "Hindi",
    "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
    "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese",
    "ru": "Russian", "sv": "Swedish", "tr": "Turkish", "zh": "Chinese (Simplified)",
}

# cumulative usage for cost reporting (API path only)
usage_totals = {"input_tokens": 0, "output_tokens": 0, "requests": 0}

# v2.1: per-run report — lines that fell back to source text after all
# retries (NEVER a silent gap; the caller surfaces these as a warning)
last_report = {"fallbacks": []}


class _Truncated(Exception):
    """Model output hit max_tokens — retry with a smaller chunk."""


def _lang_name(code):
    return LANG_NAMES.get(code, code)


def _system_prompt(target_lang, source_lang):
    src = "the source language" if source_lang in (None, "auto") else _lang_name(source_lang)
    tgt = _lang_name(target_lang)
    return f"""You are an expert subtitle translator translating {src} into {tgt}.

Rules:
- Translate each numbered subtitle line into natural, idiomatic {tgt} as a native speaker would actually say it. Never translate word-for-word.
- Preserve the speaker's register and tone (casual stays casual, technical stays technical, jokes stay funny).
- Idioms and figures of speech: use the equivalent {tgt} idiom, not a literal rendering.
- Keep proper nouns, product names, numbers and units as-is unless the target language conventionally localizes them.
- These are subtitles: keep each translation concise enough to read quickly — aim for at most ~84 characters per line entry (it will be displayed as up to 2 screen lines of 42 chars). Trim filler words before trimming meaning.
- Lines are consecutive dialogue from one video: use the surrounding lines to resolve pronouns, gender, formality and continuity. Context lines marked CONTEXT are for reference only — do NOT include them in the output.
- Output ONLY a JSON object: {{"translations": ["...", "..."]}} with exactly one string per numbered input line, in the same order. No commentary, no markdown fences."""


def _chunk_prompt(chunk, before_ctx, after_ctx):
    parts = []
    if before_ctx:
        parts.append("CONTEXT (already shown, do not translate):")
        parts.extend(f"  {t}" for t in before_ctx)
    parts.append(f"TRANSLATE these {len(chunk)} lines:")
    for i, t in enumerate(chunk, 1):
        parts.append(f"{i}. {t}")
    if after_ctx:
        parts.append("CONTEXT (comes next, do not translate):")
        parts.extend(f"  {t}" for t in after_ctx)
    return "\n".join(parts)


def _extract_json(text):
    """Lenient JSON extraction (CLI output may carry stray text/fences)."""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in model output: {text[:200]!r}")
    return json.loads(text[start:end + 1])


# ── transport: Claude API (anthropic SDK) ─────────────────────────────────────

def _call_api(system, user, model):
    import anthropic
    client = anthropic.Anthropic()
    kwargs = {}
    if not model.startswith("claude-fable"):
        # translation needs no extended thinking; save tokens/latency.
        # (claude-fable-5: thinking is always-on and 'disabled' is rejected — omit.)
        kwargs["thinking"] = {"type": "disabled"}
    resp = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "translations": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["translations"],
                    "additionalProperties": False,
                },
            }
        },
        **kwargs,
    )
    usage_totals["input_tokens"] += resp.usage.input_tokens
    usage_totals["output_tokens"] += resp.usage.output_tokens
    usage_totals["requests"] += 1
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude refused the translation request")
    if resp.stop_reason == "max_tokens":
        raise _Truncated(f"output truncated at max_tokens (model {model})")
    text = next(b.text for b in resp.content if b.type == "text")
    return text


# ── transport: claude CLI subprocess (no API key configured) ─────────────────

def _find_claude_cli():
    exe = shutil.which("claude")
    if exe:
        return exe
    fallback = os.path.expanduser(r"~\.local\bin\claude.exe")
    if os.path.exists(fallback):
        return fallback
    raise RuntimeError(
        "No ANTHROPIC_API_KEY set and `claude` CLI not found — "
        "set ANTHROPIC_API_KEY or install Claude Code."
    )

def _call_cli(system, user, model):
    exe = _find_claude_cli()
    prompt = system + "\n\n" + user
    # prompt is piped via stdin (positional arg triggers interactive mode)
    proc = subprocess.run(
        [exe, "--print", "--model", model],
        input=prompt, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"claude CLI failed rc={proc.returncode}: {proc.stderr[-400:]}")
    usage_totals["requests"] += 1
    # rough token bookkeeping so the cost note still works on the CLI path
    usage_totals["input_tokens"] += len(prompt) // 4
    usage_totals["output_tokens"] += len(proc.stdout) // 4
    return proc.stdout


def _call_claude(system, user, model):
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _call_api(system, user, model)
    return _call_cli(system, user, model)


# ── public entry point ────────────────────────────────────────────────────────

def _translate_chunk(chunk, before, after, system, model, retries=3):
    """
    Translate one chunk with a HARD count invariant (len out == len chunk).
    Repair ladder — never returns a silent gap:
      1. up to `retries` straight attempts (count mismatch / bad JSON /
         empty strings / truncation all count as failures)
      2. split the chunk in half and recurse (also defeats max_tokens
         truncation on big chunks)
      3. a single cue that still fails falls back to its SOURCE text and is
         recorded in last_report["fallbacks"] (caller surfaces a warning) —
         something always renders.
    """
    last_err = None
    for _ in range(retries):
        try:
            raw = _call_claude(system, _chunk_prompt(chunk, before, after), model)
            data = _extract_json(raw)
            t = data["translations"]
            if len(t) != len(chunk):
                raise ValueError(f"expected {len(chunk)} translations, got {len(t)}")
            t = [str(x).strip() for x in t]
            if any(not x for x in t):
                raise ValueError("empty translation string(s) in response")
            return t
        except _Truncated as e:
            last_err = e
            break  # bigger context won't help — go straight to halving
        except Exception as e:  # noqa: BLE001 — retry then repair
            last_err = e
    if len(chunk) > 1:
        print(f"[translate] chunk of {len(chunk)} failed ({last_err}) — "
              f"splitting and retrying halves", flush=True)
        mid = len(chunk) // 2
        left = _translate_chunk(chunk[:mid], before, chunk[mid:mid + CONTEXT_LINES],
                                system, model, retries)
        right = _translate_chunk(chunk[mid:], chunk[max(0, mid - CONTEXT_LINES):mid],
                                 after, system, model, retries)
        return left + right
    # single cue, all retries burned: fall back to source text, LOUDLY
    print(f"[translate] WARNING: line untranslatable after retries "
          f"({last_err}) — keeping source text: {chunk[0][:80]!r}", flush=True)
    last_report["fallbacks"].append(chunk[0])
    return [chunk[0]]


def translate_lines(lines, target_lang, source_lang="auto",
                    model=None, progress_cb=None):
    """Translate a list of subtitle cue texts. ALWAYS returns a list of the
    same length (untranslatable lines fall back to source text, reported in
    last_report['fallbacks'])."""
    if not lines:
        return []
    model = model or DEFAULT_MODEL
    system = _system_prompt(target_lang, source_lang)
    last_report["fallbacks"] = []
    out = []
    chunks = [lines[i:i + CHUNK_SIZE] for i in range(0, len(lines), CHUNK_SIZE)]
    for ci, chunk in enumerate(chunks):
        before = lines[max(0, ci * CHUNK_SIZE - CONTEXT_LINES): ci * CHUNK_SIZE]
        after_start = (ci + 1) * CHUNK_SIZE
        after = lines[after_start: after_start + CONTEXT_LINES]
        out.extend(_translate_chunk(chunk, before, after, system, model))
        if progress_cb:
            progress_cb(ci + 1, len(chunks))
    assert len(out) == len(lines), f"count invariant broken: {len(lines)} in, {len(out)} out"
    return out
