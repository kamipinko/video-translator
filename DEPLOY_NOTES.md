# DEPLOY NOTES — Language Pipeline Overhaul v2.0 (2026-07-14)

**Deploy to Railway AUTHORIZED by Pinko (amended orders, relayed 2026-07-14).**

## What changed

| Layer | Before | After |
|---|---|---|
| ASR | whisper `tiny` int8 CPU always, no VAD, segment timestamps | device chain: CUDA `large-v3-turbo` float16 → **loud** CPU `tiny` int8 fallback (never silent; surfaced in job message + `/health`); VAD + **word timestamps** |
| Translation | GoogleTranslator line-by-line, no context | **Claude** (`claude-sonnet-5`) chunked 40 cues w/ context lines; natural phrasing/register/idiom prompt; JSON-schema output on API path |
| English target | whisper `task="translate"` | whisper transcribes native; Claude translates all targets incl. English |
| Segmentation | raw whisper segments | sentence-boundary cues, 1–6s, ≤84 chars, orphan merging, 2×42 lines |
| Burn | drawtext, DejaVu, static | **libass .ass** — Manga font (bundled `fonts/Manga-Regular.ttf`), white + black outline + shadow, **pop scale-in 78→100% (150ms)** per caption |

Files: `processor.py`, `subtitles.py` (new), `translate_llm.py` (new), `main.py`
(+`/health`), `fonts/Manga-Regular.ttf` (new), `requirements.txt` (+anthropic,
−deep-translator), `test_local_e2e.py` (new).

## Env config

| Var | Local (RedRig) | Railway |
|---|---|---|
| `WHISPER_DEVICE` | `auto` (default; CUDA→loud CPU) | `cpu` (deliberate CPU mode) |
| `WHISPER_MODEL` | `large-v3-turbo` (default) | n/a (CPU mode) |
| `WHISPER_MODEL_CPU` | `tiny` (default; `small` if wanted) | `tiny` (512MB RAM) |
| `ANTHROPIC_API_KEY` | unset → `claude --print` CLI transport | **set** (SDK transport; no CLI on Railway) |
| `CLAUDE_MODEL` | `claude-sonnet-5` ($3/$15 MTok; intro $2/$10 to 2026-08-31). Max quality: `claude-fable-5` ($10/$50) | same |

`RAILWAY_DOCKERFILE_PATH=Dockerfile` also set (Railpack-ignores-Dockerfile gotcha).

## Deploy procedure (executed 2026-07-14)

1. `railway link --project 490faaad-… --service video-translator` (CLI; the
   GraphQL token in `~/.railway/config.json` is DEAD — use the CLI)
2. `railway variables --set … --skip-deploys` (the vars above)
3. ⚠️ GitHub→Railway auto-deploy is DEAD (deployment list was stuck at May 7).
   Deploy with `railway up --detach` from the repo root instead (uses
   `.railwayignore`; builds the local tree with the Dockerfile).
4. Verify: `GET /health` returns `"version": "2.0-claude"` + transport `api`
5. Live E2E: POST /translate small clip → poll /status → /download → frame QC

## Cost (measured)

claude-sonnet-5: ~535 input + ~230 output tokens per 60s of dense speech
→ **≈ $0.005/video-minute sticker ($3/$15), ≈ $0.003 on intro pricing.**
10-min video ≈ 5¢. Fable-5 same usage ≈ $0.017/min.

## Rollback

`git revert` the v2.0 commit and push; old drawtext pipeline had no env needs.
Railway vars are additive-safe (old code ignores them).
