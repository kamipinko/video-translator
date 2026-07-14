"""
Subtitle segmentation from whisper word timestamps.

Builds properly-timed subtitle cues:
  - splits on sentence boundaries (. ! ? ... etc.) using word timestamps
  - splits on silence gaps (> 1.0s between words)
  - min cue duration 1.0s, max 6.0s
  - max ~84 chars per cue (2 lines x 42 chars)
  - no orphan cues (1-2 stray words get merged into a neighbor)
  - balanced 2-line breaks at word boundaries (<= 42 chars/line target)

v2.1 rendering fix (words cut off at the sides):
  - the .ass PlayResX/PlayResY now MATCH the actual video (libass scales the
    script grid onto the frame — a fixed 1920x1080 grid on a 1080x1920
    vertical video rendered the font 1.78x too big and hard-clipped lines,
    because WrapStyle 2 never wraps)
  - line breaking is PIXEL-measured with the real bundled Manga TTF (PIL
    textlength), not a fixed 42-char count — Manga glyphs run wide
  - safe side margins of 6% each side; per-cue font auto-shrink (\\fs) down
    to a 60% floor when a cue can't fit the pixel budget; unbreakable words
    longer than a full line (also CJK, which has no spaces) get hard-broken
  - WrapStyle 0 as a belt-and-suspenders: if anything still overflows,
    libass wraps instead of clipping
"""

import os

MAX_CHARS_PER_LINE = 42
MAX_LINES = 2
MAX_CUE_CHARS = MAX_CHARS_PER_LINE * MAX_LINES  # 84
MIN_DUR = 1.0
MAX_DUR = 6.0
GAP_SPLIT = 1.0          # silence gap that forces a new cue
ORPHAN_MAX_WORDS = 2     # cues this short get merged into a neighbor
ORPHAN_MAX_CHARS = 14

_SENT_END = tuple(".!?…。！？")


def _is_sentence_end(word_text):
    w = word_text.rstrip().rstrip('"”’)»')
    return w.endswith(_SENT_END) and not w.endswith("...")  # "..." mid-thought? keep: treat … as end
    # NOTE: trailing ellipsis chars still end a cue via max-dur/gap rules.


def flatten_words(segments):
    """faster-whisper segments (word_timestamps=True) -> flat list of word dicts."""
    words = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for w in seg_words:
                t = (w.word or "").strip()
                if t:
                    words.append({"text": t, "start": w.start, "end": w.end})
        else:
            # no word timestamps for this segment — fall back to the segment itself
            t = (seg.text or "").strip()
            if t:
                words.append({"text": t, "start": seg.start, "end": seg.end})
    return words


def build_cues(segments):
    """Return list of {'start','end','text'} cues from whisper segments."""
    words = flatten_words(segments)
    if not words:
        return []

    cues = []
    cur = []

    def close():
        if cur:
            cues.append({
                "start": cur[0]["start"],
                "end": cur[-1]["end"],
                "text": " ".join(w["text"] for w in cur),
            })
            cur.clear()

    for i, w in enumerate(words):
        if cur:
            gap = w["start"] - cur[-1]["end"]
            dur_if_added = w["end"] - cur[0]["start"]
            chars_if_added = len(" ".join(x["text"] for x in cur)) + 1 + len(w["text"])
            if gap > GAP_SPLIT or dur_if_added > MAX_DUR or chars_if_added > MAX_CUE_CHARS:
                close()
        cur.append(w)
        dur = cur[-1]["end"] - cur[0]["start"]
        if _is_sentence_end(w["text"]) and dur >= MIN_DUR:
            close()
    close()

    cues = _merge_orphans(cues)
    _enforce_min_duration(cues)
    return cues


def _merge_orphans(cues):
    out = []
    for cue in cues:
        wc = len(cue["text"].split())
        if (out and (wc <= ORPHAN_MAX_WORDS or len(cue["text"]) <= ORPHAN_MAX_CHARS)):
            prev = out[-1]
            merged_len = len(prev["text"]) + 1 + len(cue["text"])
            merged_dur = cue["end"] - prev["start"]
            gap = cue["start"] - prev["end"]
            if merged_len <= MAX_CUE_CHARS + 8 and merged_dur <= MAX_DUR + 1.0 and gap <= GAP_SPLIT:
                prev["text"] = prev["text"] + " " + cue["text"]
                prev["end"] = cue["end"]
                continue
        out.append(dict(cue))
    return out


def _enforce_min_duration(cues):
    for i, cue in enumerate(cues):
        if cue["end"] - cue["start"] < MIN_DUR:
            limit = cues[i + 1]["start"] - 0.05 if i + 1 < len(cues) else cue["start"] + MIN_DUR
            cue["end"] = max(cue["end"], min(cue["start"] + MIN_DUR, limit))


def split_lines(text, max_chars=MAX_CHARS_PER_LINE):
    """Break cue text into 1-2 balanced lines at word boundaries."""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return [text]
    words = text.split(" ")
    best = None
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        score = max(len(l1), len(l2))
        if best is None or score < best[0]:
            best = (score, [l1, l2])
    return best[1] if best else [text]


# ── pixel-accurate line layout (v2.1) ─────────────────────────────────────────

_FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fonts", "Manga-Regular.ttf")
_REF_SIZE = 100          # measure at 100px, scale linearly
_MIN_FONT_SCALE = 0.60   # never shrink below 60% of the base size
_SAFETY = 0.96           # measurement fudge (libass vs PIL metrics, outline)
_ref_font = None


def _font():
    global _ref_font
    if _ref_font is None:
        from PIL import ImageFont
        _ref_font = ImageFont.truetype(_FONT_PATH, _REF_SIZE)
    return _ref_font


def text_px(text, fontsize):
    """Rendered width of `text` at `fontsize`, measured with the real TTF."""
    return _font().getlength(text) * fontsize / _REF_SIZE


def layout_for(video_w, video_h):
    """Style numbers for a given video resolution (script grid == video).
    Fontsize keys off the MIN dimension (64 @ 1080): on a 9:16 vertical the
    old height-keyed size rendered 1.78x too large for the narrow frame."""
    fontsize = max(20, round(min(video_w, video_h) * 64 / 1080))
    margin_x = round(video_w * 0.06)                 # 6% safe margin per side
    margin_v = max(20, round(video_h * 0.06))
    outline = max(2, round(fontsize * 4 / 64))
    shadow = max(1, round(fontsize * 2 / 64))
    budget = (video_w - 2 * margin_x) * _SAFETY      # px available per line
    return {"fontsize": fontsize, "margin_x": margin_x, "margin_v": margin_v,
            "outline": outline, "shadow": shadow, "budget": budget}


def _hard_break(word, limit_px, fontsize):
    """Split a word wider than a full line into chunks that fit."""
    chunks, cur = [], ""
    for ch in word:
        if cur and text_px(cur + ch, fontsize) > limit_px:
            chunks.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        chunks.append(cur)
    return chunks


def _best_split(words, n_lines, fontsize):
    """Split words into n_lines contiguous lines minimizing the widest line."""
    if n_lines == 1:
        line = " ".join(words)
        return [line], text_px(line, fontsize)
    best = None
    if n_lines == 2:
        for i in range(1, len(words)):
            l1, l2 = " ".join(words[:i]), " ".join(words[i:])
            w = max(text_px(l1, fontsize), text_px(l2, fontsize))
            if best is None or w < best[1]:
                best = ([l1, l2], w)
    else:  # 3 lines — rare last resort
        for i in range(1, len(words) - 1):
            for j in range(i + 1, len(words)):
                ls = [" ".join(words[:i]), " ".join(words[i:j]), " ".join(words[j:])]
                w = max(text_px(l, fontsize) for l in ls)
                if best is None or w < best[1]:
                    best = (ls, w)
    return best if best else ([" ".join(words)], text_px(" ".join(words), fontsize))


def wrap_and_fit(text, budget_px, fontsize):
    """
    Fit `text` into <=2 lines within budget_px.
    Returns (lines, scale): scale < 1.0 means render with \\fs shrink.
    Falls back to 3 lines rather than shrinking below the 60% floor.
    """
    text = " ".join(text.split())
    if not text:
        return [text], 1.0
    words = []
    for w in text.split(" "):
        if text_px(w, fontsize) > budget_px:
            words.extend(_hard_break(w, budget_px, fontsize))
        else:
            words.append(w)

    if text_px(" ".join(words), fontsize) <= budget_px:
        return [" ".join(words)], 1.0

    lines, widest = _best_split(words, 2, fontsize)
    if widest <= budget_px:
        return lines, 1.0
    scale = budget_px / widest
    if scale >= _MIN_FONT_SCALE:
        return lines, scale
    # even 60% font wouldn't fit 2 lines — use 3 lines (rare: CJK/dense cue)
    if len(words) >= 3:
        lines3, widest3 = _best_split(words, 3, fontsize)
        if widest3 < widest:
            return lines3, max(_MIN_FONT_SCALE, min(1.0, budget_px / widest3))
    return lines, _MIN_FONT_SCALE


def _ass_time(sec):
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# Manga caption style (matches the video-edit pipeline's Shorts caption look:
# Manga font, white fill, heavy black outline + soft shadow, pop scale-in).
# PlayResX/PlayResY are filled in with the ACTUAL video dimensions.
ASS_HEADER_TMPL = """[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
ScaledBorderAndShadow: yes
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Manga,Manga,{fontsize},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,{outline},{shadow},2,{margin_x},{margin_x},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# caption pop-out: scale-in 78% -> 100% over 150ms (Captions.tsx spring), tiny fade
ASS_POP = r"{\fscx78\fscy78\t(0,150,\fscx100\fscy100)\fad(40,60)}"


def cues_to_ass(cues, video_w=1920, video_h=1080):
    """Render cues as a manga-styled .ass sized for the ACTUAL video frame,
    with pixel-measured line breaks and per-cue auto-shrink."""
    lay = layout_for(video_w, video_h)
    lines = [ASS_HEADER_TMPL.format(play_w=video_w, play_h=video_h,
                                    fontsize=lay["fontsize"], outline=lay["outline"],
                                    shadow=lay["shadow"], margin_x=lay["margin_x"],
                                    margin_v=lay["margin_v"])]
    for cue in cues:
        cue_lines, scale = wrap_and_fit(cue["text"], lay["budget"], lay["fontsize"])
        text = "\\N".join(_ass_escape(l) for l in cue_lines)
        shrink = ""
        if scale < 0.999:
            shrink = rf"{{\fs{max(1, round(lay['fontsize'] * scale))}}}"
        lines.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},"
            f"Manga,,0,0,0,,{ASS_POP}{shrink}{text}"
        )
    return "\n".join(lines) + "\n"


def cues_to_srt(cues, fmt_time):
    lines = []
    for i, cue in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{fmt_time(cue['start'])} --> {fmt_time(cue['end'])}")
        lines.extend(split_lines(cue["text"]))
        lines.append("")
    return "\n".join(lines)
