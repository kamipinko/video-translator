"""
Subtitle segmentation from whisper word timestamps.

Builds properly-timed subtitle cues:
  - splits on sentence boundaries (. ! ? ... etc.) using word timestamps
  - splits on silence gaps (> 1.0s between words)
  - min cue duration 1.0s, max 6.0s
  - max ~84 chars per cue (2 lines x 42 chars)
  - no orphan cues (1-2 stray words get merged into a neighbor)
  - balanced 2-line breaks at word boundaries (<= 42 chars/line target)
"""

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


def _ass_time(sec):
    cs = int(round(sec * 100))
    return f"{cs // 360000}:{cs // 6000 % 60:02d}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def _ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


# Manga caption style (matches the video-edit pipeline's Shorts caption look:
# Manga font, white fill, heavy black outline + soft shadow, pop scale-in).
ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Manga,Manga,64,&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,0,0,0,0,100,100,0,0,1,4,2,2,60,60,64,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

# caption pop-out: scale-in 78% -> 100% over 150ms (Captions.tsx spring), tiny fade
ASS_POP = r"{\fscx78\fscy78\t(0,150,\fscx100\fscy100)\fad(40,60)}"


def cues_to_ass(cues):
    """Render cues as a manga-styled .ass with per-caption pop-in animation."""
    lines = [ASS_HEADER]
    for cue in cues:
        text = "\\N".join(_ass_escape(l) for l in split_lines(cue["text"]))
        lines.append(
            f"Dialogue: 0,{_ass_time(cue['start'])},{_ass_time(cue['end'])},"
            f"Manga,,0,0,0,,{ASS_POP}{text}"
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
