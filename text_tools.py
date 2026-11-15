"""与 AstrBot 无关的纯函数：文本清理、情绪映射、判定结果解析、WAV 封装。"""
from __future__ import annotations

import io
import json
import re
import wave

ALLOWED_TAGS = ("[笑]", "[叹气]", "[咳嗽]", "[清嗓子]")
_TAG_RE = re.compile(r"\[(?:笑|叹气|咳嗽|清嗓子)\]")
_ANY_BRACKET_TAG_RE = re.compile(r"\[[^\[\]\n]{1,6}\]")
_BOUNDARY = set("，,。.！!？?；;：:…~～—、")

MEME_RE = re.compile(r"&&\s*([A-Za-z_]+)\s*&&")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_CODE_BLOCK_RE = re.compile(r"```")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((?:[^)]+)\)")
_ACTION_RE = re.compile(r"[（(][^（）()\n]{1,30}[）)]")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF"
    "\U00002B00-\U00002BFF\U0000FE0F\U0000200D\U000020E3]"
)

# 表情包插件标签 → (情绪, 语气指导, 行内标签)
MEME_EMOTIONS: dict[str, tuple[str, str, str]] = {
    "happy": ("开心", "语气开心，带着笑意", "[笑]"),
    "like": ("喜欢", "语气甜甜的，带着喜爱", ""),
    "angry": ("生气", "语气气鼓鼓的，有点不满", ""),
    "sad": ("难过", "语气低落委屈", "[叹气]"),
    "sigh": ("无奈", "语气无奈，轻轻叹气", "[叹气]"),
    "surprised": ("惊讶", "语气惊讶，音调上扬", ""),
    "confused": ("疑惑", "语气疑惑不解，句尾上扬", ""),
    "shy": ("害羞", "语气害羞，声音小一点，略带迟疑", ""),
    "baka": ("嫌弃", "语气嫌弃又带点娇嗔", ""),
    "fool": ("调皮", "语气调皮搞怪", "[笑]"),
    "color": ("坏笑", "语气坏坏的，带着暧昧的笑意", "[笑]"),
    "meow": ("撒娇", "语气撒娇可爱", ""),
    "sleep": ("困倦", "语气困倦慵懒", ""),
    "morning": ("元气", "语气元气满满，明亮有活力", ""),
    "work": ("认真", "语气认真专注", ""),
    "givemoney": ("讨好", "语气讨好撒娇", ""),
    "cpu": ("懵", "语气懵懵的，反应不过来", ""),
    "see": ("好奇", "语气好奇，带点探究", ""),
    "reply": ("平静", "语气自然平静", ""),
}


def extract_meme_hints(text: str) -> list[str]:
    return [m.group(1).lower() for m in MEME_RE.finditer(text or "")]


def should_skip_text(text: str, skip_url_and_code: bool) -> str | None:
    """返回跳过原因；None 表示可以合成。在清理前对原文判断。"""
    if skip_url_and_code and (_CODE_BLOCK_RE.search(text) or _URL_RE.search(text)):
        return "含链接或代码块"
    return None


def clean_for_speech(text: str, strip_actions: bool = True) -> str:
    t = MEME_RE.sub("", text)
    t = re.sub(r"&&+", "", t)
    t = _MD_LINK_RE.sub(r"\1", t)
    t = _URL_RE.sub("", t)
    if strip_actions:
        t = _ACTION_RE.sub("", t)
    # 去掉 LLM 自己写的方括号标签以外的 markdown 记号，保留允许的行内标签
    t = re.sub(r"`+", "", t)
    t = re.sub(r"(\*\*|__|~~)", "", t)
    t = re.sub(r"(?m)^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)", "", t)
    t = _EMOJI_RE.sub("", t)
    t = re.sub(r"[~～]{2,}", "～", t)
    t = re.sub(r"\s*\n+\s*", "，", t)
    t = re.sub(r"[ \t　]+", " ", t)
    # 群聊人设常用空格代替标点：两个汉字之间的空格改成逗号，让 Breeze 有停顿；
    # 中英混排（如“用 Python 写”）的空格只是排版，保留
    t = re.sub(r"(\[(?:笑|叹气|咳嗽|清嗓子)\]) +", r"\1", t)
    t = re.sub(r"(?<=[一-鿿]) (?=[一-鿿\[])", "，", t)
    t = re.sub(r"([，。！？、,.!?；;：:…～~])[，,]+", r"\1", t)
    t = re.sub(r"[，,]+([。！？!?…～~])", r"\1", t)
    t = t.strip(" ，,")
    if t and t[-1] not in "。！？!?…～~.":
        t += "。"
    return t


def _norm(s: str) -> str:
    return re.sub(r"[\s，,。.！!？?、~～…]+", "", s)


def apply_tagged_text(
    original: str, tagged: str | None, max_tags: int, allowed: tuple[str, ...] | list[str] = ALLOWED_TAGS
) -> str:
    """只接受“原文 + 插入标签”的结果，否则原样返回。

    不在 allowed 里的标签、插在句中的标签会被去掉；完全没有可用标签时返回原文。
    """
    if not tagged or max_tags <= 0:
        return original
    tagged = tagged.strip()
    # 出现了未知的方括号标签（模型自己编的） → 整个拒绝
    for m in _ANY_BRACKET_TAG_RE.finditer(tagged):
        if m.group(0) not in ALLOWED_TAGS and m.group(0) not in original:
            return original
    if _norm(_TAG_RE.sub("", tagged)) != _norm(_TAG_RE.sub("", original)):
        return original
    out, kept, pos = [], 0, 0
    for m in _TAG_RE.finditer(tagged):
        out.append(tagged[pos : m.start()])
        before = "".join(out).rstrip()
        at_boundary = not before or before[-1] in _BOUNDARY
        if m.group(0) in allowed and at_boundary and kept < max_tags:
            out.append(m.group(0))
            kept += 1
        pos = m.end()
    out.append(tagged[pos:])
    return "".join(out) if kept else original


def parse_judge_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


_PACE_RE = re.compile(r"语速|速度|节奏|快|慢|急促|拖长|停顿|放缓|加速")


def clip_direction(direction: str, limit: int = 40) -> str:
    """规整语气指导，并去掉所有涉及语速/节奏的分句（语速由 speed_hint 统一固定）。"""
    d = re.sub(r"\s+", "", str(direction or "")).strip("。.，, ")
    parts = [x for x in re.split(r"[，,、；;。]", d) if x and not _PACE_RE.search(x)]
    return "，".join(parts)[:limit]


def build_instruction(voice: str, direction: str, use_voice: bool, speed_hint: str = "") -> str | None:
    parts = []
    voice = (voice or "").strip().rstrip("。.")
    if use_voice and voice:
        parts.append(voice)
    if direction:
        parts.append(direction)
    speed_hint = (speed_hint or "").strip().rstrip("。.")
    if speed_hint:
        parts.append(speed_hint)
    if not parts:
        return None
    return "，".join(parts) + "。"


def pcm_to_wav(pcm: bytes, sample_rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm[: len(pcm) - len(pcm) % 2])
    return buf.getvalue()
