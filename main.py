import asyncio
import os
import random
import time
import uuid
from pathlib import Path

import aiohttp

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.provider import LLMResponse
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import At, Face, Image, Plain, Record, Reply
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .text_tools import (
    ALLOWED_TAGS,
    MEME_EMOTIONS,
    apply_tagged_text,
    build_instruction,
    clean_for_speech,
    clip_direction,
    extract_meme_hints,
    parse_judge_json,
    pcm_to_wav,
    should_skip_text,
)

HINT_KEY = "breeze_tts_meme_hints"
JUDGE_SYSTEM = """你是语音合成的配音导演。用户会给你一段聊天机器人即将用语音说出的话，请判断说这句话时应有的情绪，并只输出一个 JSON 对象，不要任何其他文字：
{"emotion": "情绪词", "direction": "语气指导", "tagged_text": "带标签的原文"}
要求：
1. emotion：一个简短的中文情绪词，如 开心、温柔、撒娇、无奈、生气、难过、惊讶、害羞、平静、调皮、认真。
2. direction：不超过 20 个字的中文表演指导，只描述情绪和语气，例如“语气开心，带着笑意”。不要提语速、快慢、节奏、停顿，也不要描述性别、年龄、音色。
3. tagged_text：{tag_rule}
"""
TAG_USAGE = {
    "[笑]": "[笑] 只用于真正好笑、开心或坏笑的时候",
    "[叹气]": "[叹气] 只用于真正无奈、难过、疲惫的时候",
    "[咳嗽]": "[咳嗽] 只用于真的在咳嗽或假装咳嗽掩饰的时候",
    "[清嗓子]": "[清嗓子] 只用于郑重开口或掩饰尴尬的时候",
}


def tag_rule_on(allowed: list[str], n: int) -> str:
    return (
        f"原文一字不改。可以插入声音标签，可用的只有 {' '.join(allowed)}，最多 {n} 个，"
        "且只能放在句首或标点符号之后，绝不能插在词语或短语中间。"
        "大多数情况下不需要标签，宁缺毋滥：15 字以内的短句、生气、撒娇、嗔怪、惊讶的话一般不加；"
        + "；".join(TAG_USAGE[t] for t in allowed)
        + "。不需要就原样返回原文。"
    )


TAG_RULE_OFF = "原样返回原文。"


class BreezeTTSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._temp_dir = Path(get_astrbot_temp_path()) / "breeze_tts"
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_temp(max_age=0)  # 启动时清掉上次遗留的文件

    async def _send_record(self, event: AstrMessageEvent, path: Path):
        """语音单独发一条，发完（无论成败）立即删除临时 wav。

        各平台适配器在 send() 返回前已把文件读走（aiocqhttp 转 base64、Telegram/QQ 官方机器人上传），
        所以发完删除是安全的。"""
        try:
            await event.send(MessageChain([Record.fromFileSystem(str(path))]))
        finally:
            try:
                path.unlink(missing_ok=True)
            except OSError as e:
                logger.warning(f"[breeze_tts] 删除临时语音失败：{e}")

    # ---------- 配置 ----------
    def _cfg(self, key, default=None):
        return self.config.get(key, default)

    def _emo(self, key, default=None):
        return (self.config.get("emotion") or {}).get(key, default)

    @property
    def base_url(self) -> str:
        return str(self._cfg("base_url", "http://127.0.0.1:7860")).rstrip("/")

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def terminate(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ---------- 表情包标记（在 meme_manager 删掉之前记下来） ----------
    @filter.on_llm_response(priority=100000)
    async def remember_meme_hints(self, event: AstrMessageEvent, resp: LLMResponse):
        try:
            hints = extract_meme_hints(resp.completion_text if resp else "")
            if hints:
                event.set_extra(HINT_KEY, hints)
        except Exception:
            pass

    # ---------- 主流程 ----------
    def _session_block_reason(self, event: AstrMessageEvent, check_sender: bool = True) -> str | None:
        """返回本会话不转语音的原因；None 表示生效。"""
        platforms = [p for p in (self._cfg("platforms") or []) if p]
        if platforms and event.get_platform_name() not in platforms:
            return f"平台 {event.get_platform_name()} 不在生效平台里"
        umo = event.unified_msg_origin
        if umo in (self._cfg("session_blacklist") or []):
            return "本会话已关闭"
        white = [w for w in (self._cfg("session_whitelist") or []) if w]
        if white and umo not in white:
            return "本会话不在白名单"
        if check_sender:
            senders = [str(x).strip() for x in (self._cfg("sender_whitelist") or []) if str(x).strip()]
            if senders and str(event.get_sender_id()) not in senders:
                return "发言人不在白名单"
        return None

    def _session_allowed(self, event: AstrMessageEvent) -> bool:
        return self._session_block_reason(event) is None

    def _save(self):
        self.config.save_config()

    @filter.on_decorating_result(priority=10)
    async def maybe_speak(self, event: AstrMessageEvent):
        if not self._cfg("enabled", True):
            return
        result = event.get_result()
        if not result or not result.chain or not result.is_llm_result():
            return
        if not self._session_allowed(event):
            return
        if random.random() > float(self._cfg("probability", 0.15)):
            return

        texts, others = [], []
        for comp in result.chain:
            if isinstance(comp, Plain):
                texts.append(comp.text)
            elif isinstance(comp, Image | Face | At | Reply):
                others.append(comp)
            else:  # 已有语音、文件、转发等：不处理
                return
        raw = "\n".join(t for t in texts if t.strip())
        if not raw:
            return
        if reason := should_skip_text(raw, bool(self._cfg("skip_url_and_code", True))):
            logger.debug(f"[breeze_tts] 跳过：{reason}")
            return
        speech = clean_for_speech(raw, bool(self._cfg("strip_actions", True)))
        n = len(speech)
        if n < int(self._cfg("min_len", 2)) or n > int(self._cfg("max_len", 100)):
            logger.debug(f"[breeze_tts] 跳过：长度 {n}")
            return

        started = time.monotonic()
        try:
            path, info = await self.synthesize(event, speech, raw_for_hint=raw, user_msg=event.message_str)
        except Exception as e:
            logger.warning(f"[breeze_tts] 合成失败，改发文字：{e}")
            return

        logger.info(
            f"[breeze_tts] {event.unified_msg_origin} 转语音 {n} 字，"
            f"情绪={info.get('emotion')}，用时 {time.monotonic() - started:.1f}s，"
            f"指导={info.get('instruction')}，文本={info.get('text')}"
        )
        # 语音必须单独成一条消息，先发出去；图片等其余部分留给正常流程
        await self._send_record(event, path)
        if self._cfg("dual_output", False):
            return
        keep = [c for c in others if isinstance(c, Image | Face)]
        result.chain.clear()
        result.chain.extend(keep)

    # ---------- 情绪 ----------
    async def judge_emotion(
        self, event: AstrMessageEvent, speech: str, raw: str, user_msg: str | None = None
    ) -> dict:
        mode = self._emo("mode", "llm")
        info: dict = {"emotion": None, "direction": "", "text": speech}
        if mode == "off":
            return info
        allowed = [t for t in (self._emo("allowed_tags") or ["[笑]"]) if t in ALLOWED_TAGS]
        max_tags = int(self._emo("max_tags", 2)) if self._emo("inline_tags", True) and allowed else 0

        if mode == "llm":
            try:
                pid = self._emo("provider_id") or await self.context.get_current_chat_provider_id(
                    event.unified_msg_origin
                )
                tag_rule = tag_rule_on(allowed, max_tags) if max_tags else TAG_RULE_OFF
                hints = event.get_extra(HINT_KEY) or extract_meme_hints(raw)
                prompt = ""
                if user_msg:
                    prompt += f"对方刚才说：{user_msg[:200]}\n"
                if hints:
                    prompt += f"机器人回复时附带的表情标记：{', '.join(hints)}\n"
                prompt += f"机器人要说的话：{speech}"
                resp = await asyncio.wait_for(
                    self.context.llm_generate(
                        chat_provider_id=pid,
                        system_prompt=JUDGE_SYSTEM.replace("{tag_rule}", tag_rule),
                        prompt=prompt,
                    ),
                    timeout=float(self._emo("timeout", 20)),
                )
                data = parse_judge_json(resp.completion_text or "")
                if data:
                    info["emotion"] = str(data.get("emotion") or "")[:10] or None
                    info["direction"] = clip_direction(data.get("direction", ""))
                    info["text"] = apply_tagged_text(speech, data.get("tagged_text"), max_tags, allowed)
                    return info
                logger.warning(f"[breeze_tts] 情绪分析返回无法解析：{(resp.completion_text or '')[:120]}")
            except Exception as e:
                logger.warning(f"[breeze_tts] 情绪分析失败，改用表情标记：{e!r}")

        # meme 模式 / llm 失败的退路
        for h in event.get_extra(HINT_KEY) or extract_meme_hints(raw):
            if h in MEME_EMOTIONS:
                emotion, direction, tag = MEME_EMOTIONS[h]
                info["emotion"], info["direction"] = emotion, clip_direction(direction)
                if tag and max_tags and tag in allowed:
                    info["text"] = f"{tag}{speech}"
                break
        return info

    # ---------- Breeze ----------
    async def synthesize(
        self, event: AstrMessageEvent | None, speech: str, raw_for_hint: str = "", user_msg: str | None = None
    ):
        http = await self._http()
        mode = self._cfg("model_mode", "current")
        # 先确认服务在线，避免白白调一次 LLM
        async with http.get(f"{self.base_url}/playground/model", timeout=aiohttp.ClientTimeout(total=5)) as r:
            if r.status == 404 and mode == "current":
                # 官方 Breeze 服务没有 /playground 接口：不检查模型，直接走 /v1/audio/speech
                status = {"status": "ok"}
            else:
                status = await r.json(content_type=None)
        if status.get("status") != "ok":
            raise RuntimeError(f"Breeze 未就绪：{status.get('status')} {status.get('error') or ''}")
        if mode != "current" and status.get("model") != mode:
            raise RuntimeError(f"Breeze 当前加载的是 {status.get('model')}，配置要求 {mode}")

        if event is not None:
            info = await self.judge_emotion(event, speech, raw_for_hint or speech, user_msg)
        else:
            info = {"emotion": None, "direction": "", "text": speech}

        ref_path = (self._cfg("ref_audio_path") or "").strip()
        ref_text = (self._cfg("ref_text") or "").strip()
        use_ref = bool(ref_path and ref_text and os.path.isfile(ref_path))
        ref_bytes = Path(ref_path).read_bytes() if use_ref else b""
        # use_instruction=False：完全不发 instruction（走 tts_plain / ref_clone，最贴近 LoRA 训练时的音色），
        # 情绪只靠行内标签体现
        instruction = None
        if self._emo("use_instruction", True):
            instruction = build_instruction(
                self._cfg("voice_instruction", ""), info["direction"], use_voice=not use_ref,
                speed_hint=self._cfg("speed_hint", "语速自然平稳，不要太快"),
            )
        seed = int(self._cfg("seed", 42))
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        def form() -> aiohttp.FormData:
            f = aiohttp.FormData()
            f.add_field("text", info["text"])
            if instruction:
                f.add_field("instruction", instruction)
            # Breeze：没有 instruction 的模板（tts_plain / ref_clone）只接受 cfg_scale=1
            f.add_field("cfg_scale", str(float(self._cfg("cfg_scale", 3.0)) if instruction else 1.0))
            f.add_field("seed", str(seed))
            if mode != "current":
                f.add_field("model_mode", mode)
            if use_ref:
                f.add_field("ref_text", ref_text)
                f.add_field("ref_audio", ref_bytes, filename=os.path.basename(ref_path))
            return f

        url = f"{self.base_url}{'/v1/audio/speech' if mode == 'current' else '/playground/speech'}"
        deadline = time.monotonic() + float(self._cfg("busy_wait", 30))
        timeout = aiohttp.ClientTimeout(total=float(self._cfg("request_timeout", 120)), sock_connect=5)
        async with self._lock:
            while True:
                async with http.post(url, data=form(), timeout=timeout) as r:
                    if r.status == 409 and time.monotonic() < deadline:
                        detail = (await r.text())[:200]
                        if "模型已改变" in detail:
                            raise RuntimeError("Breeze 模型已被切换")
                        await asyncio.sleep(2)
                        continue
                    if r.status != 200:
                        raise RuntimeError(f"Breeze HTTP {r.status}: {(await r.text())[:200]}")
                    rate = int(r.headers.get("X-Sample-Rate", "24000"))
                    pcm = await r.read()
                    break
        if len(pcm) < rate:  # 不到 0.5 秒，多半有问题
            raise RuntimeError(f"Breeze 返回音频过短（{len(pcm)} 字节）")

        self._cleanup_temp()
        path = self._temp_dir / f"{uuid.uuid4().hex}.wav"
        path.write_bytes(pcm_to_wav(pcm, rate))
        info.update(instruction=instruction, seconds=len(pcm) / 2 / rate)
        return path, info

    def _cleanup_temp(self, max_age: float = 3600):
        now = time.time()
        for p in self._temp_dir.glob("*.wav"):
            try:
                if now - p.stat().st_mtime > max_age:
                    p.unlink()
            except OSError:
                pass

    # ---------- 指令 ----------
    @filter.command("breeze")
    async def status_cmd(self, event: AstrMessageEvent):
        """查看 Breeze 语音插件状态"""
        try:
            http = await self._http()
            async with http.get(f"{self.base_url}/playground/model", timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status == 404:
                    svc = "在线（官方接口，无模型信息）"
                else:
                    s = await r.json(content_type=None)
                    svc = f"在线，模型 {s.get('model')}（{s.get('status')}）"
        except Exception as e:
            svc = f"连不上（{type(e).__name__}）"
        lines = [
            f"Breeze 服务：{svc}",
            f"插件：{'开启' if self._cfg('enabled', True) else '关闭'}，概率 {self._cfg('probability')}，"
            f"情绪 {self._emo('mode', 'llm')}，模型要求 {self._cfg('model_mode', 'current')}",
            f"本会话：{self._session_block_reason(event) or '生效'}",
            f"会话 ID：{event.unified_msg_origin}",
            "管理员指令：/breeze开 /breeze关（本会话） /breeze全局开 /breeze全局关 /breeze概率 <0-1> /breeze说 <文本>",
        ]
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze开")
    async def session_on_cmd(self, event: AstrMessageEvent):
        """在当前会话开启语音回复（仅管理员）"""
        umo = event.unified_msg_origin
        black = self.config.setdefault("session_blacklist", [])
        white = self.config.setdefault("session_whitelist", [])
        changed = False
        while umo in black:
            black.remove(umo)
            changed = True
        # 白名单为空 = 所有会话生效，不需要加；非空时才把本会话加进去
        if [w for w in white if w] and umo not in white:
            white.append(umo)
            changed = True
        if changed:
            self._save()
        msg = "本会话已开启语音回复" if changed else "本会话本来就是开启的"
        if reason := self._session_block_reason(event, check_sender=False):
            msg += f"，但仍不会生效：{reason}"
        if not self._cfg("enabled", True):
            msg += "（注意：总开关是关的，用 /breeze全局开 打开）"
        senders = [str(x).strip() for x in (self._cfg("sender_whitelist") or []) if str(x).strip()]
        if senders:
            msg += f"\n发言人白名单生效中，只有这些人的消息会转语音：{', '.join(senders)}"
        yield event.plain_result(msg)

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze关")
    async def session_off_cmd(self, event: AstrMessageEvent):
        """在当前会话关闭语音回复（仅管理员）"""
        umo = event.unified_msg_origin
        black = self.config.setdefault("session_blacklist", [])
        if umo in black:
            yield event.plain_result("本会话本来就是关闭的")
            return
        # 只加黑名单，不动白名单：白名单删空会变成“所有会话生效”
        black.append(umo)
        self._save()
        yield event.plain_result("本会话已关闭语音回复，/breeze开 可重新打开")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze全局开")
    async def global_on_cmd(self, event: AstrMessageEvent):
        """打开插件总开关（仅管理员）"""
        self.config["enabled"] = True
        self._save()
        yield event.plain_result(f"Breeze 语音总开关已打开（概率 {self._cfg('probability')}）")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze全局关")
    async def global_off_cmd(self, event: AstrMessageEvent):
        """关闭插件总开关（仅管理员）"""
        self.config["enabled"] = False
        self._save()
        yield event.plain_result("Breeze 语音总开关已关闭，所有会话都不再转语音")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze概率")
    async def prob_cmd(self, event: AstrMessageEvent):
        """/breeze概率 <0-1>：设置回复转语音的概率（仅管理员）"""
        arg = event.message_str.partition(" ")[2].strip().rstrip("%")
        try:
            value = float(arg)
            if "%" in event.message_str or value > 1:
                value = value / 100  # 允许写 30 或 30%
            if not 0 <= value <= 1:
                raise ValueError
        except ValueError:
            yield event.plain_result(f"用法：/breeze概率 0.3（0～1，也可以写 30%）。当前 {self._cfg('probability')}")
            return
        self.config["probability"] = round(value, 4)
        self._save()
        yield event.plain_result(f"转语音概率已设为 {self.config['probability']}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @filter.command("breeze说")
    async def say_cmd(self, event: AstrMessageEvent):
        """/breeze说 <文本>：带情绪分析合成一条语音（仅管理员）"""
        text = event.message_str.partition(" ")[2].strip()
        if not text:
            yield event.plain_result("用法：/breeze说 <文本>")
            return
        speech = clean_for_speech(text, bool(self._cfg("strip_actions", True)))
        try:
            path, info = await self.synthesize(event, speech, raw_for_hint=text)
        except Exception as e:
            yield event.plain_result(f"合成失败：{e}")
            return
        await self._send_record(event, path)
        yield event.plain_result(
            f"情绪：{info.get('emotion')}\n指导：{info.get('instruction')}\n文本：{info.get('text')}\n时长：{info.get('seconds', 0):.1f}s"
        )
