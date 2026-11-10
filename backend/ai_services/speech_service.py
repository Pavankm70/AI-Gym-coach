"""
Text-to-Speech (TTS) Voice Coaching Service
===========================================
Generates speech audio for AI Coach responses using gTTS with
in-memory caching, deduplication, and cooldown protection.
"""

import io
import time
import base64
from typing import Optional


class SpeechService:
    def __init__(self, min_interval_seconds: float = 3.5, dedup_window_seconds: float = 10.0):
        self.min_interval_seconds = min_interval_seconds
        self.dedup_window_seconds = dedup_window_seconds
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""
        self.is_muted = False

    def set_muted(self, muted: bool):
        self.is_muted = muted

    def reset(self):
        self.last_spoken_time = 0.0
        self.last_spoken_text = ""

    def synthesize_speech_base64(self, text: str, force: bool = False, now: Optional[float] = None, is_muted: Optional[bool] = None) -> Optional[str]:
        """
        Synthesizes speech audio as a base64 encoded MP3 data URI.
        Returns None if muted, within cooldown, or duplicate.
        """
        if (is_muted is True) or self.is_muted or not text or not text.strip():
            return None

        clean_text = text.strip()
        if now is None:
            now = time.time()

        if not force:
            # Prevent speech on every frame (cooldown)
            if self.last_spoken_time > 0 and (now - self.last_spoken_time < self.min_interval_seconds):
                return None

            # Prevent identical message repetition within window
            if clean_text == self.last_spoken_text and self.last_spoken_time > 0 and (now - self.last_spoken_time < self.dedup_window_seconds):
                return None

        try:
            from gtts import gTTS

            # Truncate to maximum 30 words for fast, responsive audio playback
            words = clean_text.split()
            spoken_text = " ".join(words[:25]) if len(words) > 25 else clean_text

            fp = io.BytesIO()
            tts = gTTS(text=spoken_text, lang="en", tld="com", slow=False)
            tts.write_to_fp(fp)
            fp.seek(0)

            b64_audio = base64.b64encode(fp.read()).decode("utf-8")
            self.last_spoken_time = now
            self.last_spoken_text = clean_text

            return f"data:audio/mp3;base64,{b64_audio}"
        except Exception as e:
            # Non-fatal: Never crash CV monitoring if TTS fails
            print(f"[SpeechService] TTS warning (non-fatal): {e}")
            return None


# Global speech service instance
_SPEECH_SERVICE = SpeechService()


def get_speech_service() -> SpeechService:
    return _SPEECH_SERVICE
