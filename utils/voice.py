"""
Voice I/O module.
- Voice Input:  SpeechRecognition (mic → text)
- Voice Output: gTTS (text → mp3 → Streamlit audio player)

Note: Mic input requires pyaudio + portaudio system lib.
If unavailable, the UI gracefully falls back to text input only.
"""

import io
import base64
import tempfile
import os
from typing import Optional


# ── Voice Output (Text → Speech) ──────────────────────────────────────────────

def text_to_speech(text: str, lang: str = "en", slow: bool = False) -> Optional[bytes]:
    """
    Convert text to MP3 audio bytes using gTTS (Google Text-to-Speech, free).
    Returns raw mp3 bytes, or None on failure.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:3000], lang=lang, slow=slow)  # cap at 3000 chars
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception as e:
        print(f"[TTS Error] {e}")
        return None


def get_audio_html(audio_bytes: bytes) -> str:
    """
    Embed audio bytes as an autoplay HTML audio element.
    Used to play TTS output inline in Streamlit.
    """
    b64 = base64.b64encode(audio_bytes).decode()
    return f"""
    <audio controls autoplay style="width:100%; margin-top:8px;">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
    </audio>
    """


# ── Voice Input (Mic → Text) ───────────────────────────────────────────────────

def check_mic_available() -> bool:
    """Return True if pyaudio + SpeechRecognition are both importable."""
    try:
        import speech_recognition as sr
        import pyaudio
        return True
    except ImportError:
        return False


def record_and_transcribe(timeout: int = 5, phrase_limit: int = 15) -> Optional[str]:
    """
    Record from mic and transcribe via Google Speech Recognition (free, online).
    Returns transcribed string or None on failure/no speech.

    timeout      : seconds to wait for speech to start
    phrase_limit : max seconds of speech to record
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(
                source, timeout=timeout, phrase_time_limit=phrase_limit
            )
        text = recognizer.recognize_google(audio)
        return text
    except Exception as e:
        print(f"[Voice Input Error] {e}")
        return None


def transcribe_audio_file(audio_file_bytes: bytes, file_format: str = "wav") -> Optional[str]:
    """
    Transcribe an uploaded audio file (wav/mp3) via Google Speech Recognition.
    Useful when mic is not available — user uploads a recording instead.
    """
    try:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as tmp:
            tmp.write(audio_file_bytes)
            tmp_path = tmp.name
        with sr.AudioFile(tmp_path) as source:
            audio = recognizer.record(source)
        os.unlink(tmp_path)
        return recognizer.recognize_google(audio)
    except Exception as e:
        print(f"[Audio File Transcription Error] {e}")
        return None
