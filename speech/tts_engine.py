import pyttsx3
import threading

_engine = None
_lock = threading.Lock()

def _get_engine():
    global _engine
    if _engine is None:
        try:
            _engine = pyttsx3.init()
        except Exception:
            pass
    return _engine

def speak(text: str):
    """Speak text in a non-blocking thread."""
    if not text.strip():
        return
        
    def _speak_task():
        with _lock:
            engine = _get_engine()
            if engine:
                engine.say(text)
                engine.runAndWait()
                
    threading.Thread(target=_speak_task, daemon=True).start()

def stop():
    """Stop the current speech."""
    with _lock:
        engine = _get_engine()
        if engine:
            engine.stop()

def set_rate(rate: int):
    """Set speech rate (words per minute). Default is usually 200."""
    with _lock:
        engine = _get_engine()
        if engine:
            engine.setProperty('rate', rate)

def set_volume(volume: float):
    """Set volume between 0.0 and 1.0."""
    with _lock:
        engine = _get_engine()
        if engine:
            engine.setProperty('volume', volume)
