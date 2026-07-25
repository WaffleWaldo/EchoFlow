"""Pipeline orchestration: record → transcribe → refine → inject."""

from __future__ import annotations

import enum
import logging
import math
import threading
import time
from typing import TYPE_CHECKING

import numpy as np

from echoflow.core.audio import AudioRecorder
from echoflow.core.refiner import Refiner
from echoflow.core.transcriber import Transcriber
from echoflow.data.dictionary import Dictionary
from echoflow.services.injector import Injector
from echoflow.services.niri import get_focused_window

if TYPE_CHECKING:
    from echoflow.config import Config
    from echoflow.services.overlay import Overlay

log = logging.getLogger(__name__)

# RMS → normalized level mapping
_DB_FLOOR = -60.0
_DB_CEIL = 0.0


class State(enum.Enum):
    IDLE = "idle"
    RECORDING = "recording"
    PROCESSING = "processing"


class Pipeline:
    """Owns all components and orchestrates the voice-to-text flow."""

    def __init__(self, config: Config, overlay: Overlay | None = None) -> None:
        self._config = config
        self._state = State.IDLE
        self._lock = threading.Lock()
        self._overlay = overlay

        # Core
        self._recorder = AudioRecorder(config.audio)
        self._transcriber = Transcriber(config.stt)
        self._refiner = Refiner(config.refiner)

        # Data
        self._dictionary = Dictionary(config.dictionary.path)

        # Services
        self._injector = Injector()

        # Only check Ollama connectivity at startup. The model itself is
        # loaded lazily when a dictation starts (see _start_recording) so an
        # idle daemon never pins VRAM — that starved games of memory.
        self._refiner.check_connection()

    @property
    def state(self) -> State:
        return self._state

    def toggle(self) -> str:
        """Toggle between idle and recording. Returns new state description."""
        with self._lock:
            if self._state == State.IDLE:
                return self._start_recording()
            elif self._state == State.RECORDING:
                return self._stop_recording()
            else:
                return f"busy ({self._state.value})"

    def _start_recording(self) -> str:
        self._state = State.RECORDING
        self._target_window = get_focused_window()
        # Warm the model while the user is speaking: a cold load overlaps
        # recording time instead of delaying the transcript, and a warm one
        # is a near-free request that refreshes the keep_alive countdown.
        threading.Thread(target=self._refiner.warmup, daemon=True).start()
        if self._overlay:
            self._overlay.show_recording()
        self._recorder.start(on_chunk=self._on_audio_chunk)
        return "recording"

    def _on_audio_chunk(self, chunk: np.ndarray) -> None:
        """Compute RMS of audio chunk and forward to overlay as 0.0–1.0 level."""
        if self._overlay is None:
            return
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms > 0:
            db = 20.0 * math.log10(rms)
        else:
            db = _DB_FLOOR
        level = (db - _DB_FLOOR) / (_DB_CEIL - _DB_FLOOR)
        level = max(0.0, min(1.0, level))
        self._overlay.update_audio_level(level)

    def _stop_recording(self) -> str:
        """Stop recording and kick off processing in a background thread."""
        self._recorder.stop()
        self._state = State.PROCESSING
        if self._overlay:
            self._overlay.show_processing()
        thread = threading.Thread(target=self._process, daemon=True)
        thread.start()
        return "processing"

    def _process(self) -> None:
        """Run the transcribe → refine → inject pipeline."""
        t_start = time.perf_counter()
        try:
            audio = self._recorder.get_audio()
            if audio.size == 0:
                log.warning("No audio captured")
                if self._overlay:
                    self._overlay.show_error()
                return

            audio_duration = audio.size / self._recorder.sample_rate
            log.info("Processing %.1fs of audio", audio_duration)

            # Transcribe
            whisper_prompt = self._dictionary.as_whisper_prompt()
            t0 = time.perf_counter()
            transcript = self._transcriber.transcribe(audio, initial_prompt=whisper_prompt)
            transcribe_ms = (time.perf_counter() - t0) * 1000

            if not transcript.strip():
                log.warning("Empty transcription")
                if self._overlay:
                    self._overlay.show_error()
                return

            # Use the window that was focused when recording started
            window = self._target_window

            # Refine + inject, streamed sentence-by-sentence so text lands on
            # screen as the model generates it instead of all at the end.
            dict_context = self._dictionary.as_llm_context()
            t0 = time.perf_counter()
            session = self._injector.begin_session(app_id=window["app_id"])
            first_ms = 0.0
            for chunk in self._refiner.refine_stream(
                transcript,
                dictionary_context=dict_context,
            ):
                if not chunk:
                    continue
                if first_ms == 0.0:
                    first_ms = (time.perf_counter() - t0) * 1000
                session.feed(chunk)
            success = session.end()
            refine_inject_ms = (time.perf_counter() - t0) * 1000

            total_ms = (time.perf_counter() - t_start) * 1000
            log.info(
                "Pipeline complete: audio=%.1fs, transcribe=%dms, "
                "refine+inject=%dms (first text @%dms), total=%dms",
                audio_duration, transcribe_ms, refine_inject_ms, first_ms, total_ms,
            )

            if self._overlay:
                if success:
                    self._overlay.show_done()
                else:
                    self._overlay.show_error()

        except Exception:
            log.exception("Pipeline error")
            if self._overlay:
                self._overlay.show_error()
        finally:
            with self._lock:
                self._state = State.IDLE

    def shutdown(self) -> None:
        """Clean shutdown."""
        if self._state == State.RECORDING:
            self._recorder.stop()
        log.info("Pipeline shut down")
