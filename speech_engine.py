
"""
speech_engine.py
Core recording engine for the Real-Time Speech Analysis project.

This file is intended to work together with:
    - live_speech_module.py (analysis functions)
    - app.py (Streamlit UI)

Author: Refactored for modular architecture.
"""

import threading
import queue
import time
import sounddevice as sd
import numpy as np
import whisper
import librosa

from live_speech_module import (
    analyze_speech,
    update_session_features,
    add_log,
    display_session_summary,
    save_transcript,
    save_summary,
    save_csv_log,
    get_session_summary,
    get_session_logs,
    get_session_transcript,
    clear_session
)

class SpeechEngine:

    def __init__(self):
        self.sample_rate = 44100
        self.channels = 1

        self.silence_threshold = 0.01
        self.silence_duration_to_stop = 1.0
        self.minimum_speech_duration = 0.5

        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.model = None

        self.stream = None
        self.transcription_thread = None

        self.is_running = False

        self.speech_buffer = []
        self.chunk_number = 1
        self.is_speaking = False
        self.silence_start_time = None

    def load_model(self):
        if self.model is None:
            print("Loading Whisper...")
            self.model = whisper.load_model("base")
            print("Whisper loaded.")

    def _callback(self, indata, frames, time_info, status):
        rms = np.sqrt(np.mean(indata ** 2))

        if rms > self.silence_threshold:
            self.is_speaking = True
            self.speech_buffer.append(indata.copy())
            self.silence_start_time = None
            return

        if not self.is_speaking:
            return

        self.speech_buffer.append(indata.copy())

        if self.silence_start_time is None:
            self.silence_start_time = time.time()

        if time.time() - self.silence_start_time >= self.silence_duration_to_stop:
            audio = np.concatenate(self.speech_buffer, axis=0)

            if len(audio) / self.sample_rate >= self.minimum_speech_duration:
                self.audio_queue.put((self.chunk_number, audio))
                self.chunk_number += 1

            self.speech_buffer.clear()
            self.is_speaking = False
            self.silence_start_time = None

    def _worker(self):
        while True:
            if self.stop_event.is_set() and self.audio_queue.empty():
                break

            try:
                chunk_number, audio = self.audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                original = audio.copy()
                duration = len(audio) / self.sample_rate

                whisper_audio = audio.flatten().astype(np.float32)

                if self.sample_rate != 16000:
                    whisper_audio = librosa.resample(
                        whisper_audio,
                        orig_sr=self.sample_rate,
                        target_sr=16000,
                    )

                result = self.model.transcribe(
                    whisper_audio,
                    fp16=False,
                )

                text = result["text"].strip()

                if not text:
                    continue

                features = analyze_speech(
                    text,
                    result["segments"],
                    duration,
                    original,
                    self.sample_rate,
                )

                add_log(text, features)

                update_session_features(features)

                print(f"[Chunk {chunk_number}] {text}")

            finally:
                self.audio_queue.task_done()

    def start_recording(self):
        if self.is_running:
            return

        clear_session()

        self.speech_buffer = []
        self.chunk_number = 1
        self.is_speaking = False
        self.silence_start_time = None
        
        self.load_model()

        self.stop_event.clear()

        self.transcription_thread = threading.Thread(
            target=self._worker,
            daemon=True,
        )
        self.transcription_thread.start()

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
        )

        self.stream.start()

        self.is_running = True

    def stop_recording(self):
        if not self.is_running:
            return

        self.stop_event.set()

        self.stream.stop()
        self.stream.close()

        if self.speech_buffer:
            audio = np.concatenate(self.speech_buffer, axis=0)
            if len(audio) / self.sample_rate >= self.minimum_speech_duration:
                self.audio_queue.put((self.chunk_number, audio))

        self.audio_queue.join()
        self.transcription_thread.join()

        display_session_summary()
        save_transcript()
        save_summary()
        save_csv_log()

        self.is_running = False

    def get_transcript(self):
        return get_session_transcript()

    def get_summary(self):
        return get_session_summary()

    def get_logs(self):
        return get_session_logs()

    def recording_status(self):
        return self.is_running