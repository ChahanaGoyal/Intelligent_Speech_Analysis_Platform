"""
speech_engine.py

Browser-compatible Speech Engine
"""

import os
import tempfile

import librosa
import whisper

from live_speech_module import (
    analyze_speech,
    update_session_features,
    add_log,
    clear_session as reset_analysis_session,
    get_session_summary,
    get_session_logs,
    get_session_transcript,
)


class SpeechEngine:

    def __init__(self):

        print("Loading Whisper model...")

        self.model = whisper.load_model("base")

        print("Whisper Loaded Successfully!")

    ########################################################
    # START A NEW SESSION
    ########################################################

    def clear_session(self):
        """
        Clears all previous transcript and statistics.
        Call this only when starting a new session.
        """
        reset_analysis_session()

    ########################################################
    # PROCESS ONE RECORDED CHUNK
    ########################################################

    def process_audio(self, uploaded_audio):

        if uploaded_audio is None:
            return

        temp_path = None

        try:

            # Save browser recording temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(uploaded_audio.read())
                temp_path = tmp.name

            ################################################
            # LOAD AUDIO
            ################################################

            waveform, sample_rate = librosa.load(
                temp_path,
                sr=None,
                mono=True
            )

            recording_duration = librosa.get_duration(
                y=waveform,
                sr=sample_rate
            )

            ################################################
            # WHISPER
            ################################################

            result = self.model.transcribe(
                temp_path,
                fp16=False
            )

            transcript = result["text"].strip()

            if transcript == "":
                return

            ################################################
            # FEATURE EXTRACTION
            ################################################

            features = analyze_speech(
                transcript,
                result["segments"],
                recording_duration,
                waveform,
                sample_rate
            )

            ################################################
            # UPDATE SESSION
            ################################################

            add_log(
                transcript,
                features
            )

            update_session_features(
                features
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    ########################################################
    # GETTERS
    ########################################################

    def get_transcript(self):

        return get_session_transcript()

    def get_summary(self):

        return get_session_summary()

    def get_logs(self):

        return get_session_logs()