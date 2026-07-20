"""
==========================================================
Real-Time Speech Analysis Module

This file only contains speech analysis logic.

Recording and Streamlit UI are handled separately by

speech_engine.py

==========================================================
"""

import os
import re
import csv
import json
from datetime import datetime

import numpy as np

############################################################
# OUTPUT DIRECTORY
############################################################

OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__),
    "outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

############################################################
# SESSION FEATURES
############################################################

session_features = {

    "total_chunks":0,

    "total_hesitations":0,

    "repeat_request_count":0,

    "confusion_count":0,

    "speech_rates":[],

    "total_speaking_duration":0.0,

    "total_pause_count":0,

    "total_pause_duration":0.0,

    "total_silence_duration":0.0

}

############################################################
# SESSION LOGS
############################################################

session_logs=[]

session_transcripts=[]

############################################################
# RESET SESSION
############################################################

def reset_session():

    session_features["total_chunks"]=0

    session_features["total_hesitations"]=0

    session_features["repeat_request_count"]=0

    session_features["confusion_count"]=0

    session_features["speech_rates"].clear()

    session_features["total_speaking_duration"]=0.0

    session_features["total_pause_count"]=0

    session_features["total_pause_duration"]=0.0

    session_features["total_silence_duration"]=0.0

    session_logs.clear()

    session_transcripts.clear()

############################################################
# AUDIO PAUSE DETECTION
############################################################

def detect_pauses(

    audio,

    sample_rate,

    silence_threshold=0.01,

    minimum_pause_duration=0.30

):

    if audio is None or len(audio) == 0:
        return 0, 0.0

    audio = audio.flatten()

    frame_duration=0.03

    frame_size=int(

        sample_rate*

        frame_duration

    )

    frame_is_speech=[]

    for start in range(

        0,

        len(audio)-frame_size+1,

        frame_size

    ):

        frame=audio[

            start:start+frame_size

        ]

        rms=np.sqrt(

            np.mean(frame**2)

        )

        frame_is_speech.append(

            rms>silence_threshold

        )

    if len(frame_is_speech)==0:

        return 0,0.0

    first=None

    last=None

    for i,v in enumerate(frame_is_speech):

        if v:

            first=i

            break

    for i in range(

        len(frame_is_speech)-1,

        -1,

        -1

    ):

        if frame_is_speech[i]:

            last=i

            break

    if first is None or last is None:

        return 0,0.0

    pause_count=0

    total_pause=0.0

    silence_frames=0

    for i in range(first,last+1):

        if not frame_is_speech[i]:

            silence_frames+=1

        else:

            pause=silence_frames*frame_duration

            if pause>=minimum_pause_duration:

                pause_count+=1

                total_pause+=pause

            silence_frames=0

    return pause_count,round(total_pause,2)
############################################################
# SPEECH ANALYSIS
############################################################

def analyze_speech(

    text,

    segments,

    recording_duration,

    original_audio,

    sample_rate

):

    text=text.lower()

    words=re.findall(

        r"\b\w+\b",

        text

    )

    ########################################################
    # HESITATION DETECTION
    ########################################################

    hesitation_words=[

        "um",

        "umm",

        "uh",

        "erm",

        "ah",

        "hmm",

        "like",

        "actually",

        "maybe"

    ]

    hesitation_count=0

    for word in words:

        if word in hesitation_words:

            hesitation_count+=1

    ########################################################
    # REPEAT REQUEST
    ########################################################

    repeat_keywords=[

        "repeat",

        "again",

        "once more",

        "repeat that",

        "repeat this",

        "say again",

        "can you repeat",

        "could you repeat",

        "please repeat"

    ]

    repeat_request=False

    for phrase in repeat_keywords:

        if phrase in text:

            repeat_request=True

            break

    ########################################################
    # CONFUSION DETECTION
    ########################################################

    confusion_keywords=[

        "don't understand",

        "do not understand",

        "cannot understand",

        "can't understand",

        "confused",

        "hard",

        "difficult",

        "unclear",

        "not clear",

        "stuck",

        "help"

    ]

    confusion_detected=False

    for phrase in confusion_keywords:

        if phrase in text:

            confusion_detected=True

            break

    ########################################################
    # SPEAKING DURATION
    ########################################################

    speaking_duration=0.0

    for segment in segments:

        speaking_duration+=(

            segment["end"]

            -

            segment["start"]

        )

    ########################################################
    # SPEECH RATE
    ########################################################

    total_words=len(words)

    if speaking_duration>0:

        speech_rate=(

            total_words

            /

            speaking_duration

        )*60

    else:

        speech_rate=0

    ########################################################
    # PAUSE DETECTION
    ########################################################

    pause_count,pause_duration=detect_pauses(

        original_audio,

        sample_rate

    )

    ########################################################
    # SILENCE
    ########################################################

    silence_duration=(

        recording_duration

        -

        speaking_duration

    )

    if silence_duration<0:

        silence_duration=0

    ########################################################
    # RETURN FEATURES
    ########################################################

    return{

        "hesitation_count":

            hesitation_count,

        "repeat_request":

            repeat_request,

        "confusion_detected":

            confusion_detected,

        "speaking_duration":

            round(

                speaking_duration,

                2

            ),

        "speech_rate":

            round(

                speech_rate,

                2

            ),

        "pause_count":

            pause_count,

        "pause_duration":

            pause_duration,

        "silence_duration":

            round(

                silence_duration,

                2

            )

    }

############################################################
# UPDATE SESSION FEATURES
############################################################

def update_session_features(

    features

):

    session_features["total_chunks"]+=1

    session_features["total_hesitations"]+=(

        features["hesitation_count"]

    )

    if features["repeat_request"]:

        session_features[

            "repeat_request_count"

        ]+=1

    if features["confusion_detected"]:

        session_features[

            "confusion_count"

        ]+=1

    if features["speech_rate"]>0:

        session_features[

            "speech_rates"

        ].append(

            features["speech_rate"]

        )

    session_features[

        "total_speaking_duration"

    ]+=(

        features["speaking_duration"]

    )

    session_features[

        "total_pause_count"

    ]+=(

        features["pause_count"]

    )

    session_features[

        "total_pause_duration"

    ]+=(

        features["pause_duration"]

    )

    session_features[

        "total_silence_duration"

    ]+=(

        features["silence_duration"]

    )

############################################################
# ADD LOG ENTRY
############################################################

def add_log(

    transcript,

    features

):

    session_transcripts.append(

        transcript

    )

    session_logs.append({

        "transcript":

            transcript,

        **features

    })
    ############################################################
# DISPLAY SESSION SUMMARY
############################################################

def display_session_summary():

    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)

    total_chunks = session_features["total_chunks"]

    speech_rates = session_features["speech_rates"]

    average_rate = (
        sum(speech_rates) / len(speech_rates)
        if speech_rates
        else 0
    )

    print(f"Speech Chunks        : {total_chunks}")
    print(f"Total Hesitations    : {session_features['total_hesitations']}")
    print(f"Repeat Requests      : {session_features['repeat_request_count']}")
    print(f"Confusion Count      : {session_features['confusion_count']}")
    print(f"Average Speech Rate  : {average_rate:.2f} WPM")
    print(f"Speaking Duration    : {session_features['total_speaking_duration']:.2f} sec")
    print(f"Pause Count          : {session_features['total_pause_count']}")
    print(f"Pause Duration       : {session_features['total_pause_duration']:.2f} sec")
    print(f"Silence Duration     : {session_features['total_silence_duration']:.2f} sec")
    print("=" * 60)

############################################################
# SAVE TRANSCRIPT
############################################################

def save_transcript():

    transcript_path = os.path.join(
        OUTPUT_DIR,
        "transcript.txt"
    )

    with open(
        transcript_path,
        "w",
        encoding="utf-8"
    ) as f:

        for line in session_transcripts:
            f.write(line + "\n")

    print(f"Transcript saved -> {transcript_path}")

############################################################
# SAVE SUMMARY
############################################################

def save_summary():

    speech_rates = session_features["speech_rates"]

    average_rate = (
        sum(speech_rates) / len(speech_rates)
        if speech_rates
        else 0
    )

    summary = {

        "generated_at":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "total_chunks":
            session_features["total_chunks"],

        "total_hesitations":
            session_features["total_hesitations"],

        "repeat_request_count":
            session_features["repeat_request_count"],

        "confusion_count":
            session_features["confusion_count"],

        "average_speech_rate":
            round(average_rate, 2),

        "total_speaking_duration":
            round(
                session_features["total_speaking_duration"],
                2
            ),

        "total_pause_count":
            session_features["total_pause_count"],

        "total_pause_duration":
            round(
                session_features["total_pause_duration"],
                2
            ),

        "total_silence_duration":
            round(
                session_features["total_silence_duration"],
                2
            )
    }

    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(f"Summary saved -> {summary_path}")

############################################################
# SAVE CSV LOG
############################################################

def save_csv_log():

    csv_path = os.path.join(
        OUTPUT_DIR,
        "speech_log.csv"
    )

    if len(session_logs) == 0:
        return

    headers = list(session_logs[0].keys())

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=headers
        )

        writer.writeheader()

        writer.writerows(session_logs)

    print(f"CSV log saved -> {csv_path}")
    ############################################################
# GETTERS
############################################################

def get_session_summary():
    """
    Returns the complete session statistics dictionary.
    """
    return session_features


def get_session_logs():
    """
    Returns all analyzed speech logs.
    """
    return session_logs


def get_session_transcript():
    """
    Returns the entire transcript as a single string.
    """
    return "\n".join(session_transcripts)


############################################################
# CLEAR SESSION
############################################################

def clear_session():
    """
    Clears all session data so a new recording starts fresh.
    """
    reset_session()


############################################################
# MODULE READY MESSAGE
############################################################

print("Speech Analysis Module Loaded Successfully.")