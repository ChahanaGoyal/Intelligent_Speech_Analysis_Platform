import streamlit as st
import pandas as pd
import plotly.express as px
import time
import json
import os

from speech_engine import SpeechEngine

############################################################
# PAGE CONFIG
############################################################

st.set_page_config(
    page_title="Real-Time Speech Analysis",
    page_icon="🎤",
    layout="wide"
)

############################################################
# SESSION STATE
############################################################

if "engine" not in st.session_state:
    st.session_state.engine = SpeechEngine()

if "recording" not in st.session_state:
    st.session_state.recording = False

engine = st.session_state.engine

############################################################
# TITLE
############################################################

st.title("🎤 Real-Time Speech Analysis System")

st.markdown(
"""
Record your speech, analyze it using Whisper,
and visualize the results instantly.
"""
)

############################################################
# SIDEBAR
############################################################

with st.sidebar:

    st.header("Project")

    st.info(
        """
        Features

        • Speech to Text

        • Hesitation Detection

        • Confusion Detection

        • Speech Rate

        • Pause Detection

        • CSV Export

        • JSON Export
        """
    )

############################################################
# RECORDING BUTTONS
############################################################

col1,col2,col3=st.columns([1,1,2])

with col1:

    if st.button(
        "▶ Start Recording",
        use_container_width=True
    ):

        if not st.session_state.recording:

            engine.start_recording()

            st.session_state.recording=True

with col2:

    if st.button(
        "⏹ Stop Recording",
        use_container_width=True
    ):

        if st.session_state.recording:

            engine.stop_recording()

            st.session_state.recording=False

with col3:

    if st.session_state.recording:

        st.success("🟢 Recording...")

    else:

        st.error("🔴 Not Recording")

st.divider()

############################################################
# LIVE TRANSCRIPT
############################################################

st.subheader("📝 Transcript")

transcript=engine.get_transcript()

transcript_placeholder=st.empty()

transcript_placeholder.text_area(

    "Recognized Speech",

    transcript,

    height=250

)
############################################################
# LIVE SUMMARY
############################################################

summary = engine.get_summary()

speech_rates = summary.get("speech_rates", [])

if len(speech_rates) > 0:
    average_rate = round(
        sum(speech_rates) / len(speech_rates),
        2
    )
else:
    average_rate = 0

st.divider()

st.subheader("📊 Session Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Speech Chunks",
    summary.get("total_chunks", 0)
)

c2.metric(
    "Hesitations",
    summary.get("total_hesitations", 0)
)

c3.metric(
    "Repeat Requests",
    summary.get("repeat_request_count", 0)
)

c4.metric(
    "Speech Rate",
    f"{average_rate} WPM"
)

############################################################
# SUMMARY TABLE
############################################################

summary_df = pd.DataFrame({

    "Metric":[

        "Speech Chunks",

        "Hesitations",

        "Repeat Requests",

        "Confusions",

        "Speaking Duration",

        "Pause Count",

        "Pause Duration",

        "Silence Duration"

    ],

    "Value":[

        summary.get("total_chunks",0),

        summary.get("total_hesitations",0),

        summary.get("repeat_request_count",0),

        summary.get("confusion_count",0),

        summary.get("total_speaking_duration",0),

        summary.get("total_pause_count",0),

        summary.get("total_pause_duration",0),

        summary.get("total_silence_duration",0)

    ]

})

left,right = st.columns([1,2])

with left:

    st.dataframe(
        summary_df,
        hide_index=True,
        use_container_width=True
    )

with right:

    logs = engine.get_logs()

    if len(logs) > 0:

        df = pd.DataFrame(logs)

        st.dataframe(
            df,
            use_container_width=True
        )

    else:

        st.info("No speech analyzed yet.")

############################################################
# CHARTS
############################################################

if summary.get("total_chunks",0) > 0:

    chart = pd.DataFrame({

        "Metric":[

            "Speaking",

            "Pause",

            "Silence"

        ],

        "Seconds":[

            summary.get("total_speaking_duration",0),

            summary.get("total_pause_duration",0),

            summary.get("total_silence_duration",0)

        ]

    })

    st.subheader("📈 Duration Analysis")

    fig = px.bar(

        chart,

        x="Metric",

        y="Seconds",

        text="Seconds"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.pie(

        chart,

        names="Metric",

        values="Seconds"

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

############################################################
# DOWNLOADS
############################################################

st.divider()

st.subheader("📥 Downloads")

d1,d2,d3 = st.columns(3)

transcript = engine.get_transcript()

with d1:

    st.download_button(

        "Transcript",

        transcript,

        file_name="transcript.txt"

    )

with d2:

    st.download_button(

        "Summary",

        json.dumps(summary, indent=4),

        file_name="summary.json",

        mime="application/json"

    )

with d3:

    logs = engine.get_logs()

    if len(logs) > 0:

        csv = pd.DataFrame(logs).to_csv(index=False)

    else:

        csv = ""

    st.download_button(

        "Speech Log",

        csv,

        file_name="speech_log.csv",

        mime="text/csv"

    )

############################################################
# AUTO REFRESH
############################################################

if st.session_state.recording:

    time.sleep(1)

    st.rerun()

############################################################
# FOOTER
############################################################

st.divider()

st.caption(
    "Built using Streamlit • Whisper • Plotly"
)