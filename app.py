import json
import pandas as pd
import plotly.express as px
import streamlit as st

from speech_engine import SpeechEngine

############################################################
# PAGE CONFIG
############################################################

st.set_page_config(
    page_title="Intelligent Speech Analysis Platform",
    page_icon="🎤",
    layout="wide"
)

############################################################
# SESSION STATE
############################################################

if "engine" not in st.session_state:
    st.session_state.engine = SpeechEngine()

if "session_started" not in st.session_state:
    st.session_state.session_started = False

engine = st.session_state.engine

############################################################
# TITLE
############################################################

st.title("🎤 Intelligent Real-Time Speech Analysis Platform")

st.markdown(
"""
Analyze speech using Whisper AI and automatically detect:

- Hesitations
- Confusion
- Repeat Requests
- Speech Rate
- Pause Analysis
"""
)

############################################################
# SIDEBAR
############################################################

with st.sidebar:

    st.header("Session Controls")

    if st.button("🟢 New Session", use_container_width=True):

        engine.clear_session()

        st.session_state.session_started = True

        st.success("New session started.")

    if st.button("🔴 End Session", use_container_width=True):

        st.session_state.session_started = False

        st.success("Session ended.")

############################################################
# RECORD AUDIO
############################################################

st.header("🎤 Record Speech")

audio = st.audio_input(
    "Click below and record a speech chunk"
)

if audio is not None:

    if not st.session_state.session_started:

        st.warning("Start a new session first.")

    else:

        with st.spinner("Analyzing speech..."):

            engine.process_audio(audio)

        st.success("Speech analyzed successfully!")

############################################################
# TRANSCRIPT
############################################################

st.divider()

st.header("📝 Transcript")

transcript = engine.get_transcript()

st.text_area(
    "Recognized Speech",
    transcript,
    height=250
)

############################################################
# SUMMARY
############################################################

summary = engine.get_summary()

speech_rates = summary.get("speech_rates", [])

average_rate = (
    round(sum(speech_rates) / len(speech_rates), 2)
    if len(speech_rates) > 0
    else 0
)

st.divider()

st.header("📊 Session Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Chunks",
    summary.get("total_chunks", 0)
)

c2.metric(
    "Hesitations",
    summary.get("total_hesitations", 0)
)

c3.metric(
    "Confusions",
    summary.get("confusion_count", 0)
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

    if len(logs)>0:

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

if summary.get("total_chunks",0)>0:

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

    st.divider()

    st.header("📈 Duration Analysis")

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

st.header("📥 Download Reports")

d1,d2,d3 = st.columns(3)

with d1:

    st.download_button(

        "Transcript",

        transcript,

        file_name="transcript.txt"

    )

with d2:

    st.download_button(

        "Summary",

        json.dumps(summary,indent=4),

        file_name="summary.json",

        mime="application/json"

    )

with d3:

    logs = engine.get_logs()

    if len(logs)>0:

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
# FOOTER
############################################################

st.divider()

st.caption(
    "Built using Streamlit • Whisper • Plotly • Librosa"
)