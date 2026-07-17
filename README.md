#  Real-Time Speech Analysis System

A Python-based real-time speech analysis system that records live speech, converts it to text using OpenAI Whisper, extracts speech-related features, and provides an interactive Streamlit dashboard for real-time visualization and session analytics.

##  Features

-  Real-time microphone recording
-  Speech-to-text transcription using OpenAI Whisper
-  Speech rate calculation (Words Per Minute)
-  Hesitation detection
-  Confusion detection
-  Repeat request detection
-  Pause and silence detection
-  Interactive Streamlit dashboard
-  Export transcript, summary (JSON), and speech log (CSV)

## 🛠 Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- NumPy
- Pandas
- Plotly
- SoundDevice
- Librosa

##  Project Structure

```
Speech_Module/
│
├── app.py
├── speech_engine.py
├── live_speech_module.py
├── outputs/
│   ├── transcript.txt
│   ├── summary.json
│   └── speech_log.csv
├── requirements.txt
└── README.md
```

##  How to Run

1. Clone the repository.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit application:

```bash
streamlit run app.py
```

4. Click **Start Recording**, speak into the microphone, then click **Stop Recording** to view the analysis.

##  Output

The application generates:

- **Transcript** (`transcript.txt`)
- **Session Summary** (`summary.json`)
- **Speech Log** (`speech_log.csv`)

##  Future Improvements

- Emotion recognition from speech
- Speaker identification
- Sentiment analysis
- Live speech quality scoring
- Cloud deployment