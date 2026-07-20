# 🎤 Intelligent Speech Analysis Platform

An AI-powered speech analysis platform that records speech, transcribes it using OpenAI Whisper, extracts meaningful communication features, and presents interactive analytics through a Streamlit dashboard.

🌐 **Live Demo:** https://intelligentspeechanalysisplatform.streamlit.app/

## Features

- 🎙 Record speech directly from the browser
- 📝 Automatic speech-to-text transcription using Whisper
- ⏸ Pause detection
- 🤔 Confusion detection
- 💬 Repeat request detection
- ⚡ Hesitation detection
- 📊 Speech rate analysis
- 📈 Interactive dashboard
- 📂 Download transcript, summary, and speech logs


## Tech Stack

- Python
- Streamlit
- OpenAI Whisper
- Librosa
- NumPy
- Pandas
- Plotly


## Project Structure

```
Speech_Module/
│
├── app.py
├── speech_engine.py
├── live_speech_module.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
```


## Installation

Clone the repository

```bash
git clone https://github.com/ChahanaGoyal/Intelligent_Speech_Analysis_Platform.git
cd Intelligent_Speech_Analysis_Platform
```

Install dependencies

```bash
pip install -r requirements.txt
```


## Running the Project

Start the Streamlit application

```bash
streamlit run app.py
```

The application will automatically open in your browser.


## How It Works

1. Record a speech sample using the browser microphone.
2. Whisper converts speech into text.
3. The platform extracts communication features including:
   - Hesitations
   - Confusion indicators
   - Repeat requests
   - Speech rate
   - Pause analysis
   - Silence duration
4. Results are displayed in an interactive dashboard.
5. Download reports in TXT, JSON, or CSV format.

## Dashboard Includes

- Speech transcript
- Session summary
- Chunk-wise speech logs
- Speaking vs Pause vs Silence visualization
- Downloadable reports


## Example Workflow

```
Record Speech
Whisper Transcription
Feature Extraction
Dashboard
```


## Future Improvements

- Real-time speech streaming
- Emotion recognition
- Speaker identification
- Confidence score estimation
- Keyword extraction
- Multi-language support
