# 🎙️ Audio Transcription App (Vosk Only, Streamlit)

This Streamlit app transcribes `.m4a` audio (or zip archives of `.m4a`) to Spanish text using the [Vosk](https://alphacephei.com/vosk/) offline speech recognition engine.

---

## ✅ Features

- Upload `.m4a` files or ZIP folders
- Upload the Vosk model as a `.zip` file inside the app
- Offline transcription (no Whisper/OpenAI)
- Batch processing with progress bar
- Optional TTS (Text-to-Speech)
- Export transcriptions to `.txt` or `.pdf`
- Fully compatible with Python 3.13 and Streamlit Cloud

---

## 🧰 Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/streamlit-transcriber.git
cd streamlit-transcriber
