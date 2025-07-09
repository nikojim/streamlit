# 🎙️ Audio Transcription App (Vosk + Whisper)

This Streamlit app allows you to transcribe `.m4a` audio files (or zip folders of them) to text using:

- 🧠 [Vosk](https://alphacephei.com/vosk/) (offline, fast, lightweight)
- 🤖 [OpenAI Whisper](https://github.com/openai/whisper) (fallback)

Supports:
- ✅ Single and batch `.m4a` file uploads
- ✅ Real-time progress bar and status
- ✅ Text-to-Speech output (TTS)
- ✅ Export to `.txt` or `.pdf`
- ✅ 100% compatible with **Python 3.13**
- ✅ Ready for **Streamlit Cloud deployment**

---

## 🚀 Features

| Feature                | Vosk       | Whisper     |
|------------------------|------------|-------------|
| Offline transcription  | ✅         | ❌ (uses CPU/GPU) |
| Spanish language       | ✅         | ✅          |
| Multiple files support | ✅         | ✅          |
| TTS voice playback     | ✅         | ✅          |
| Export to `.txt`/`.pdf`| ✅         | ✅          |

---

## 🧰 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/audio-transcriber-app.git
cd audio-transcriber-app
