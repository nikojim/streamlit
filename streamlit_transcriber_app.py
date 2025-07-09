import streamlit as st
from vosk import Model, KaldiRecognizer
import subprocess
import wave
import json
import os
import pyttsx3
from fpdf import FPDF
import shutil
import zipfile
import tempfile

# === GLOBAL MODEL PATH ===
MODEL_DIR = None

# === HELPERS ===
def convert_to_wav(input_path, output_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", output_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def transcribe_with_vosk(wav_path, update_progress):
    model = Model(MODEL_DIR)
    rec = KaldiRecognizer(model, 16000)
    wf = wave.open(wav_path, "rb")

    results = []
    total = wf.getnframes()
    processed = 0

    while True:
        data = wf.readframes(4000)
        if not data:
            break
        processed += 4000
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            results.append(res.get("text", ""))
        update_progress(min(0.9, 0.3 + processed / total * 0.6))

    final = json.loads(rec.FinalResult())
    results.append(final.get("text", ""))
    return " ".join(results)

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("voice", "spanish")
    engine.say(text)
    engine.runAndWait()

def export_to_txt(text):
    with open("transcripcion.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return "transcripcion.txt"

def export_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output("transcripcion.pdf")
    return "transcripcion.pdf"

def handle_transcription(file_path, progress_cb):
    progress_cb(0.1)
    wav_path = "converted.wav"
    convert_to_wav(file_path, wav_path)
    return transcribe_with_vosk(wav_path, progress_cb)

# === STREAMLIT APP ===
st.set_page_config(page_title="🎙️ Transcriptor de Audio con Vosk", layout="centered")
st.title("🎧 Transcriptor (.m4a → Texto) con Vosk")

st.markdown("### 📦 Cargar modelo Vosk (ZIP)")
model_zip = st.file_uploader("Sube el modelo Vosk en formato .zip (ej: vosk-model-small-es-0.42.zip)", type="zip")

if model_zip:
    model_dir = tempfile.mkdtemp()
    with open(os.path.join(model_dir, "model.zip"), "wb") as f:
        f.write(model_zip.read())
    with zipfile.ZipFile(os.path.join(model_dir, "model.zip"), 'r') as zip_ref:
        zip_ref.extractall(model_dir)

    # Find the first directory that contains 'conf', 'am', etc.
    for root, dirs, files in os.walk(model_dir):
        if "conf" in dirs:
            MODEL_DIR = root
            break

    if MODEL_DIR:
        st.success("✅ Modelo cargado correctamente")
    else:
        st.error("❌ No se pudo encontrar un modelo Vosk válido en el ZIP")

if MODEL_DIR:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Subir audio (.m4a o .zip con varios)")
        uploaded_file = st.file_uploader("Archivo de audio o ZIP", type=["m4a", "zip"])

    with col2:
        tts = st.checkbox("🔈 Leer en voz alta", value=False)
        export_format = st.radio("💾 Exportar como", ["TXT", "PDF"])

    if uploaded_file:
        progress = st.progress(0.01)
        status = st.empty()
        texts = []

        def update_progress(p):
            progress.progress(p)

        temp_dir = tempfile.mkdtemp()
        input_paths = []

        if uploaded_file.name.endswith(".zip"):
            zip_path = os.path.join(temp_dir, "input.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_file.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            input_paths = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".m4a")]
        else:
            input_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())
            input_paths = [input_path]

        for idx, path in enumerate(input_paths):
            status.info(f"Transcribiendo archivo {idx+1}/{len(input_paths)}: {os.path.basename(path)}")
            try:
                result = handle_transcription(path, update_progress)
                texts.append((os.path.basename(path), result))
            except Exception as e:
                texts.append((os.path.basename(path), f"❌ Error: {e}"))

        progress.empty()
        status.success("✅ Transcripciones completas")

        full_text = ""
        for name, txt in texts:
            st.subheader(f"📝 {name}")
            st.text_area("Transcripción", txt, height=150)
            full_text += f"\n{name}\n{'-'*40}\n{txt}\n\n"
            if tts:
                speak(txt)

        st.markdown("---")
        if export_format == "TXT":
            out_path = export_to_txt(full_text)
        else:
            out_path = export_to_pdf(full_text)

        with open(out_path, "rb") as f:
            st.download_button("📥 Descargar transcripción", f, file_name=out_path)

        shutil.rmtree(temp_dir)
        if os.path.exists("converted.wav"):
            os.remove("converted.wav")
