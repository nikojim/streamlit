import streamlit as st
import tempfile
import os
import sys
from io import BytesIO
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import soundfile as sf
import whisper
import torch
import time
from collections import Counter

# Set page config
st.set_page_config(
    page_title="Whisper Audio Transcription",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available Whisper models optimized for large file processing
WHISPER_MODELS = {
    "tiny": {"size": "39 MB", "speed": "⚡ Muy rápido", "quality": "📊 Básica", "desc": "Ideal para archivos grandes y pruebas rápidas", "max_recommended": "60 min"},
    "base": {"size": "74 MB", "speed": "🚀 Rápido", "quality": "📈 Buena", "desc": "Balance óptimo para archivos de hasta 45 minutos", "max_recommended": "45 min"},
    "small": {"size": "244 MB", "speed": "⏱️ Medio", "quality": "📊 Muy buena", "desc": "Alta calidad para archivos de hasta 30 minutos", "max_recommended": "30 min"}
}

# File size limits
MAX_FILE_SIZE_MB = 150
RECOMMENDED_FILE_SIZE_MB = 50

@st.cache_resource
def load_whisper_model(model_name):
    """Load Whisper model with caching and progress tracking"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text(f"🔄 Descargando modelo Whisper '{model_name}'... (20%)")
        progress_bar.progress(0.2)
        
        # Check device availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        status_text.text(f"⚙️ Cargando en dispositivo: {device} (50%)")
        progress_bar.progress(0.5)
        
        # Load model
        model = whisper.load_model(model_name, device=device)
        
        progress_bar.progress(0.9)
        status_text.text("✅ Modelo cargado exitosamente (90%)")
        
        progress_bar.progress(1.0)
        status_text.text("✅ Modelo listo para usar (100%)")
        
        # Clean up progress components
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return model
        
    except Exception as e:
        st.error(f"❌ Error cargando modelo Whisper: {str(e)}")
        return None

def get_audio_info(file_path):
    """Get audio duration and sample rate using soundfile (librosa alternative)"""
    try:
        # Try with soundfile first (fastest and most reliable)
        info = sf.info(file_path)
        duration = info.duration
        sample_rate = info.samplerate
        return duration, sample_rate
    except Exception as e:
        try:
            # Fallback: Use whisper's built-in audio loading
            audio = whisper.load_audio(file_path)
            duration = len(audio) / 16000  # Whisper uses 16kHz
            sample_rate = 16000
            return duration, sample_rate
        except Exception as e2:
            st.warning(f"⚠️ No se pudo analizar el audio: {str(e2)}")
            return None, None

def prepare_audio_for_whisper(audio_file):
    """Prepare audio file for Whisper processing with enhanced compatibility"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📁 Guardando archivo temporal... (20%)")
        progress_bar.progress(0.2)
        
        # Create temporary file with proper extension
        file_extension = os.path.splitext(audio_file.name)[1].lower()
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Write file data
        if hasattr(audio_file, 'getvalue'):
            # File from session state (BytesIO)
            temp_input.write(audio_file.getvalue())
        else:
            # Regular uploaded file
            audio_file.seek(0)
            temp_input.write(audio_file.read())
        
        temp_input.close()
        
        status_text.text("🎵 Verificando formato de audio... (50%)")
        progress_bar.progress(0.5)
        
        # Get basic audio info first
        try:
            duration, sample_rate = get_audio_info(temp_input.name)
            if duration:
                st.info(f"🎵 Audio detectado: {duration/60:.1f} min, {sample_rate} Hz")
            else:
                st.warning("⚠️ No se pudo analizar el audio, pero continuando...")
        except Exception as e:
            st.warning(f"⚠️ Advertencia en análisis de audio: {str(e)}")
            duration = None
        
        status_text.text("🔧 Preparando audio para Whisper... (80%)")
        progress_bar.progress(0.8)
        
        # Test if Whisper can load the audio
        try:
            # Try to load a small sample first to catch format issues early
            test_audio = whisper.load_audio(temp_input.name)
            if len(test_audio) == 0:
                raise ValueError("El archivo de audio está vacío o no es válido")
            
            # Check if audio is too short
            if len(test_audio) < 1600:  # Less than 0.1 seconds at 16kHz
                raise ValueError("El archivo de audio es demasiado corto (menos de 0.1 segundos)")
            
            # Check if audio is too long for processing
            max_duration = 6 * 3600  # 6 hours max
            if len(test_audio) > max_duration * 16000:
                raise ValueError(f"El archivo de audio es demasiado largo (máximo {max_duration/3600} horas)")
                
        except Exception as e:
            os.unlink(temp_input.name)
            raise ValueError(f"Formato de audio no compatible: {str(e)}")
        
        status_text.text("✅ Audio preparado para transcripción (100%)")
        progress_bar.progress(1.0)
        
        # Clean up progress components
        time.sleep(0.5)
        progress_bar.empty()
        status_text.empty()
        
        return temp_input.name, duration
        
    except Exception as e:
        st.error(f"❌ Error preparando audio: {str(e)}")
        return None, None

def transcribe_audio_whisper(audio_file_path, model, language="es", audio_duration=None):
    """Transcribe audio using Whisper with enhanced error handling"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_info = st.empty()
        
        start_time = time.time()
        
        status_text.text("🎤 Iniciando transcripción con Whisper... (2%)")
        progress_bar.progress(0.02)
        
        # Show file information
        if audio_duration:
            if audio_duration > 3600:  # 1 hour
                time_info.text(f"📊 Procesando archivo grande: {audio_duration/3600:.1f} horas ({audio_duration/60:.0f} minutos)")
            elif audio_duration > 60:
                time_info.text(f"📊 Procesando: {audio_duration/60:.1f} minutos de audio")
            else:
                time_info.text(f"📊 Procesando: {audio_duration:.0f} segundos de audio")
        else:
            time_info.text("📊 Analizando archivo de audio...")
        
        status_text.text("🔊 Cargando audio con Whisper... (10%)")
        progress_bar.progress(0.10)
        
        # Load audio using Whisper with error handling
        try:
            audio_data = whisper.load_audio(audio_file_path)
            
            # Validate audio data
            if len(audio_data) == 0:
                raise ValueError("El archivo de audio está vacío")
            
            # Check for very short audio
            if len(audio_data) < 1600:  # Less than 0.1 seconds
                raise ValueError("El archivo de audio es demasiado corto")
                
            status_text.text(f"🔊 Audio cargado: {len(audio_data)/16000:.1f} segundos (15%)")
            progress_bar.progress(0.15)
            
        except Exception as e:
            raise ValueError(f"Error cargando audio: {str(e)}")
        
        # Configure transcription options with safer defaults
        transcribe_options = {
            "language": language if language != "auto" else None,
            "task": "transcribe",
            "verbose": False,
            "temperature": 0.0,  # Deterministic output
            "best_of": 1,  # Reduce complexity
            "beam_size": 1,  # Reduce complexity to avoid tensor issues
            "patience": 1.0,
            "length_penalty": 1.0,
            "suppress_tokens": "-1",  # Don't suppress any tokens
            "initial_prompt": None,  # No initial prompt to avoid conflicts
            "condition_on_previous_text": True,
            "fp16": torch.cuda.is_available(),  # Use FP16 only on GPU
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
        }
        
        # Simplify options for problematic files
        if audio_duration and audio_duration > 1800:  # For files > 30 minutes
            transcribe_options.update({
                "beam_size": 1,  # Most conservative
                "best_of": 1,
                "temperature": 0.0,
                "condition_on_previous_text": False,  # Reduce complexity
            })
        
        status_text.text("🌍 Detectando idioma... (20%)")
        progress_bar.progress(0.20)
        
        # Detect language safely
        try:
            # Pad or trim to avoid tensor size issues
            audio_sample = whisper.pad_or_trim(audio_data)
            mel = whisper.log_mel_spectrogram(audio_sample).to(model.device)
            
            # Detect language
            _, probs = model.detect_language(mel)
            detected_lang = max(probs, key=probs.get)
            
            if language == "auto":
                language = detected_lang
                status_text.text(f"🌍 Idioma detectado: {detected_lang} (25%)")
            else:
                status_text.text(f"🌍 Idioma: {language} (detectado: {detected_lang}) (25%)")
                
        except Exception as e:
            st.warning(f"⚠️ No se pudo detectar idioma: {str(e)}. Usando configuración por defecto.")
            if language == "auto":
                language = "es"  # Default to Spanish
        
        progress_bar.progress(0.30)
        status_text.text("🎯 Iniciando transcripción principal... (30%)")
        
        # Progress simulation for long files
        if audio_duration and audio_duration > 300:  # 5 minutes
            progress_updates = min(15, int(audio_duration / 60))  # Max 15 updates
            
            for i in range(progress_updates):
                current_progress = 0.30 + (0.5 * (i + 1) / progress_updates)
                progress_percentage = int(current_progress * 100)
                progress_bar.progress(current_progress)
                
                elapsed = time.time() - start_time
                estimated_total = audio_duration * 0.25 if audio_duration else elapsed * 3
                remaining = max(0, estimated_total - elapsed)
                
                status_text.text(f"🎯 Transcribiendo... {progress_percentage}% - Tiempo restante: ~{remaining:.0f}s")
                time.sleep(0.5)  # Short delay for UI updates
        
        # Perform actual transcription with error handling
        try:
            status_text.text("🎯 Ejecutando transcripción Whisper... (80%)")
            progress_bar.progress(0.80)
            
            result = model.transcribe(audio_file_path, **transcribe_options)
            
        except Exception as e:
            # Try with even simpler options if transcription fails
            st.warning(f"⚠️ Error en transcripción, intentando con configuración simplificada...")
            
            simple_options = {
                "language": language if language != "auto" else "es",
                "task": "transcribe",
                "verbose": False,
                "temperature": 0.0,
                "beam_size": 1,
                "best_of": 1,
                "fp16": False,  # Disable FP16 as fallback
            }
            
            try:
                result = model.transcribe(audio_file_path, **simple_options)
            except Exception as e2:
                raise RuntimeError(f"Falló la transcripción incluso con configuración simplificada: {str(e2)}")
        
        progress_bar.progress(0.90)
        status_text.text("📝 Procesando resultados... (90%)")
        
        # Extract and validate results
        transcription = result.get("text", "").strip()
        if not transcription:
            raise ValueError("La transcripción resultó vacía. Verifica que el audio contenga voz clara.")
        
        segments = result.get("segments", [])
        language_result = result.get("language", language)
        
        # Post-process for large files
        if len(transcription) > 10000:  # Large transcription
            status_text.text("📝 Optimizando texto largo... (95%)")
            # Add paragraph breaks for better readability
            sentences = transcription.split('. ')
            paragraphs = []
            current_paragraph = []
            
            for sentence in sentences:
                current_paragraph.append(sentence)
                if len(' '.join(current_paragraph)) > 500:  # ~500 characters per paragraph
                    paragraphs.append('. '.join(current_paragraph) + '.')
                    current_paragraph = []
            
            if current_paragraph:
                paragraphs.append('. '.join(current_paragraph))
            
            transcription = '\n\n'.join(paragraphs)
        
        # Calculate final statistics
        word_count = len(transcription.split())
        char_count = len(transcription)
        elapsed_time = time.time() - start_time
        
        progress_bar.progress(1.0)
        
        # Display comprehensive statistics
        if audio_duration:
            speed_factor = audio_duration / elapsed_time
            status_text.text(f"✅ Transcripción completada - {word_count:,} palabras - {speed_factor:.1f}x velocidad (100%)")
            
            if audio_duration > 3600:
                time_info.text(f"⏱️ {elapsed_time/60:.1f} min de procesamiento para {audio_duration/3600:.1f} horas de audio")
            else:
                time_info.text(f"⏱️ {elapsed_time/60:.1f} min de procesamiento para {audio_duration/60:.1f} min de audio")
        else:
            status_text.text(f"✅ Transcripción completada - {word_count:,} palabras (100%)")
            time_info.text(f"⏱️ Tiempo de procesamiento: {elapsed_time/60:.1f} minutos")
        
        # Show detailed statistics for large transcriptions
        if segments:
            avg_confidence = sum(segment.get("avg_logprob", 0) for segment in segments) / len(segments)
            st.success(f"📊 {len(segments)} segmentos | 🌍 Idioma: {language_result} | 📝 {char_count:,} caracteres | 🎯 Confianza promedio: {avg_confidence:.2f}")
        
        # Clean up progress components
        time.sleep(3)
        progress_bar.empty()
        status_text.empty()
        time_info.empty()
        
        return transcription, result
        
    except Exception as e:
        # Clean up on error
        try:
            progress_bar.empty()
            status_text.empty()
            time_info.empty()
        except:
            pass
        
        # Provide specific error guidance
        error_msg = str(e)
        if "tensor" in error_msg.lower():
            st.error("❌ Error de procesamiento de audio. Esto puede deberse a:")
            st.markdown("""
            - **Formato de audio incompatible**: Intenta convertir a MP3 o WAV
            - **Archivo corrupto**: Verifica que el audio se reproduzca correctamente
            - **Modelo incompatible**: Prueba con el modelo 'tiny' o 'base'
            - **Archivo muy corto**: El audio debe tener al menos 1 segundo
            """)
        else:
            st.error(f"❌ Error durante transcripción: {error_msg}")
        
        return None, None

def generate_text_summary(text, max_sentences=3):
    """Generate a smart summary of the transcribed text"""
    try:
        # Split text into sentences
        sentences = text.replace('\n', ' ').split('. ')
        sentences = [s.strip() + '.' for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) <= max_sentences:
            return '. '.join(sentences)
        
        # For longer texts, try to extract key sentences
        # Look for sentences with common important indicators
        important_keywords = [
            'objetivo', 'propósito', 'importante', 'principal', 'resultado', 'conclusión',
            'problema', 'solución', 'decidir', 'acuerdo', 'plan', 'estrategia',
            'resumen', 'summary', 'conclusion', 'important', 'main', 'key'
        ]
        
        # Score sentences based on position and keywords
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # First and last sentences get higher scores
            if i == 0:
                score += 3
            elif i == len(sentences) - 1:
                score += 2
            elif i < 3:  # Early sentences
                score += 1
            
            # Check for important keywords
            sentence_lower = sentence.lower()
            for keyword in important_keywords:
                if keyword in sentence_lower:
                    score += 2
            
            # Prefer sentences of medium length
            if 50 < len(sentence) < 200:
                score += 1
            
            sentence_scores.append((sentence, score))
        
        # Sort by score and take top sentences
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s[0] for s in sentence_scores[:max_sentences]]
        
        # Maintain original order
        summary_sentences = []
        for sentence in sentences:
            if sentence in top_sentences:
                summary_sentences.append(sentence)
                if len(summary_sentences) >= max_sentences:
                    break
        
        return ' '.join(summary_sentences)
        
    except Exception as e:
        # Fallback to simple first sentences
        sentences = text.split('. ')[:max_sentences]
        return '. '.join(sentences) + '.'

def create_pdf(text, metadata=None):
    """Create PDF from transcription text"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📄 Creando documento PDF... (20%)")
        progress_bar.progress(0.2)
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=18,
            spaceAfter=20,
            textColor='#1f77b4'
        )
        
        spanish_style = ParagraphStyle(
            'Spanish',
            parent=styles['Normal'],
            fontSize=12,
            leading=16,
            spaceAfter=12,
            encoding='utf-8',
            alignment=0  # Left alignment
        )
        
        progress_bar.progress(0.4)
        status_text.text("📝 Estructurando contenido... (40%)")
        
        # Build document
        story = []
        story.append(Paragraph("Transcripción de Audio con Whisper", title_style))
        story.append(Spacer(1, 0.2*inch))
        
        # Add metadata if available
        if metadata:
            story.append(Paragraph("Información del Procesamiento:", styles['Heading2']))
            story.append(Paragraph(f"Modelo utilizado: {metadata.get('model', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Idioma detectado: {metadata.get('language', 'N/A')}", styles['Normal']))
            story.append(Paragraph(f"Palabras transcritas: {metadata.get('word_count', 'N/A')}", styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
        
        # Add summary if text is long enough
        if len(text.split()) > 50:
            progress_bar.progress(0.5)
            status_text.text("📋 Generando resumen automático... (50%)")
            
            summary = generate_text_summary(text, max_sentences=3)
            story.append(Paragraph("Resumen Ejecutivo:", styles['Heading2']))
            story.append(Paragraph(summary, spanish_style))
            story.append(Spacer(1, 0.2*inch))
        
        story.append(Paragraph("Transcripción Completa:", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        
        progress_bar.progress(0.7)
        status_text.text("✍️ Procesando texto... (70%)")
        
        # Split text into paragraphs
        paragraphs = text.split('\n\n')
        if len(paragraphs) == 1:
            # Split by sentences if no paragraph breaks
            sentences = text.split('. ')
            paragraphs = ['. '.join(sentences[i:i+3]) for i in range(0, len(sentences), 3)]
        
        progress_bar.progress(0.8)
        status_text.text("📋 Generando párrafos... (80%)")
        
        for para in paragraphs:
            if para.strip():
                story.append(Paragraph(para.strip(), spanish_style))
                story.append(Spacer(1, 0.1*inch))
        
        progress_bar.progress(0.95)
        status_text.text("🔨 Construyendo PDF final... (95%)")
        
        doc.build(story)
        buffer.seek(0)
        
        progress_bar.progress(1.0)
        status_text.text("✅ PDF generado exitosamente (100%)")
        
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        return buffer
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error generando PDF: {str(e)}")
        return None

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = {
        'whisper': 'openai-whisper',
        'torch': 'torch',
        'soundfile': 'soundfile',
        'numpy': 'numpy',
        'reportlab': 'reportlab'
    }
    
    missing_packages = []
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        st.error(f"❌ Paquetes faltantes: {', '.join(missing_packages)}")
        st.code(f"pip install {' '.join(missing_packages)}")
        st.stop()

def main():
    st.title("🎤 Transcripción de Audio con Whisper")
    st.markdown("**Transcripción de audio usando OpenAI Whisper** - Optimizado para archivos grandes hasta 150MB")
    
    # Check requirements
    check_requirements()
    
    # Initialize session state variables
    if 'transcription_in_progress' not in st.session_state:
        st.session_state.transcription_in_progress = False
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None
    if 'transcription_metadata' not in st.session_state:
        st.session_state.transcription_metadata = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'uploaded_file_data' not in st.session_state:
        st.session_state.uploaded_file_data = None
    if 'uploaded_file_type' not in st.session_state:
        st.session_state.uploaded_file_type = None
    if 'whisper_model' not in st.session_state:
        st.session_state.whisper_model = None
    if 'whisper_model_name' not in st.session_state:
        st.session_state.whisper_model_name = None
    if 'language' not in st.session_state:
        st.session_state.language = 'es'
    
    # System info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(f"🐍 Python {sys.version.split()[0]}")
    with col2:
        device_info = "🚀 GPU" if torch.cuda.is_available() else "💻 CPU"
        st.info(f"Dispositivo: {device_info}")
    with col3:
        st.info("🌐 Streamlit Cloud")
    
    # Show transcription status if in progress
    if st.session_state.transcription_in_progress:
        st.warning("🔄 **Transcripción en progreso...** Por favor, no actualices la página ni cierres la pestaña.")
        st.info("💡 La transcripción puede tomar varios minutos. El progreso se mostrará abajo.")
        
        # Add a stop button for long transcriptions
        if st.button("⏹️ Cancelar Transcripción", type="secondary"):
            st.session_state.transcription_in_progress = False
            st.session_state.transcription_result = None
            st.session_state.transcription_metadata = None
            st.rerun()
    
    # Model selection (only if not transcribing)
    if not st.session_state.transcription_in_progress:
        st.subheader("🎛️ Configuración del Modelo")
        
        # Model selector with detailed info
        model_choice = st.selectbox(
            "Selecciona el modelo Whisper:",
            options=list(WHISPER_MODELS.keys()),
            format_func=lambda x: f"{x.title()} ({WHISPER_MODELS[x]['size']}) - {WHISPER_MODELS[x]['speed']} - {WHISPER_MODELS[x]['quality']}",
            index=1,  # Default to 'base'
            help="Modelos más grandes ofrecen mejor calidad pero son más lentos",
            key="model_selector"
        )
        
        # Show model details with large file recommendations
        model_info = WHISPER_MODELS[model_choice]
        col1, col2 = st.columns([2, 1])
        with col1:
            st.info(f"📋 {model_info['desc']}")
        with col2:
            st.info(f"💾 {model_info['size']} | 📊 Max recomendado: {model_info['max_recommended']}")
        
        # Large file processing tips
        if model_choice == "small":
            st.warning("⚠️ Modelo 'small' recomendado solo para archivos <30 min en Streamlit Cloud")
        elif model_choice == "tiny":
            st.success("✅ Modelo 'tiny' optimizado para archivos grandes (hasta 1 hora)")
        
        # Language selection
        language = st.selectbox(
            "Idioma de transcripción:",
            options=["es", "en", "auto"],
            format_func=lambda x: {"es": "🇪🇸 Español", "en": "🇺🇸 Inglés", "auto": "🌍 Detección automática"}[x],
            index=0,
            help="Selecciona el idioma del audio o usa detección automática",
            key="language_selector"
        )
        
        # Store language in session state
        st.session_state.language = language
        
        # Load model
        if (st.session_state.whisper_model is None or 
            st.session_state.whisper_model_name != model_choice):
            
            if st.button("🔄 Cargar Modelo", type="primary", key="load_model_btn"):
                with st.spinner(f"Cargando modelo {model_choice}..."):
                    model = load_whisper_model(model_choice)
                    if model:
                        st.session_state.whisper_model = model
                        st.session_state.whisper_model_name = model_choice
                        st.success(f"✅ Modelo '{model_choice}' cargado exitosamente")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("❌ Error cargando el modelo")
        else:
            # Model is already loaded
            st.success(f"✅ Modelo '{st.session_state.whisper_model_name}' ya está cargado")
        
        # Check if model is loaded
        if st.session_state.whisper_model is None:
            st.warning("⚠️ Primero carga un modelo para continuar")
            return
    
    # Display previous transcription results if available
    if (st.session_state.transcription_result and 
        not st.session_state.transcription_in_progress):
        
        st.success("✅ **Transcripción anterior disponible**")
        
        transcription = st.session_state.transcription_result
        metadata = st.session_state.transcription_metadata
        
        # Display the results
        display_transcription_results(transcription, metadata)
        
        # Option to start new transcription
        if st.button("🔄 Nueva Transcripción", type="secondary"):
            st.session_state.transcription_result = None
            st.session_state.transcription_metadata = None
            st.session_state.uploaded_file_name = None
            st.session_state.uploaded_file_data = None
            st.session_state.uploaded_file_type = None
            st.rerun()
        
        st.markdown("---")
    
    # File upload section (only if not transcribing)
    uploaded_file = None  # Initialize variable
    
    if not st.session_state.transcription_in_progress:
        st.subheader("📁 Subir Archivo de Audio")
        
        uploaded_file = st.file_uploader(
            "Selecciona tu archivo de audio (máximo 150 MB)",
            type=['wav', 'mp3', 'm4a', 'flac', 'aac', 'ogg'],
            help="Formatos soportados: WAV, MP3, M4A, FLAC, AAC, OGG. Límite: 150MB",
            key="audio_file_uploader"
        )
        
        if uploaded_file is not None:
            # File info and validation
            file_size_mb = uploaded_file.size / 1024 / 1024
            st.write(f"📄 **Archivo:** {uploaded_file.name}")
            st.write(f"📊 **Tamaño:** {file_size_mb:.2f} MB")
            
            # File size validation
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"❌ El archivo excede el límite de {MAX_FILE_SIZE_MB}MB. Tamaño actual: {file_size_mb:.1f}MB")
                st.info("💡 **Sugerencias para reducir el tamaño:**")
                st.markdown("""
                - Convierte a MP3 con menor bitrate (128kbps)
                - Reduce la calidad de audio si no es crítica
                - Divide el archivo en partes más pequeñas
                - Usa formato comprimido como AAC
                """)
                return
            
            # File size warnings and recommendations
            if file_size_mb > RECOMMENDED_FILE_SIZE_MB:
                st.warning(f"⚠️ **Archivo grande detectado ({file_size_mb:.1f}MB)**")
                
                # Estimate processing time
                estimated_duration = file_size_mb * 2  # Rough estimate: 2 minutes per MB
                estimated_processing = estimated_duration * 0.2  # 20% of duration for processing
                
                st.info(f"⏱️ **Tiempo estimado de procesamiento:** {estimated_processing:.0f}-{estimated_processing*2:.0f} minutos")
                
                # Model recommendations for large files
                if st.session_state.whisper_model_name == "small" and file_size_mb > 80:
                    st.warning("🔄 **Recomendación:** Considera usar modelo 'base' o 'tiny' para archivos >80MB")
                elif st.session_state.whisper_model_name == "base" and file_size_mb > 120:
                    st.warning("🔄 **Recomendación:** Considera usar modelo 'tiny' para archivos >120MB")
                    
            elif file_size_mb > 20:
                st.info(f"📊 Archivo mediano detectado. Tiempo estimado: 3-8 minutos de procesamiento.")
            
            # Audio player
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            # Transcription section
            if st.button("🎯 Transcribir Audio", type="primary", key="transcribe_btn"):
                # Set transcription in progress
                st.session_state.transcription_in_progress = True
                st.session_state.uploaded_file_name = uploaded_file.name
                # Store file data in session state for processing
                st.session_state.uploaded_file_data = uploaded_file.getvalue()
                st.session_state.uploaded_file_type = uploaded_file.type
                st.rerun()
    
    # Perform transcription if in progress
    if st.session_state.transcription_in_progress:
        # Check if we have stored file data
        if ('uploaded_file_data' in st.session_state and 
            'uploaded_file_name' in st.session_state and
            'uploaded_file_type' in st.session_state):
            
            # Recreate file-like object from stored data
            from io import BytesIO
            file_data = BytesIO(st.session_state.uploaded_file_data)
            file_data.name = st.session_state.uploaded_file_name
            file_data.type = st.session_state.uploaded_file_type
            file_data.size = len(st.session_state.uploaded_file_data)
            
            perform_transcription(file_data, st.session_state.whisper_model, st.session_state.get('language', 'es'))
        else:
            st.error("❌ No se encontraron datos del archivo. Por favor, sube el archivo nuevamente.")
            st.session_state.transcription_in_progress = False
            st.rerun()

def perform_transcription(uploaded_file, model, language):
    """Perform the actual transcription in a separate function to maintain state"""
    
    file_size_mb = uploaded_file.size / 1024 / 1024
    
    try:
        st.info(f"🚀 Iniciando transcripción con modelo '{st.session_state.whisper_model_name}' para archivo de {file_size_mb:.1f}MB")
        
        # Large file warning
        if file_size_mb > 100:
            st.warning("⚠️ **Archivo muy grande detectado.** El procesamiento puede tomar 30+ minutos. **NO CIERRES ESTA PESTAÑA.**")
        
        # Step 1: Prepare audio
        st.subheader("📁 Paso 1: Preparación del Audio")
        audio_result = prepare_audio_for_whisper(uploaded_file)
        
        if audio_result and len(audio_result) == 2:
            audio_path, duration = audio_result
            
            if audio_path:
                # Step 2: Transcribe
                st.subheader("🎤 Paso 2: Transcripción con Whisper")
                
                # Show processing estimates for large files
                if duration and duration > 1800:  # 30 minutes
                    st.info(f"⏳ **Procesando archivo de {duration/3600:.1f} horas. Esto tomará aproximadamente {duration*0.15/60:.0f}-{duration*0.3/60:.0f} minutos.**")
                
                transcription, result = transcribe_audio_whisper(
                    audio_path, 
                    model, 
                    language,
                    duration
                )
                
                # Clean up
                os.unlink(audio_path)
                
                if transcription:
                    # Store results in session state
                    st.session_state.transcription_result = transcription
                    st.session_state.transcription_metadata = {
                        'model': st.session_state.whisper_model_name,
                        'language': result.get("language", "N/A") if result else "N/A",
                        'word_count': len(transcription.split()),
                        'duration': f"{duration/60:.1f} min" if duration else "N/A",
                        'file_size': f"{file_size_mb:.1f} MB",
                        'file_name': st.session_state.uploaded_file_name,
                        'result': result
                    }
                    
                    # Mark transcription as complete
                    st.session_state.transcription_in_progress = False
                    
                    st.success("🎉 ¡Transcripción completada exitosamente!")
                    st.balloons()
                    
                    # Force refresh to show results
                    st.rerun()
                    
                else:
                    st.session_state.transcription_in_progress = False
                    st.error("❌ No se pudo transcribir el audio")
                    st.info("💡 **Posibles soluciones:**")
                    st.markdown("""
                    - Verifica que el archivo contenga audio válido
                    - Intenta con un modelo diferente (tiny para archivos grandes)
                    - Asegúrate de que el audio tenga voz clara
                    - Para archivos muy grandes, considera dividirlos en partes
                    """)
            else:
                st.session_state.transcription_in_progress = False
                st.error("❌ Error al procesar el archivo de audio")
        else:
            st.session_state.transcription_in_progress = False
            st.error("❌ Error en la preparación del audio")
    
    except Exception as e:
        st.session_state.transcription_in_progress = False
        st.error(f"❌ Error durante la transcripción: {str(e)}")

def display_transcription_results(transcription, metadata):
    """Display transcription results without problematic features"""
    
    # Generate and display automatic summary at the beginning
    if len(transcription.split()) > 50:
        st.subheader("📋 Resumen Automático")
        summary = generate_text_summary(transcription, max_sentences=3)
        
        st.info("💡 **Resumen generado automáticamente de los puntos más importantes:**")
        st.markdown(f"*{summary}*")
        
        # Summary statistics
        original_words = len(transcription.split())
        summary_words = len(summary.split())
        compression_ratio = (summary_words / original_words) * 100
        
        st.caption(f"📊 Resumen: {summary_words} palabras de {original_words:,} originales ({compression_ratio:.1f}% del texto original)")
        st.markdown("---")
    
    # Display transcription
    st.subheader("📝 Transcripción Completa:")
    text_height = min(max(300, len(transcription) // 100), 600)
    st.text_area(
        "Texto transcrito",
        transcription,
        height=text_height,
        help="Puedes seleccionar y copiar el texto. Para textos largos, usa el scroll.",
        label_visibility="visible",
        key="transcription_final_display"
    )
    
    # Statistics
    word_count = metadata.get('word_count', len(transcription.split()))
    char_count = len(transcription)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Palabras", f"{word_count:,}")
    with col2:
        st.metric("Caracteres", f"{char_count:,}")
    with col3:
        st.metric("Idioma", metadata.get('language', 'N/A'))
    with col4:
        result = metadata.get('result')
        if result:
            segments = result.get("segments", [])
            st.metric("Segmentos", len(segments))
    
    # Simple download section - ONLY TXT
    st.subheader("💾 Descargar Transcripción")
    
    # Only TXT download - no PDF
    txt_data = transcription.encode('utf-8')
    file_size_kb = len(txt_data) / 1024
    st.download_button(
        label=f"📄 Descargar TXT ({file_size_kb:.1f} KB)",
        data=txt_data,
        file_name=f"transcripcion_whisper_{metadata.get('file_name', 'audio')}.txt",
        mime="text/plain",
        help="Archivo de texto plano compatible con cualquier editor",
        key="download_txt_only"
    )
    
    # Simple additional tools - no complex state management
    if word_count > 1000:
        with st.expander("🔧 Herramientas de Análisis"):
            st.markdown("### 📊 Estadísticas Adicionales")
            
            # Simple word analysis without complex state
            words = transcription.lower().split()
            stop_words = {
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 
                'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'una', 'como', 
                'pero', 'sus', 'han', 'hay', 'esta', 'este', 'más', 'muy', 'todo', 'ya'
            }
            
            filtered_words = [
                word.strip('.,!?;:"()[]¿¡') for word in words 
                if len(word) > 2 and word.lower() not in stop_words and word.isalpha()
            ]
            
            if filtered_words:
                word_freq = Counter(filtered_words)
                top_words = word_freq.most_common(10)
                
                st.write("**🔤 Top 10 Palabras Más Frecuentes:**")
                for word, count in top_words:
                    percentage = (count / len(filtered_words)) * 100
                    st.write(f"• **{word.title()}**: {count} veces ({percentage:.1f}%)")
                
                # Simple statistics
                unique_words = len(set(filtered_words))
                vocabulary_richness = (unique_words / len(filtered_words)) * 100
                
                col1, col2 = st.columns(2)
                col1.metric("Palabras únicas", f"{unique_words:,}")
                col2.metric("Riqueza vocabulario", f"{vocabulary_richness:.1f}%")success("🎉 ¡Transcripción completada exitosamente!")
    
    # Enhanced documentation for large files
    with st.expander("📚 Guía para Archivos Grandes (hasta 150MB)"):
        st.markdown("""
        ### 🎯 Recomendaciones por Tamaño de Archivo:
        
        | Tamaño | Duración Aprox. | Modelo Recomendado | Tiempo Estimado | Calidad |
        |--------|-----------------|-------------------|-----------------|---------|
        | **< 25MB** | < 30 min | Base/Small | 2-5 min | ⭐⭐⭐⭐ |
        | **25-75MB** | 30-90 min | Base/Tiny | 5-15 min | ⭐⭐⭐ |
        | **75-150MB** | 90+ min | **Tiny** | 15-45 min | ⭐⭐ |
        
        ### 🚀 Optimización para Archivos Grandes:
        
        **✅ Preparación del archivo:**
        - Usa formatos comprimidos (MP3, AAC) en lugar de WAV
        - Bitrate recomendado: 128-192 kbps para voz
        - Mono en lugar de estéreo para archivos de voz
        
        **⚙️ Configuración recomendada:**
        - **Archivos >60 min**: Usa modelo 'tiny' para velocidad
        - **Audio con ruido**: Usa modelo 'base' o 'small' para mejor precisión
        - **Múltiples idiomas**: Selecciona 'detección automática'
        
        **⏱️ Tiempos de procesamiento esperados:**
        - Modelo Tiny: ~15% del tiempo del audio
        - Modelo Base: ~25% del tiempo del audio  
        - Modelo Small: ~40% del tiempo del audio
        
        **💡 Consejos para evitar interrupciones:**
        - **NUNCA actualices la página** durante la transcripción
        - **Mantén la pestaña activa** (no la minimices)
        - **Cierra otras pestañas pesadas** para liberar memoria
        - **Conecta el cargador** si usas un portátil
        - **Usa una conexión estable** a internet
        
        ### 🔧 Solución de Problemas:
        
        **Si la app se actualiza durante la transcripción:**
        - La transcripción se perderá y deberás reiniciar
        - Para archivos grandes, considera dividirlos en partes más pequeñas
        - Usa el modelo 'tiny' para transcripciones más rápidas
        
        **Si el procesamiento falla:**
        1. Intenta con modelo 'tiny' (más rápido, menos memoria)
        2. Reduce el tamaño del archivo (comprimir audio)
        3. Divide el archivo en partes más pequeñas
        4. Verifica que el navegador tenga suficiente memoria
        """)

if __name__ == "__main__":
    main()