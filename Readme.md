# 🎤 Whisper Audio Transcription App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange.svg)](https://github.com/openai/whisper)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A powerful **Streamlit web application** for transcribing audio files using **OpenAI's Whisper** models. Optimized for large files up to **150MB** with real-time progress tracking and professional output formats.

## 🌟 Features

- **🎯 Multiple Whisper Models**: Choose from Tiny, Base, and Small models
- **📁 Large File Support**: Handle audio files up to 150MB
- **🌍 Multi-language**: Spanish, English, and auto-detection
- **⚡ Real-time Progress**: Detailed progress tracking with time estimates
- **📊 Smart Analytics**: Word count, confidence scoring, and quality metrics
- **💾 Multiple Export Formats**: Download as TXT or formatted PDF
- **🔧 Intelligent Optimization**: Model recommendations based on file size
- **📱 Responsive Design**: Works on desktop and mobile devices

## 🚀 Live Demo

Try the app online: **[your-app-url.streamlit.app](https://your-app-url.streamlit.app)**

## 📋 Table of Contents

- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Formats](#-supported-formats)
- [Model Comparison](#-model-comparison)
- [Performance Guide](#-performance-guide)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Contributing](#-contributing)
- [License](#-license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large files)
- Internet connection (for model downloads)

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/whisper-transcription-app.git
   cd whisper-transcription-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

### Docker Setup

```bash
# Build the image
docker build -t whisper-app .

# Run the container
docker run -p 8501:8501 whisper-app
```

## 🎯 Usage

### Quick Start

1. **Select a Whisper model** (Base recommended for most users)
2. **Choose the language** (Spanish, English, or auto-detect)
3. **Load the model** by clicking "🔄 Cargar Modelo"
4. **Upload your audio file** (up to 150MB)
5. **Click "🎯 Transcribir Audio"** and wait for completion
6. **Download results** as TXT or PDF

### Step-by-Step Guide

#### 1. Model Selection
Choose the appropriate model based on your needs:

- **Tiny (39MB)**: Fast processing, good for large files
- **Base (74MB)**: Balanced speed and quality ⭐ **Recommended**
- **Small (244MB)**: Best quality, slower processing

#### 2. File Upload
Supported formats: WAV, MP3, M4A, FLAC, AAC, OGG

**File Size Guidelines:**
- ✅ **< 25MB**: Optimal performance
- ⚠️ **25-75MB**: Medium processing time
- 🔥 **75-150MB**: Longer processing (use Tiny model)

#### 3. Processing
The app will show detailed progress including:
- Audio preparation and validation
- Real-time transcription progress
- Time estimates and completion statistics
- Quality indicators and confidence scores

#### 4. Results
Get comprehensive results with:
- Full transcription text
- Word and character counts
- Language detection results
- Quality metrics and confidence scores

## 📁 Supported Formats

| Format | Extension | Recommended | Notes |
|--------|-----------|-------------|-------|
| **MP3** | `.mp3` | ✅ **Best** | Compressed, good quality |
| **M4A** | `.m4a` | ✅ **Good** | Apple format, efficient |
| **WAV** | `.wav` | ⚠️ **Large** | Uncompressed, high quality |
| **FLAC** | `.flac` | ⚠️ **Large** | Lossless compression |
| **AAC** | `.aac` | ✅ **Good** | Efficient compression |
| **OGG** | `.ogg` | ✅ **Good** | Open source format |

## 🔄 Model Comparison

| Model | Size | Speed | Quality | Best For | Max Duration* |
|-------|------|-------|---------|----------|---------------|
| **Tiny** | 39 MB | ⚡⚡⚡ | ⭐⭐ | Large files, demos | 6+ hours |
| **Base** | 74 MB | ⚡⚡ | ⭐⭐⭐ | General use | 3-4 hours |
| **Small** | 244 MB | ⚡ | ⭐⭐⭐⭐ | High quality needs | 1-2 hours |

*Recommended maximum duration for optimal performance

## ⚡ Performance Guide

### Processing Time Estimates

| Audio Duration | Tiny Model | Base Model | Small Model |
|----------------|------------|------------|-------------|
| **5 minutes** | 1-2 min | 2-3 min | 3-5 min |
| **30 minutes** | 5-8 min | 8-12 min | 12-20 min |
| **1 hour** | 10-15 min | 15-25 min | 25-40 min |
| **2 hours** | 20-30 min | 30-50 min | 50-80 min |

### Optimization Tips

#### For Large Files (>50MB):
1. **Use Tiny model** for faster processing
2. **Compress audio** to MP3 format (128-192 kbps)
3. **Convert stereo to mono** for voice recordings
4. **Remove silence** from beginning/end
5. **Keep browser tab active** during processing

#### For Best Quality:
1. **Use Small model** for critical transcriptions
2. **Ensure clear audio** without background noise
3. **Use higher bitrate** source files
4. **Select correct language** instead of auto-detect

## 🌐 Deployment

### Streamlit Cloud

1. **Fork this repository**
2. **Connect to Streamlit Cloud**
3. **Configure deployment**:
   - Repository: `yourusername/whisper-transcription-app`
   - Branch: `main`
   - Main file: `app.py`
4. **Deploy** and get your public URL

#### Required Files for Deployment:
```
├── app.py                 # Main application
├── requirements.txt       # Python dependencies
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── README.md             # This file
└── .gitignore           # Git ignore rules
```

### Environment Variables

Create a `.streamlit/secrets.toml` file for sensitive configurations:

```toml
[general]
max_file_size_mb = 150
enable_analytics = true
```

### Custom Domain

To use a custom domain with Streamlit Cloud:

1. Set up your domain DNS to point to Streamlit
2. Configure the domain in your Streamlit Cloud dashboard
3. Update the badge URL in this README

## 🔧 API Reference

### Main Functions

#### `load_whisper_model(model_name: str)`
Loads and caches a Whisper model.

**Parameters:**
- `model_name`: Model size ("tiny", "base", "small")

**Returns:**
- Loaded Whisper model object

#### `transcribe_audio_whisper(audio_path: str, model, language: str, duration: float)`
Transcribes audio using the specified model.

**Parameters:**
- `audio_path`: Path to audio file
- `model`: Loaded Whisper model
- `language`: Target language ("es", "en", "auto")
- `duration`: Audio duration in seconds

**Returns:**
- Tuple of (transcription_text, result_dict)

### Configuration

#### File Size Limits
```python
MAX_FILE_SIZE_MB = 150      # Maximum upload size
RECOMMENDED_FILE_SIZE_MB = 50  # Recommended size for optimal performance
```

#### Model Settings
```python
WHISPER_MODELS = {
    "tiny": {"size": "39 MB", "speed": "⚡ Very Fast", "quality": "📊 Basic"},
    "base": {"size": "74 MB", "speed": "🚀 Fast", "quality": "📈 Good"},
    "small": {"size": "244 MB", "speed": "⏱️ Medium", "quality": "📊 Very Good"}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/whisper-transcription-app.git
cd whisper-transcription-app

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run the app in development mode
streamlit run app.py --server.runOnSave true
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Include type hints where possible

## 🐛 Issues and Support

### Common Issues

#### 1. Model Download Fails
```bash
# Clear cache and retry
rm -rf ~/.cache/whisper
```

#### 2. Out of Memory Error
- Use a smaller model (Tiny instead of Small)
- Reduce file size or split large files
- Close other applications

#### 3. Processing Timeout
- Use Tiny model for very large files
- Check internet connection
- Ensure browser tab stays active

### Getting Help

- 📖 **Documentation**: Check this README and in-app help
- 🐛 **Bug Reports**: [Open an issue](https://github.com/yourusername/whisper-transcription-app/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/whisper-transcription-app/discussions)
- 📧 **Contact**: your-email@example.com

## 📊 System Requirements

### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 2 GB free space
- **Browser**: Chrome, Firefox, Safari, Edge (latest versions)

### Recommended Specs
- **CPU**: 4+ cores, 3.0+ GHz
- **RAM**: 8+ GB
- **Storage**: 5+ GB free space
- **GPU**: CUDA-compatible (optional, for faster processing)

## 🏆 Acknowledgments

- **OpenAI** for the incredible Whisper model
- **Streamlit** for the amazing web framework
- **The open-source community** for continuous improvements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 📈 Roadmap

### Upcoming Features
- [ ] **Batch Processing**: Multiple file uploads
- [ ] **Speaker Diarization**: Identify different speakers
- [ ] **Translation**: Auto-translate to different languages
- [ ] **API Endpoint**: REST API for programmatic access
- [ ] **Cloud Storage**: Integration with Google Drive, Dropbox
- [ ] **Subtitles Export**: SRT and VTT file generation

### Version History
- **v1.0.0** - Initial release with Whisper integration
- **v1.1.0** - Added large file support (150MB)
- **v1.2.0** - Enhanced progress tracking and analytics
- **v1.3.0** - PDF export and quality metrics

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/whisper-transcription-app&type=Date)](https://star-history.com/#yourusername/whisper-transcription-app&Date)

---

**Made with ❤️ using OpenAI Whisper and Streamlit**

[🔝 Back to top](#-whisper-audio-transcription-app)