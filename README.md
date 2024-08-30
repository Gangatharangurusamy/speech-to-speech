# Speech-to-Speech Pipeline

## Overview

This project implements a Speech-to-Speech pipeline that converts spoken language into text, processes the text with a language model, and then converts the resulting text back into speech. The pipeline integrates several key components including Voice Activity Detection (VAD), speech recognition using the Whisper model, natural language processing with the Gemini Pro language model, and text-to-speech synthesis using the Parler-TTS model. The entire system is deployed as a web application using Streamlit.

## Features

- **Voice Activity Detection (VAD)**: Real-time detection of speech in audio streams.
- **Speech Recognition**: Converts spoken words into text using the Faster Whisper model.
- **Language Model Integration**: Processes the transcribed text using the Gemini Pro model to generate a response.
- **Text-to-Speech Synthesis**: Converts the response text back into speech using the Parler-TTS model.
- **Streamlit Deployment**: Provides an easy-to-use web interface for interacting with the pipeline.

## File Structure

- `app.py`: Main Streamlit app that integrates all components of the pipeline.
- `vad_implementation.py`: Implements Voice Activity Detection using the WebRTC VAD module.
- `vad_s2t_demo.py`: Demonstrates VAD integration with Whisper for speech-to-text conversion.
- `llm.py`: Handles the text processing using the Gemini Pro language model.
- `t2s.py`: Implements text-to-speech conversion using the Parler-TTS model.
- `main.py`: Modular integration of all components for deployment.
- `requirements.txt`: Python dependencies required to run the project.

## Setup Instructions

### Prerequisites

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- PyTorch
- Whisper Model
- Parler-TTS Model
- Google API key for accessing Gemini Pro

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/speech-to-speech-pipeline.git
   cd speech-to-speech-pipeline
   ```

### Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
### Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Set up environment variables:

### Create a `.env` file in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your-google-api-key
```
### Open your web browser and navigate to http://localhost:8501 to access the interface.

## Usage
### Input Speech: Speak into your microphone.
### Speech Detection: The VAD system detects your speech.
### Speech-to-Text: Your speech is transcribed into text using the Whisper model.
### Language Processing: The transcribed text is processed by the Gemini Pro model to generate a response.
### Text-to-Speech: The generated response is converted back into speech using the Parler-TTS model.
### Output Speech: The synthesized speech is played back.

## Customization
### Adjustable Parameters: The speed, pitch, and volume of the output speech can be customized in t2s.py.
### Model Choices: You can experiment with different models or settings in vad_s2t_demo.py and llm.py.

## Troubleshooting
### Ensure all dependencies are installed correctly.
### Check your microphone settings.
### Verify the API key in the .env file.
### Refer to terminal output for debugging information.

## Acknowledgments
### Streamlit
### Faster Whisper
### Parler-TTS
### Google Gemini Pro
