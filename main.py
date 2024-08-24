import streamlit as st
import wave
import numpy as np
from vad_s2t_demo import VAD
from llm import generate_response
from t2s import text_to_speech

st.title("Speech-to-Speech Pipeline")

vad = VAD()

@st.cache
def recognize_speech(audio_data):
    # Convert the frames to a numpy array and save as a WAV file
    temp_file = "temp_audio.wav"
    wf = wave.open(temp_file, 'wb')
    wf.setnchannels(1)  # Mono channel
    wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
    wf.setframerate(16000)
    wf.writeframes(audio_data)
    wf.close()

    # Transcribe audio using Faster Whisper
    segments, info = vad.model.transcribe(temp_file, beam_size=5)
    return info.language, info.language_probability

@st.cache
def generate_text(language, language_probability):
    question = f"What is the meaning of {language}?"
    response = generate_response(question, max_sentences=2)
    return response

@st.cache
def text_to_speech_response(text):
    return text_to_speech(text, description="A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.")

st.markdown("## Speech-to-Text")
audio_data = st.file_uploader("Upload an audio file", type=["wav"])
if audio_data is not None:
    language, language_probability = recognize_speech(audio_data.read())
    st.write(f"Detected language: {language} with probability {language_probability:.2f}")
    response = generate_text(language, language_probability)
    st.write(f"Assistant: {response}")
    audio_response = text_to_speech_response(response)
    st.audio(audio_response, format="audio/wav")