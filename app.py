import streamlit as st
import webrtcvad
import pyaudio
import collections
import numpy as np
import wave
import io
from faster_whisper import WhisperModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
import tempfile
import soundfile as sf

# Load environment variables
load_dotenv()

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt', quiet=True)

# Voice Activity Detection and Speech Recognition
class VAD:
    def __init__(self, mode=3, sample_rate=16000, frame_duration_ms=30):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.ring_buffer = collections.deque(maxlen=30)
        self.triggered = False
        self.model = WhisperModel("medium", device="cpu", compute_type="float32")

    def process_audio(self, audio_data):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            sf.write(temp_file.name, audio_data, self.sample_rate)
            segments, _ = self.model.transcribe(temp_file.name, beam_size=5)
        return " ".join([segment.text for segment in segments])

# Language Model
def setup_llm():
    llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                                 google_api_key=os.getenv("GOOGLE_API_KEY"),
                                 temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Provide a concise response to the following: {question}"
    )
    return LLMChain(llm=llm, prompt=prompt)

def generate_response(chain, question, max_sentences=2):
    try:
        with get_openai_callback() as cb:
            response = chain.run(question=question)
        sentences = sent_tokenize(response)
        return ' '.join(sentences[:max_sentences])
    except Exception as e:
        st.error(f"LLM Error: {e}")
        return "I apologize, but I encountered an error while processing your request."

# Text-to-Speech using Parler TTS
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
        self.sampling_rate = 16000

    def text_to_speech(self, text, description):
        inputs = self.tokenizer(description, return_tensors="pt").to(self.device)
        prompt = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            audio = self.model.generate(
                input_ids=inputs.input_ids,
                prompt_input_ids=prompt.input_ids,
                attention_mask=inputs.attention_mask,
                prompt_attention_mask=prompt.attention_mask,
            )
        
        return audio.cpu().numpy().squeeze()

# Streamlit app
def main():
    st.title("Speech-to-Speech Pipeline")

    vad = VAD()
    llm_chain = setup_llm()
    tts = TTS()

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None

    # File upload option
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio_bytes = uploaded_file.read()
        audio_array, _ = sf.read(io.BytesIO(audio_bytes))
        st.session_state.audio_data = audio_array

    # Microphone recording option
    if st.button("Record from Microphone"):
        with st.spinner("Recording..."):
            audio_data = record_audio(duration=5)  # Record for 5 seconds
            st.session_state.audio_data = audio_data
        st.success("Recording complete!")

    if st.session_state.audio_data is not None:
        if st.button("Process"):
            # Speech to Text
            with st.spinner("Transcribing..."):
                transcription = vad.process_audio(st.session_state.audio_data)
            st.write("Transcription:", transcription)

            # LLM Response
            with st.spinner("Generating response..."):
                response = generate_response(llm_chain, transcription)
            st.write("Response:", response)

            # Text to Speech
            with st.spinner("Converting text to speech..."):
                description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
                audio = tts.text_to_speech(response, description)
            
            # Save and play audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                sf.write(tf.name, audio, tts.sampling_rate)
                st.audio(tf.name)

def record_audio(duration):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.float32))

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.concatenate(frames)

if __name__ == "__main__":
    main()