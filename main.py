import os
os.environ['GOOGLE_API_KEY'] = 'AIzaSyA9GfsX8G2ivedwGKKhthq9fD619p1ZO9o'
import streamlit as st
import webrtcvad
import numpy as np
from faster_whisper import WhisperModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import tempfile
import soundfile as sf
import pyaudio
import wave
import librosa

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
        self.model = WhisperModel("small", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float32")

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

# Text-to-Speech
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
        
        audio = audio.cpu().numpy().squeeze()
        # Increase the speed of the audio
        return librosa.effects.time_stretch(audio, rate=1.2)

def record_audio(duration=5, sample_rate=16000):
    try:
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

        st.info(f"Recording for {duration} seconds...")
        frames = []
        for _ in range(0, int(sample_rate / 1024 * duration)):
            data = stream.read(1024)
            frames.append(np.frombuffer(data, dtype=np.float32))

        st.info("Recording finished.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        return np.concatenate(frames)
    except OSError as e:
        st.error(f"Error accessing audio device: {e}")
        st.warning("Microphone input is not available. Please use file upload instead.")
        return None

# Streamlit app
def main():
    st.title("Speech-to-Speech Pipeline")

    vad = VAD()
    llm_chain = setup_llm()
    tts = TTS()

    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None

    input_method = st.radio("Choose input method:", ("Microphone", "File Upload"))

    if input_method == "Microphone":
        if st.button("Record Audio"):
            audio_data = record_audio()
            if audio_data is not None:
                st.session_state.audio_data = audio_data
                st.success("Audio recorded successfully!")
            else:
                st.error("Failed to record audio. Please try file upload instead.")
    else:
        uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])
        if uploaded_file is not None:
            st.session_state.audio_data, _ = sf.read(uploaded_file)
            st.success("File uploaded successfully!")

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
                description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
                audio = tts.text_to_speech(response, description)
            
            # Save and play audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
                sf.write(tf.name, audio, tts.sampling_rate)
                st.audio(tf.name)

if __name__ == "__main__":
    main()