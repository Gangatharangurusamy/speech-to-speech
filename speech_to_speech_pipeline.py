import webrtcvad
import pyaudio
import collections
import sys
import numpy as np
import wave
import io
from threading import Thread
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
import soundfile as sf
import sounddevice as sd


# Load environment variables
load_dotenv()

# Initialize VAD

class VAD:
    def __init__(self, mode=3, sample_rate=16000, frame_duration_ms=30):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.ring_buffer = collections.deque(maxlen=30)
        self.triggered = False

        # Load Faster Whisper model
        model_size = "small"  # or use "base", "large-v3", etc.
        self.model = WhisperModel(model_size, device="cpu", compute_type="float32")  # or adjust device and compute_type

    def frame_generator(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=self.sample_rate,
                        input=True,
                        frames_per_buffer=self.frame_size)

        while True:
            frame = stream.read(self.frame_size)
            yield frame

    def process_audio(self):
        frames = []
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = True
                    print("Speech detected!")
                    frames.extend([f for f, s in self.ring_buffer])
                    self.ring_buffer.clear()
            else:
                sys.stdout.write('.' if is_speech else '_')
                sys.stdout.flush()
                frames.append(frame)
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = False
                    print("\nSpeech ended.")
                    self.ring_buffer.clear()
                    self.recognize_speech(frames)
                    frames = []

    def recognize_speech(self, frames):
        # Convert the frames to a numpy array and save as a WAV file
        audio_data = b''.join(frames)
        temp_file = "temp_audio.wav"
        
        # Save audio to a WAV file with correct format
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data)

        # Transcribe audio using Faster Whisper
        segments, info = self.model.transcribe(temp_file, beam_size=5)
        
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in segments:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))





# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                             google_api_key=os.getenv("GOOGLE_API_KEY"),
                             temperature=0.7)

prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Provide a concise response to the following: {question}"
)

chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(question, max_sentences=2):
    try:
        # Generate the response
        with get_openai_callback() as cb:
            response = chain.run(question=question)

        # Split the response into sentences
        sentences = sent_tokenize(response)

        # Limit to max_sentences
        limited_response = ' '.join(sentences[:max_sentences])

        return limited_response
    except Exception as e:
        print(f"An error occurred: {e}")
        return "I apologize, but I encountered an error while processing your request."

# Initialize Text-to-Speech

device = "cpu"
model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
sampling_rate = 16000

def text_to_speech(text, description, speed=1.0, pitch=1.0, volume=1.5, play_steps_in_s=0.5):
    print("Starting text_to_speech function...")
    play_steps = int(sampling_rate * play_steps_in_s)
    streamer = ParlerTTSStreamer(model, device=device, play_steps=play_steps)

    # Tokenization
    inputs = tokenizer(description, return_tensors="pt").to(device)
    prompt = tokenizer(text, return_tensors="pt").to(device)

    # Create generation kwargs
    generation_kwargs = dict(
        input_ids=inputs.input_ids,
        prompt_input_ids=prompt.input_ids,
        attention_mask=inputs.attention_mask,
        prompt_attention_mask=prompt.attention_mask,
        streamer=streamer,
        do_sample=True,
        temperature=1.0,
        min_new_tokens=10,
    )

    # Initialize Thread
    print("Starting generation thread...")
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    audio_chunks = []
    
    # Iterate over chunks of audio
    print("Waiting for audio chunks...")
    for new_audio in streamer:
        if new_audio.shape[0] == 0:
            break
        
        # Apply adjustable parameters
        new_audio = new_audio * volume  # Amplify volume
        if speed != 1.0:  # Only adjust speed if necessary
            new_audio = np.interp(np.arange(0, len(new_audio), speed), np.arange(0, len(new_audio)), new_audio)
        
        audio_chunks.append(new_audio)
        
        print("Received audio chunk, length:", len(new_audio))
        
        # Uncomment these lines if you want to play audio in real-time
        sd.play(new_audio, sampling_rate)
        sd.wait()

    print("Finished receiving audio chunks.")
    return np.concatenate(audio_chunks)
# Speech-to-Speech Pipeline
class SpeechToSpeechPipeline:
    def __init__(self):
        self.vad = VAD()
        self.description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

    def process_audio_frame(self, audio_frame):
        # Convert the audio frame to the format expected by your VAD
        frame = (audio_frame * 32767).astype(np.int16).tobytes()
        
        is_speech = self.vad.vad.is_speech(frame, self.vad.sample_rate)
        
        if not self.vad.triggered:
            self.vad.ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in self.vad.ring_buffer if speech])
            if num_voiced > 0.9 * self.vad.ring_buffer.maxlen:
                self.vad.triggered = True
                print("Speech detected!")
                frames = [f for f, s in self.vad.ring_buffer]
                self.vad.ring_buffer.clear()
                self.process_speech(frames)
        else:
            frames = [frame]
            self.vad.ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in self.vad.ring_buffer if not speech])
            if num_unvoiced > 0.9 * self.vad.ring_buffer.maxlen:
                self.vad.triggered = False
                print("\nSpeech ended.")
                self.vad.ring_buffer.clear()
                self.process_speech(frames)

    def process_speech(self, frames):
        # Convert speech to text
        transcribed_text, info = self.vad.recognize_speech(frames)
        print("Transcribed:", transcribed_text)
        print(f"Detected language: {info.language} with probability {info.language_probability}")

        # Generate response using LLM
        response = generate_response(transcribed_text)
        print("LLM Response:", response)

        # Convert response to speech
        audio = text_to_speech(response, self.description)
        sd.play(audio, sampling_rate)
        sd.wait()

        # Update Streamlit session state
        st.session_state.messages.append({"role": "user", "content": transcribed_text})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.experimental_rerun()

if __name__ == "__main__":
    pipeline = SpeechToSpeechPipeline()
    try:
        pipeline.process_audio()
    except KeyboardInterrupt:
        print("\nStopped")