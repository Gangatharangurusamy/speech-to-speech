import streamlit as st
import os
import torch
import soundfile as sf

# Import functions from the respective scripts
from vad_s2t_demo import VAD
from llm import generate_response
from t2s import text_to_speech

# Streamlit UI Setup
st.title("Speech-to-Speech Assistant")
st.write("Speak into your microphone, and the system will process your query and respond with synthesized speech.")

# Initialize VAD
vad = VAD()

# Start the listening and processing pipeline
if st.button("Start Listening"):
    st.write("Listening for your query...")

    # Process the audio in smaller chunks
    try:
        user_query = vad.process_audio()  # Streamlined VAD to handle memory efficiently
        st.write(f"Transcribed Query: {user_query}")

        if user_query:
            response = generate_response(user_query)
            st.write(f"Assistant: {response}")

            # Adjustable parameters
            speed = st.slider("Speed", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            pitch = st.slider("Pitch", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
            volume = st.slider("Volume", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

            description = "A female speaker delivers a slightly expressive and animated speech."
            with torch.no_grad():  # Disable gradient calculations to save memory
                audio = text_to_speech(response, description, speed=speed, pitch=pitch, volume=volume)
            
            # Play the audio in the Streamlit app
            st.audio(audio, format='audio/wav')
            sf.write("output.wav", audio, 16000)
            st.write("Audio saved to output.wav")
    except OSError as e:
        st.error(f"An error occurred during audio processing: {e}")
    except Exception as ex:
        st.error(f"An unexpected error occurred: {ex}")

