import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
from threading import Thread
import soundfile as sf
import sounddevice as sd
import numpy as np
import sys

# Set up the model and tokenizer
device = "cpu"
print("Loading model...")
try:
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
except Exception as e:
    print(f"Failed to load tokenizer: {e}")
    sys.exit(1)

sampling_rate = 16000  # Confirm this is the correct sampling rate for your model

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

# Example usage
if __name__ == "__main__":
    text = "This is a test of the Parler-TTS model running on CPU."
    description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

    # Adjustable parameters
    speed = 1.5  # 1.0 is normal speed, <1 is slower, >1 is faster
    pitch = 1.0  # 1.0 is normal pitch, adjust as needed
    volume = 1.5  # Increased volume to improve audibility

    print("Generating audio... This may take a while on CPU.")
    try:
        audio = text_to_speech(text, description, speed, pitch, volume)
        print("Audio generation complete. Total length:", len(audio) / sampling_rate, "seconds")

        # Optionally, save the audio to a file
        sf.write("output.wav", audio, sampling_rate)
        print("Audio saved to output.wav")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Error details:", sys.exc_info())
        print("Please check your installation and model compatibility.")

print("Script completed.")
