import webrtcvad
import pyaudio
import collections
import sys
import numpy as np
import wave
import io
from faster_whisper import WhisperModel

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

if __name__ == "__main__":
    vad = VAD()
    try:
        vad.process_audio()
    except KeyboardInterrupt:
        print("\nStopped")
