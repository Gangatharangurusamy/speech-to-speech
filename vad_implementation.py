import webrtcvad
import pyaudio
import collections
import sys

class VAD:
    def __init__(self, mode=3, sample_rate=16000, frame_duration_ms=30):
        self.vad = webrtcvad.Vad(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.ring_buffer = collections.deque(maxlen=30)
        self.triggered = False

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
        for frame in self.frame_generator():
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            
            if not self.triggered:
                self.ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in self.ring_buffer if speech])
                if num_voiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = True
                    print("Speech detected!")
                    for f, s in self.ring_buffer:
                        sys.stdout.write('.')
                    sys.stdout.flush()
                    self.ring_buffer.clear()
            else:
                sys.stdout.write('.' if is_speech else '_')
                sys.stdout.flush()
                self.ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in self.ring_buffer if not speech])
                if num_unvoiced > 0.9 * self.ring_buffer.maxlen:
                    self.triggered = False
                    print("\nSpeech ended.")
                    self.ring_buffer.clear()

if __name__ == "__main__":
    vad = VAD()
    try:
        vad.process_audio()
    except KeyboardInterrupt:
        print("\nStopped")