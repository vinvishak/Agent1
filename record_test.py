import sounddevice as sd
from scipy.io.wavfile import write

SAMPLE_RATE = 16000  # 16kHz, good for speech
DURATION = 5         # seconds

def record_test():
    print("Recording for 5 seconds... Speak now.")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16')
    sd.wait()  # Wait until recording is finished
    write("test.wav", SAMPLE_RATE, audio)
    print("Saved to test.wav")

if __name__ == "__main__":
    record_test()
