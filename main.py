from flask import Flask, render_template
from mic_stream import MicrophoneStream
import pyaudio
import rtmidi
import numpy as np
import transcribe


app = Flask(__name__)

MODEL_FILE = "model-180000.pt"


def get_buffer_and_transcribe(model, q):
    CHUNK = 512
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()["maxInputChannels"]
    RATE = 16000

    midiout = rtmidi.MidiOut()
    available_ports = midiout.get_ports()
    if available_ports:
        midiout.open_port(0)
    else:
        midiout.open_virtual_port("My virtual output")

    # stream = MicrophoneStream(RATE, CHUNK, CHANNELS)
    transcriber = transcribe.OnlineTranscriber(model, return_roll=False)
    with MicrophoneStream(RATE, CHUNK, CHANNELS) as stream:
        # audio_generator = stream.generator()
        print("* recording")
        on_pitch = []
        while True:
            data = stream._buff.get()
            decoded = np.frombuffer(data, dtype=np.int16) / 32768
            if CHANNELS > 1:
                decoded = decoded.reshape(-1, CHANNELS)
                decoded = np.mean(decoded, axis=1)
            frame_output = transcriber.inference(decoded)
            on_pitch += frame_output[0]
            for pitch in frame_output[0]:
                note_on = [0x90, pitch + 21, 64]
                # msg = rtmidi.MidiMessage.noteOn(0x90, pitch + 21, 64)
                midiout.send_message(note_on)
            for pitch in frame_output[1]:
                note_off = [0x90, pitch + 21, 0]
                # msg = rtmidi.MidiMessage.noteOff(0x90, pitch + 21)
                pitch_count = on_pitch.count(pitch)
                [midiout.send_message(note_off) for i in range(pitch_count)]
            on_pitch = [x for x in on_pitch if x not in frame_output[1]]
            q.put(frame_output)


@app.route("/transcribe")
def receive_and_transcribe():
    model = transcribe.load_model(MODEL_FILE)


@app.route("/")
def home():
    return render_template("index.html")


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
