import os
import mido
from typing import IO
import flask
from autoregressive.models import AR_Transcriber
from mic_stream import MicrophoneStream
import pyaudio
import rtmidi
import numpy as np
import transcribe
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploaded"
ALLOWED_EXTENSIONS = {"mp4", "m4a", "wav", "mp3"}
MODEL_FILE = "model-180000.pt"


def allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = flask.Flask(__name__)
app.secret_key = "super secret key"
app.config["SESSION_TYPE"] = "filesystem"

# init_app(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


CHUNK = 5120


def get_buffer_and_transcribe(model: AR_Transcriber, stream: IO[bytes]):
    CHANNELS = pyaudio.PyAudio().get_default_input_device_info()["maxInputChannels"]
    # RATE = 16000
    CHANNEL = 0x90
    # mid = mido.MidiFile()
    track = mido.MidiTrack()
    # mid.tracks.append(track)
    transcriber = transcribe.OnlineTranscriber(model, return_roll=False)
    print("* transcribing")
    on_pitch = []
    frames = []
    data = stream.read(CHUNK)
    delta_time = 0
    while data:
        decoded = np.frombuffer(data, dtype=np.int16) / 32768
        if CHANNELS > 1:
            decoded = decoded.reshape(-1, CHANNELS)
            decoded = np.mean(decoded, axis=1)
        frame_output = transcriber.inference(decoded)
        on_pitch += frame_output[0]
        for pitch in frame_output[0]:
            velocity = 64
            note_on = mido.Message(
                "note_on", note=pitch + 21, velocity=velocity, time=delta_time
            )
            track.append(note_on)
            # note_on = [CHANNEL, pitch + 21, velocity]
            # msg = rtmidi.MidiMessage.noteOn(0x90, pitch + 21, 64)
            # midiout.send_message(note_on)
        for pitch in frame_output[1]:
            velocity = 0
            # note_off = [CHANNEL, pitch + 21, velocity]
            # msg = rtmidi.MidiMessage.noteOff(0x90, pitch + 21)
            # [midiout.send_message(note_off) for i in range(pitch_count)]
            pitch_count = on_pitch.count(pitch)
            note_off = mido.Message(
                "note_off", note=pitch + 21, velocity=velocity, time=delta_time
            )
            [track.append(note_off) for i in range(pitch_count)]
        on_pitch = [x for x in on_pitch if x not in frame_output[1]]
        frames.append(frame_output)
        data = stream.read(CHUNK)
    print("* transcribed.")
    return frames, track


@app.route("/download/<name>")
def download_file(name: str):
    return flask.render_template("download.html", name=name)


@app.route("/transcribe", methods=["GET", "POST"])
def receive_and_transcribe():
    if flask.request.method == "GET":
        return flask.render_template("index.html")
    # check if the post request has the file part
    if "audio-file" not in flask.request.files:
        flask.flash("No file part")
        return flask.redirect(flask.request.url)
    file = flask.request.files["audio-file"]
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flask.flash("No selected file")
        return flask.redirect(flask.request.url)
    if file and allowed_file(file.filename):
        model = transcribe.load_model(MODEL_FILE)
        frames, track = get_buffer_and_transcribe(model, file.stream)
        mid = mido.MidiFile()
        mid.tracks.append(track)
        midi_filename = file.filename + ".midi"
        midi_file = os.path.join(app.config["UPLOAD_FOLDER"], midi_filename)
        mid.save(filename=midi_file)

        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return flask.redirect(flask.url_for("download_file", name=filename))


@app.route("/")
def home():
    return flask.render_template("index.html")


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
