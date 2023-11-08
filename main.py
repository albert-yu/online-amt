import os
import flask
from mic_stream import MicrophoneStream
import pyaudio
import rtmidi
import numpy as np
import transcribe
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "/path/to/the/uploads"
ALLOWED_EXTENSIONS = {"mp4", "m4a", "wav", "mp3"}
MODEL_FILE = "model-180000.pt"


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = flask.Flask(__name__)
app.secret_key = "super secret key"
app.config["SESSION_TYPE"] = "filesystem"

# init_app(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


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


@app.route("/download")
def download_file():
    return flask.render_template("download.html")


@app.route("/transcribe", methods=["GET", "POST"])
def receive_and_transcribe():
    if flask.request.method == "GET":
        return flask.render_template("index.html")
    model = transcribe.load_model(MODEL_FILE)
    # check if the post request has the file part
    if "file" not in flask.request.files:
        flask.flash("No file part")
        return flask.redirect(flask.request.url)
    file = flask.request.files["file"]
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == "":
        flask.flash("No selected file")
        return flask.redirect(flask.request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        return flask.redirect(flask.url_for("download", name=filename))


@app.route("/")
def home():
    return flask.render_template("index.html")


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
