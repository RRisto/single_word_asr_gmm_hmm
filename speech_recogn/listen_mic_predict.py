import contextlib
import io
import time
import wave

import speech_recognition as sr
from speech_recogn.HMMSpeechRecog import HMMSpeechRecog


def write_wave(audio, sample_rate):
    """Writes a .wav binary file.
    Takes path, PCM audio data, and sample rate.
    """

    temp_file = io.BytesIO()
    with contextlib.closing(wave.open(temp_file, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

    temp_file.seek(0)
    return temp_file


def _listen_predict(predict_method, sample_rate, listen_duration_sec=20, phrase_time_limit=2):
    # this class doesnt know anything about the model, just gets reference to predict_method that takes
    # binary .wav file as input
    r = sr.Recognizer()

    with sr.Microphone(sample_rate=sample_rate) as source:
        time_end = time.time() + listen_duration_sec
        run = True
        while run:
            try:
                print('Say something')
                audio = r.listen(source, phrase_time_limit=phrase_time_limit)  # listen to source
                audio_file = write_wave(audio.frame_data, audio.sample_rate)
                print(f'model predicted: {predict_method([audio_file])}')
                if time.time() > time_end:
                    run = False
            except Exception as e:
                print(f'Error {e}')
                run = False


def listen_predict_mic(model_path, listen_duration_sec=20, phrase_time_limit=1.5):
    model = HMMSpeechRecog.unpickle(model_path)
    _listen_predict(model.predict_files, model.sample_rate, listen_duration_sec, phrase_time_limit)
