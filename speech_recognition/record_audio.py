import pyaudio
import wave


def write_wave(filename, channels, rate, sample_size, frames):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(sample_size)
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def record_save_audio(wave_output_filename, chunk=1024, format=pyaudio.paInt16, channels=1, rate=16000,
                      record_seconds=2):
    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)
    print("* recording")
    frames = []

    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    write_wave(wave_output_filename, channels, rate, p.get_sample_size(format), frames)
