from speech_recognition.record_audio import record_save_audio

RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "data/test_data/output.wav"
record_save_audio(WAVE_OUTPUT_FILENAME, record_seconds=RECORD_SECONDS)