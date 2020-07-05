from speech_recogn.record_audio import record_save_audio

#save audio file
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "data/test_data/output.wav"
record_save_audio(WAVE_OUTPUT_FILENAME, record_seconds=RECORD_SECONDS)