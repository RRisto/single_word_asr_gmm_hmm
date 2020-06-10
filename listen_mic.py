import speech_recognition as sr
r = sr.Recognizer()

with sr.Microphone() as source:
    run=True
    while run:
        try:
            audio = r.listen(source, phrase_time_limit=2)  # listen to source
            audio
        except Exception as e:
            print(f'Error {e}')
            run=False