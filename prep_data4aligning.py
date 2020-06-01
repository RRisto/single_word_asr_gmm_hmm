import os
from shutil import copytree
from pathlib import Path

FROM_PATH=Path('data/audio_google')
TO_PATH=Path('data/audio_google4alignment')

copytree(FROM_PATH, TO_PATH)

#create .lab files with text in audio

def create_lab_files(audio_folder):
    wav_files=os.listdir(audio_folder)

