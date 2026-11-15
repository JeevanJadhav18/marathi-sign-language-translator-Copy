
from gtts import gTTS
import pygame
import os


def speak_marathi(text):
    if not text:
        return

    try:
        tts = gTTS(text=text, lang="mr")
        tts.save("marathi_speech.mp3")

        pygame.mixer.init()
        pygame.mixer.music.load("marathi_speech.mp3")
        pygame.mixer.music.play()

    except Exception as e:
        print("Speech Error:", e)
