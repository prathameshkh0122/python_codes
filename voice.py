from gtts import gTTS
text = "Hello everyone...!, i hope you guys are having a great day "
tts = gTTS(text = text, lang = "en")
tts.save("voice.mp3")
print("audio saved successfully")