from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav="C:/Users/user/Proj/corque/techno.wav",
                language="en")




tts.tts_to_file(text="मुझे अपनी आवाज विकसित करने में काफी समय लगा, और अब जब यह मेरे पास है तो मैं चुप नहीं रहूंगी।",
                file_path="output.wav",
                speaker_wav="C:/Users/user/Proj/corque/techno.wav",
                language="hi")