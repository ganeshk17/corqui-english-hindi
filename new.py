# import torch
# import scipy
# import numpy as np
# import time
# from transformers import VitsModel, AutoTokenizer

# # Start timing
# start_time = time.time()

# # Load the model and tokenizer
# model = VitsModel.from_pretrained(r"C:\Users\user\Proj\hin")
# tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\user\Proj\hin")

# # Input text
# text = "यूसीआई एक तात्कालिक भुगतान प्रणाली है जिसे नेशनल पेमेंट्स कॉर्पोरेशन ऑफ इंडिया (एनपीसीआई) द्वारा विकसित किया गया है, जो एक आरबीआई"
# inputs = tokenizer(text, return_tensors="pt")

# # Generate the waveform
# with torch.no_grad():
#     output = model(**inputs).waveform

# # Convert to numpy and scale to int16 format
# audio_data = output.squeeze().cpu().numpy()  # Ensure it's a 1D array
# audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)  # Scale and convert to int16

# # Save to WAV file
# scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=audio_data)

# # End timing
# end_time = time.time()
# processing_time = end_time - start_time

# print("WAV file saved successfully.")
# print(f"Processing time: {processing_time:.2f} seconds")





import torch
import numpy as np
import time
import sounddevice as sd
from transformers import VitsModel, AutoTokenizer
import threading
import queue

# ===== Load model & tokenizer =====
model = VitsModel.from_pretrained(r"C:\Users\user\Proj\hin")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\user\Proj\hin")

# ===== Audio queue =====
audio_queue = queue.Queue()
stop_flag = False

# ===== Playback thread =====
def playback_worker():
    global stop_flag
    while not stop_flag or not audio_queue.empty():
        try:
            audio_data = audio_queue.get(timeout=0.1)
            sd.play(audio_data, samplerate=model.config.sampling_rate)
            sd.wait()
        except queue.Empty:
            continue

# ===== Generator function =====
def generate_chunks(text, chunk_size=6):
    global stop_flag
    start_time = time.time()

    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    first_chunk = True
    for idx, chunk in enumerate(chunks, start=1):
        chunk_start = time.time()

        inputs = tokenizer(chunk, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform

        audio_data = output.squeeze().cpu().numpy()
        audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)

        chunk_time = time.time() - chunk_start

        if first_chunk:
            first_chunk = False
            latency = time.time() - start_time
            print(f"⏱ Time until first audio playback: {latency:.2f} seconds")

        # print(f"🎵 Chunk {idx} generation time: {chunk_time:.2f} seconds ({len(chunk.split())} words)")

        audio_queue.put(audio_data)

    stop_flag = True  # signal playback to stop when queue is empty

# ===== Main =====
text = "यूसीआई एक तात्कालिक भुगतान प्रणाली है जिसे नेशनल पेमेंट्स कॉर्पोरेशन ऑफ इंडिया (एनपीसीआई) द्वारा विकसित किया गया है, जो एक आरबीआई"

# Start playback thread
play_thread = threading.Thread(target=playback_worker)
play_thread.start()

# Generate chunks in main thread
generate_chunks(text, chunk_size=6)

# Wait until playback is done
play_thread.join()

print("✅ Streaming playback finished.")
