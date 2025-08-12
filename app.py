
import torch
import scipy
import numpy as np
from transformers import VitsModel, AutoTokenizer

# Load the model and tokenizer
model = VitsModel.from_pretrained(r"C:\Users\user\Proj\fem-hin")
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\user\Proj\fem-hin")

# Input text
text = "यूसीआई एक तात्कालिक भुगतान प्रणाली है जिसे नेशनल पेमेंट्स कॉर्पोरेशन ऑफ इंडिया (एनपीसीआई) द्वारा विकसित किया गया है, जो एक आरबीआई"
inputs = tokenizer(text, return_tensors="pt")

# Generate the waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Convert to numpy and scale to int16 format
audio_data = output.squeeze().cpu().numpy()  # Ensure it's a 1D array
audio_data = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)  # Scale and convert to int16

# Save to WAV file
scipy.io.wavfile.write("techno.wav", rate=model.config.sampling_rate, data=audio_data)

print("WAV file saved successfully.")

