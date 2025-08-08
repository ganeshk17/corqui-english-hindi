import torch
import time
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

# Choose your device (GPU or CPU)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the model and tokenizers
model = ParlerTTSForConditionalGeneration.from_pretrained(
    r"C:\Users\user\Proj\helpingAI\helpingai"
).to(device)
tokenizer = AutoTokenizer.from_pretrained(
    r"C:\Users\user\Proj\helpingAI\helpingai"
)
description_tokenizer = AutoTokenizer.from_pretrained(
    model.config.text_encoder._name_or_path
)

# Customize your inputs: text + description
prompt = "अरे, काय चाललंय? कसं चाललंय? आशा आहे की तू मस्त असशील!"
description = "A friendly, upbeat, and casual tone with a moderate speed. Speaker sounds confident and relaxed."

# Tokenize the inputs
input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# Start timing
start_time = time.time()

# Generate the audio
generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
audio_arr = generation.cpu().numpy().squeeze()

# Stop timing
end_time = time.time()
processing_time = end_time - start_time
print(f"Processing time: {processing_time:.2f} seconds")

# Save the audio to a file
sf.write("marathi.wav", audio_arr, model.config.sampling_rate)
print("Audio saved as output.wav")
