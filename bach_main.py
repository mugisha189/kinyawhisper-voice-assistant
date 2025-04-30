#!/usr/bin/env python3
"""


main.py

CLI: Records one audio sample, transcribes it, and gives an answer.
"""

import torch
import torchaudio
import sounddevice as sd
import pyttsx3
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# === Configuration ===
model_dir = "kinya-whisper-model"
record_duration = 5  # seconds
sample_rate = 16000
recorded_file = "recorded.wav"

# === Load model and processor ===
print("🔍 Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)
model.eval()
device = torch.device("cpu")

# === Load TTS engine ===
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# === Define QA Dictionary ===
qa_dict = {
    "amakuru yawe": "Ni meza, urakoze! nkufashe iki?",
    "witwa nde": "Nitwa Mudasa AI.",
    "wakora iki": "Nshobora kugufasha kumenya no kuvuga.",
    "uzi ikinyarwanda": "Nkunda gufasha abantu mu rurimi rwacu.",
    "igihugu cyacu nikihe": "igihugu cyacu ni u Rwanda",
    "uzi gukora iki?": "nzi ",
    "muraho ?": "muraho namwe",
    "uzi  kwandika ?": "nzi kwandika  no gusoma ikinyarwanda.",
    "wiriwe": "wiriwe neza",
    "wanshakiye amazi ":"ntabubasha mfite ngo nkuhereze amazi nshoboye kugusubiza  gusa",
    "ese uzi inzovu":"inzovu ni inyamanswa yo mu ishyamba",
    "komera":"komera nawe nagufasha iki se",
    "umworozi ni iki":"umworozi ni umuntu utunga amatungo",
    "umunsi ugizwe namasaha angahe":"umunsi ugizwe namasaha 24",
}

# === Record Audio ===
print("🎙️ Please say something in Kinyarwanda...")
recording = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()
torchaudio.save(recorded_file, torch.tensor(recording.T), sample_rate)
print("✅ Recording saved!")

# === Load and preprocess audio ===
waveform, sr = torchaudio.load(recorded_file)
if waveform.shape[0] > 1:
    waveform = torch.mean(waveform, dim=0, keepdim=True)
if sr != 16000:
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
    waveform = resampler(waveform)

# === Transcribe ===
input_features = processor.feature_extractor(
    waveform.squeeze().numpy(),
    sampling_rate=16000,
    return_tensors="pt"
).input_features

with torch.no_grad():
    predicted_ids = model.generate(
        input_features,
        max_length=64,
        num_beams=5,
        do_sample=False,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        early_stopping=True,
        task="transcribe"
    )

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
normalized = transcription.strip().lower()
print(f"📝 Transcription: {transcription}")

# === Respond ===
answer = qa_dict.get(normalized, "Mbabarira, sinasobanukiwe neza.")
print(f"🤖 Answer: {answer}")
engine.say(answer)
engine.runAndWait()