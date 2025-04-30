#!/usr/bin/env python3
"""
main.py

Performs transcription on all audio samples in a directory using a fine-tuned Whisper model.
Includes fuzzy matching for better question recognition.
"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torch
import os
import pyttsx3
from difflib import get_close_matches

model_dir = "kinya-whisper-model"
audio_dir = "audio"
output_file = "transcriptions.txt"

# Check if audio directory exists
if not os.path.exists(audio_dir):
    print(f"Error: Audio directory '{audio_dir}' not found.")
    exit(1)

print(f"🔍 Loading model from {model_dir}...")

# Load model and processor
model = WhisperForConditionalGeneration.from_pretrained(model_dir)
processor = WhisperProcessor.from_pretrained(model_dir)

model.eval()
device = torch.device("cpu")

# Prepare the model's generation config (do this once)
model.generation_config.forced_decoder_ids = None

print(f"✅ Model loaded successfully!")
print(f"📂 Processing all audio files in {audio_dir}...")

# Initialize TTS engine once
engine = pyttsx3.init()
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

# Q&A dictionary
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


def get_best_match(transcription, qa_dict, cutoff=0.6):
    """
    Find the best matching question in the dictionary using fuzzy matching.
    Returns the matched question or None if no good match is found.
    """
    normalized = transcription.strip().lower()

    # Try exact match first
    if normalized in qa_dict:
        return normalized

    # Try fuzzy matching
    matches = get_close_matches(normalized, qa_dict.keys(), n=1, cutoff=cutoff)
    if matches:
        match = matches[0]
        print(f"🔍 Fuzzy matched '{normalized}' to '{match}'")
        return match

    # Try partial matching for short queries
    if len(normalized) >= 3:
        for question in qa_dict.keys():
            if normalized in question:
                print(f"🔍 Partial matched '{normalized}' to '{question}'")
                return question

    return None


# Open output file for writing
with open(output_file, "w") as f_out:
    # Get all WAV files in the directory
    audio_files = [f for f in sorted(os.listdir(audio_dir)) if f.lower().endswith(".wav")]

    if not audio_files:
        print(f"⚠️ No WAV files found in {audio_dir}")
        exit(0)

    print(f"🎵 Found {len(audio_files)} audio files to process")

    # Process each audio file
    for idx, filename in enumerate(audio_files):
        filepath = os.path.join(audio_dir, filename)
        print(f"[{idx + 1}/{len(audio_files)}] Transcribing {filename}...")

        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(filepath)

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Process audio
            input_features = processor.feature_extractor(
                waveform.squeeze().numpy(),
                sampling_rate=sample_rate,
                return_tensors="pt"
            ).input_features

            # Generate transcription
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

            # Decode the output
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

            # Find the best match in the dictionary using fuzzy matching
            best_match = get_best_match(transcription, qa_dict)

            if best_match:
                answer = qa_dict[best_match]
            else:
                answer = "Sinasobanukiwe neza."  # Default response

            print(f"🎤 Transcribed: {transcription}")
            print(f"🤖 Answer: {answer}")

            # Speak the answer using pyttsx3
            print("🗣️ Speaking the answer using pyttsx3...")
            engine.say(answer)
            engine.runAndWait()

            # Save result
            f_out.write(f"{filename}: {transcription} -> {answer}\n")

        except Exception as e:
            error_msg = f"Error processing {filename}: {str(e)}"
            print(f"❌ {error_msg}")
            f_out.write(f"{filename}: ERROR - {str(e)}\n")

    print(f"✅ Processing complete! Results saved to {output_file}")