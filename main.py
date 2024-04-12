from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
from transformers import MarianMTModel, MarianTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
import librosa
import numpy as np
import os

os.environ['HF_TOKEN'] = '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'
# Initialize Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")


def transcribe_audio(audio_path):
    # Load the audio file using librosa
    raw_audio, sampling_rate = librosa.load(audio_path, sr=16000)  # Whisper models expect a sampling rate of 16kHz
    # Convert audio waveform to the format expected by Whisper
    input_features = processor(raw_audio, sampling_rate=sampling_rate, return_tensors="pt").input_features
    # Generate token ids
    predicted_ids = model.generate(input_features)
    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # audio_input = audio_input
    # Perform speech recognition
    # with torch.no_grad():
    #     logits = model(audio_input).logits
    # Decode the recognized text
    # transcription = processor.batch_decode(logits.cpu().numpy())[0]
    return transcription[0]


def translate_text(text, src_lang, target_lang):
    """
    Translate text from source language to target language using MarianMT.

    Args:
    - text (str): The text to be translated.
    - src_lang (str): Source language code (e.g., 'en').
    - target_lang (str): Target language code (e.g., 'fr').

    Returns:
    - str: The translated text.
    """
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{target_lang}'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    # Tokenize the text
    tokenized_text = tokenizer.__call__([text], return_tensors='pt')

    # Translate
    translated = model.generate(**tokenized_text)

    # Decode the translation
    translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)

    return translated_text


def summarize_text(text):
    """
    Summarize the given text using BART model.

    Args:
    - text (str): The text to be summarized.

    Returns:
    - str: The summarized text.
    """
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=200, early_stopping=True)

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary


audio_path = "audio-file.mp3"
transcribed_text = transcribe_audio(audio_path)
print("Transcribed Text:", transcribed_text)

# Example of translating the transcribed text to French
translated_text = translate_text(transcribed_text, "en", "fr")
print("Translated Text:", translated_text)

# Summarizing the transcribed text
summary = summarize_text(transcribed_text)
print("Summary:", summary)
