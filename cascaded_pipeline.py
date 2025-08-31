""" Complete cascaded/sequential pipeline that:
    - Accepts raw speech input (via file upload)
    - Transcribes it into text via speech-to-text
    - Feeds transcription into a pretrained LLM for response generation
    - Converts LLM response back to audio via text-to-speech
"""
import argparse
import time
from typing import Tuple

import torch
import torchaudio
from soundfile import LibsndfileError
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
)
from TTS.api import TTS

# Run on GPU if available, fallback to CPU
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Input and output file paths
INPUT_AUDIO_FILE_PATH = "example_files/Will_There_Be_Sunshine_Tomorrow.wav"
OUTPUT_AUDIO_FILE_PATH = "example_files/LLM_response.wav"

# Models
ASR_MODEL_ID = "openai/whisper-small"
LLM_MODEL_ID = "microsoft/DialoGPT-medium"
TTS_MODEL_ID = "tts_models/en/ljspeech/tacotron2-DDC"

# Model parameters
TARGET_SAMPLING_RATE = 16000  # required for Whisper STT

# LLM settings
LLM_DIALOG_HISTORY = [
    "User: Hello, who are you?",
    "Assistant: I am a voice assistant.",
    "User: I have a question.",
    "Assistant: How can I help you?"
]
LLM_MAX_NEW_TOKENS = 100
LLM_ENABLE_SAMPLING = True  # required for top_k, top_p, temp, otherwise greedy
LLM_TEMPERATURE = 0.9       # lower = more conservative; higher = more creative, between 0 and 1
LLM_TOP_K = 50              # samples from top_k most likely tokens
LLM_TOP_P = 0.90            # samples from top_p% cumulative probability mass, between 0 and 1


def load_input_speech_audio(input_audio_file_path: str) -> Tuple[torch.Tensor, int]:
    """ Loads input speech from a given audio file.

    :param input_audio_file_path: Path to the input audio file

    :return: Tuple (audio_waveform, sampling_rate)
    :raises LibsndfileError: if the file cannot be loaded
    """
    try:
        audio_waveform, sampling_rate = torchaudio.load(input_audio_file_path)
        print(f"Audio loaded: {input_audio_file_path}")
        print(f"Sample rate: {sampling_rate}, Shape: {audio_waveform.shape}")
        return audio_waveform, sampling_rate
    except LibsndfileError as exception:
        print(f"File not found: {input_audio_file_path}")
        raise exception


def load_and_preprocess_audio_data(input_audio_file_path: str, target_sampling_rate: int = 16000,
                                   convert_to_mono: bool = True) -> torch.Tensor:
    """ Loads input speech from a given audio file and processes the waveform.

    :param input_audio_file_path: Path to the input audio file
    :param target_sampling_rate: Target sampling rate, default is 16kHz.
    :param convert_to_mono: Flag to convert the audio to mono, set to True by default.

    :return audio_waveform: Single channel audio waveform
    """
    # Load input speech from the file
    waveform, sampling_rate = load_input_speech_audio(input_audio_file_path)

    # Convert to mono if needed
    if convert_to_mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sampling_rate != target_sampling_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate,
                                                   new_freq=target_sampling_rate)
        waveform = resampler(waveform)

    # Remove channel dimension (squeeze to 1D)
    waveform = waveform.squeeze()

    print("Speech waveform loaded and prepared for the pipeline.")
    return waveform


def asr_step(asr_model, asr_processor, input_speech: torch.Tensor) -> str:
    """ Transcribes the input speech into text using the given Automatic
    Speech Recognition (ASR) model.

    :param asr_model: ASR model
    :param asr_processor: ASR processor
    :param input_speech: Input speech as a 1D tensor

    :return output_text: Transcribed text
    """
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=asr_model,
        tokenizer=asr_processor.tokenizer,
        generate_kwargs={"task": "transcribe", "language": "en"},  # avoids Whisper warnings
        feature_extractor=asr_processor.feature_extractor,
        torch_dtype=TORCH_DTYPE,
        device=DEVICE,
    )
    # Prepare the ASR pipeline

    result = asr_pipe(input_speech.numpy())
    transcription = result['text']
    print(f"Transcription: {transcription}")

    return transcription


def llm_step(llm_model, llm_tokenizer, input_text: str, dialog_history: list = None) -> str:
    """ Run Large Language Model (LLM) to generate response to the input text using prior dialog
    context.

    :param llm_model: LLM model
    :param llm_tokenizer: LLM tokenizer
    :param input_text: Latest transcribed text input
    :param dialog_history: Optional list of dialog turns like ["User: ...", "Assistant: ..."]

    :return response_text: LLM generated text response
    """
    if dialog_history is None:
        dialog_history = []

    # Append the new user prompt and encode it
    full_prompt = "\n".join(dialog_history + [f"User: {input_text}", "Assistant:"])
    input_ids = llm_tokenizer.encode(full_prompt, return_tensors='pt').to(DEVICE)

    # Create attention mask
    attention_mask = torch.ones_like(input_ids).to(DEVICE)

    # Generate response
    output_ids = llm_model.generate(input_ids,
                                    attention_mask=attention_mask,
                                    max_new_tokens=LLM_MAX_NEW_TOKENS,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    eos_token_id=llm_tokenizer.eos_token_id,
                                    do_sample=LLM_ENABLE_SAMPLING,
                                    temperature=LLM_TEMPERATURE,
                                    top_k=LLM_TOP_K,
                                    top_p=LLM_TOP_P
                                    )

    # Decode only new tokens
    response_text = llm_tokenizer.decode(
        output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Remove everything after "User:" if it exists
    response_text = response_text.split("User:")[0].strip()
    print(f"LLM Response: {response_text}")

    return response_text


def tts_step(tts_model, input_text: str, output_audio_file_path: str):
    """ Synthesize speech for the given input text using the specified Text-to-Speech (TTS) model
    and save the output audio into a wav file.

    :param tts_model: TTS model
    :param input_text: Input text to synthesize speech for
    :param output_audio_file_path: Path to the output audio file
    """
    # Convert text to speech
    try:
        tts_model.tts_to_file(text=input_text, file_path=output_audio_file_path)
        print(f"Synthesized speech saved to {output_audio_file_path}")
    except Exception as e:
        print(f"Error during TTS conversion: {e}")


def process_pipeline(input_audio_file_path, asr_model, asr_processor,
                     llm_model, llm_tokenizer, tts_model):
    """ For a given input audio file path,
    - load and prepare the input speech waveform,
    - run the cascaded pipeline ASR -> LLM -> TTS,
    - save the synthesized speech into a file.

    :param input_audio_file_path: input audio file path
    :param asr_model: ASR model
    :param asr_processor: ASR processor
    :param llm_model: LLM model
    :param llm_tokenizer: LLM tokenizer
    :param tts_model: TTS model

    :return transcribed_text, llm_response, output_audio_file_path:
    transcribed text, llm response and the output file path.
    """
    # Convert file into waveform
    input_speech = load_and_preprocess_audio_data(input_audio_file_path=input_audio_file_path,
                                                  target_sampling_rate=TARGET_SAMPLING_RATE,
                                                  convert_to_mono=True)

    # ASR step
    start_time_asr = time.time()
    transcribed_text = asr_step(asr_model=asr_model, asr_processor=asr_processor,
                                input_speech=input_speech)

    # LLM step
    start_time_llm = time.time()
    llm_response = llm_step(llm_model=llm_model, llm_tokenizer=llm_tokenizer,
                            input_text=transcribed_text, dialog_history=LLM_DIALOG_HISTORY)

    # TTS step
    start_time_tts = time.time()
    tts_step(tts_model=tts_model, input_text=llm_response,
             output_audio_file_path=OUTPUT_AUDIO_FILE_PATH)
    end_time_tts = time.time()

    print(f"ASR time: {(start_time_llm - start_time_asr):.2f} seconds")
    print(f"LLM time: {(start_time_tts - start_time_llm):.2f} seconds")
    print(f"TTS time: {(end_time_tts - start_time_tts):.2f} seconds")
    print(f"Elapsed time after audio loading: {(end_time_tts - start_time_asr):.2f} seconds")

    return transcribed_text, llm_response, OUTPUT_AUDIO_FILE_PATH


def main():
    """ Cascaded/sequential pipeline """

    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Cascaded speech pipeline:")
    parser.add_argument(
        "--input_audio", type=str, default=INPUT_AUDIO_FILE_PATH,
        help="Path to input WAV audio file "
             "(default: example_files/Will_There_Be_Sunshine_Tomorrow.wav)"
    )
    args = parser.parse_args()

    input_path = args.input_audio

    # Model initialization
    print(f"Loading ASR model and processor: {ASR_MODEL_ID}")
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_ID,
                                                          torch_dtype=TORCH_DTYPE,
                                                          low_cpu_mem_usage=True,
                                                          use_safetensors=True).to(DEVICE)
    asr_processor = AutoProcessor.from_pretrained(ASR_MODEL_ID)

    print(f"Loading LLM tokenizer and model: {LLM_MODEL_ID}")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
    llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)

    print(f"Loading TTS model: {TTS_MODEL_ID}")
    tts_model = TTS(model_name=TTS_MODEL_ID, progress_bar=True)

    print("Running cascaded pipeline...")
    start_time = time.time()
    process_pipeline(input_audio_file_path=input_path,
                     asr_model=asr_model,
                     asr_processor=asr_processor,
                     llm_model=llm_model,
                     llm_tokenizer=llm_tokenizer,
                     tts_model=tts_model)
    elapsed_time = time.time() - start_time
    print(f"Cascaded pipeline finished successfully in {elapsed_time} seconds!")


if __name__ == "__main__":
    main()
