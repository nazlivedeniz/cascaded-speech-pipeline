""" Real-time voice assistant with concurrent audio capture and processing, using threading and a
queue.

    Complete cascaded/sequential pipeline in streaming mode that:
    - Accepts raw speech input (live microphone input, streamed in chunks)
    - Transcribes it into text via speech-to-text
    - Feeds transcription into a pretrained LLM for response generation
    - Converts LLM response back to audio via text-to-speech
    - Process audio in near real-time (i.e., within 500 ms of speaking).
"""
import os
import tempfile
import threading
import torch
import numpy as np
import sounddevice as sd
import queue
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.playback import play
from TTS.api import TTS
from transformers import AutoTokenizer, AutoModelForCausalLM
from cascaded_pipeline import TARGET_SAMPLING_RATE, DEVICE
from cascaded_pipeline import LLM_MODEL_ID, TTS_MODEL_ID
from silero_vad import VADIterator

# Settings
NUM_AUDIO_CHANNELS = 1  # mono audio
DTYPE = "float16" if torch.cuda.is_available() else "float32"

# Settings - Real time processing
CHUNK_DURATION_IN_SECS = 0.15
CHUNK_SIZE = int(CHUNK_DURATION_IN_SECS * TARGET_SAMPLING_RATE)  # in samples
MAX_AUDIO_QUEUE_DURATION_IN_SECS = 1.2
MAX_CHUNK_NUMBER = int(MAX_AUDIO_QUEUE_DURATION_IN_SECS / CHUNK_DURATION_IN_SECS)

# Settings - Voice Activity Detection (VAD)
VAD_MODEL_ID = "snakers4/silero-vad"
VAD_THRESHOLD = 0.5                     # Probability threshold for detecting speech
# VAD_FRAME_DURATION_IN_SECS = 0.03       # Use 10, 20 or 30 ms (recommended by Silero)
# VAD_FRAME_SIZE = int(VAD_FRAME_DURATION_IN_SECS * TARGET_SAMPLING_RATE)  # in samples
VAD_FRAME_SIZE = 512                    # For Silero, min 512 samples is required (at 16kHz)

# Settings - 'End of speech segment' detection by VAD
MAX_SILENCE_DURATION_IN_SECS = 0.3
MAX_SILENCE_CHUNKS = MAX_SILENCE_DURATION_IN_SECS / CHUNK_DURATION_IN_SECS
speech_buffer = []  # buffer for continuous speech collection

# Settings - ASR
ASR_MODEL_ID = "small"

# Settings - LLM
MAX_TURNS = 2  # keep last N full rounds (user + assistant)
dialog_history = [
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

# Audio queue
audio_queue = queue.Queue(maxsize=MAX_CHUNK_NUMBER)

# Model initialization
print(f"Loading ASR model and processor: Whisper {ASR_MODEL_ID}")
asr_model = WhisperModel(model_size_or_path="small", device=DEVICE, compute_type=DTYPE)

print(f"Loading LLM tokenizer and model: {LLM_MODEL_ID}")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)

print(f"Loading TTS model: {TTS_MODEL_ID}")
tts_model = TTS(model_name=TTS_MODEL_ID, progress_bar=True)

print(f"Loading VAD: {VAD_MODEL_ID}")
vad_model, utils = torch.hub.load(repo_or_dir=VAD_MODEL_ID,
                                  model='silero_vad',
                                  force_reload=False)
vad_iterator = VADIterator(vad_model)


def run_stream_asr(input_speech):
    """ Transcribe input speech into text using a streaming ASR model.

    :param input_speech: input speech waveform
    :return: a single string containing the full transcription, with leading/trailing
             whitespace removed and segments joined by spaces.
    """
    segments, _ = asr_model.transcribe(input_speech, beam_size=1)
    return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def run_stream_llm(input_text) -> str:
    """ Generate a response using LLM with turn-based dialog history.

    :param input_text: Transcribed input text
    :return response_text: LLM response
    """
    global dialog_history

    # Add current user input
    dialog_history.append(f"User: {input_text}")

    # Keep only the last N turns (each turn is User + Assistant)
    if len(dialog_history) > MAX_TURNS * 2:
        dialog_history = dialog_history[-MAX_TURNS * 2:]

    # Build the prompt
    prompt = "\n".join(dialog_history) + "\nAssistant:"

    # Encode
    input_ids = llm_tokenizer.encode(prompt, return_tensors='pt').to(DEVICE)

    # Generate LLM response
    output_ids = llm_model.generate(input_ids,
                                    max_new_tokens=LLM_MAX_NEW_TOKENS,
                                    pad_token_id=llm_tokenizer.eos_token_id,
                                    eos_token_id=llm_tokenizer.eos_token_id,
                                    do_sample=LLM_ENABLE_SAMPLING,
                                    temperature=LLM_TEMPERATURE,
                                    top_k=LLM_TOP_K,
                                    top_p=LLM_TOP_P,
                                    )

    # Decode the output
    full_response = llm_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Extract just the new assistant response
    if "Assistant:" in full_response:
        response_text = full_response.split("Assistant:")[-1].strip()
    else:
        response_text = full_response.strip()

    print(f"LLM Response: {response_text}")

    # Save assistant reply to history
    dialog_history.append(f"Assistant: {response_text}")

    return response_text


def run_stream_tts(input_text):
    """ Synthesize and play speech audio from the given input text using TTS model.

    :param input_text: input text to be converted to speech and played aloud.
    """

    def _play_audio():
        """ Temporarily save the synthesized data, play back, then delete temp file."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
            tts_model.tts_to_file(text=input_text, file_path=temp_wav.name)
            sound = AudioSegment.from_wav(temp_wav.name)
            play(sound)
            os.remove(temp_wav.name)

    # Play back in a background daemon thread to avoid blocking the main program
    threading.Thread(target=_play_audio, daemon=True).start()


def audio_callback(indata, frames, time, status):
    """ Audio callback which collects audio chunks as they arrive and puts them into audio_queue
    for later processing.
    """
    if status:
        print(status)
    try:
        audio_queue.put_nowait(indata.copy())
    except queue.Full:
        print("Audio queue full — dropping frame.")


def audio_loop():
    """ Open a live audio stream from microphone and feed the audio chunks into audio callback.
    Each time a new audio chunk arrives, the callback runs.
    """
    with sd.InputStream(callback=audio_callback,
                        samplerate=TARGET_SAMPLING_RATE,
                        channels=NUM_AUDIO_CHANNELS,
                        blocksize=CHUNK_SIZE):
        print("Listening...")
        while True:
            sd.sleep(1000)  # keep alive


def chunk_has_speech_silero(audio_array) -> bool:
    """ Silero operates on shorter frames. For a given audio array, check if there is speech
    frame by frame.

    :param audio_array: input audio array to search for speech in it
    :return: True if there is speech in the chunk, False otherwise.
    """
    # Silero expects torch.Tensor of shape [N]
    audio_tensor = torch.from_numpy(audio_array).float()

    # Slide over the audio chunk in Silero VAD frames
    for i in range(0, len(audio_tensor) - VAD_FRAME_SIZE + 1, VAD_FRAME_SIZE):
        frame = audio_tensor[i: i + VAD_FRAME_SIZE].unsqueeze(0)
        try:
            speech_probability = vad_model(frame, sr=TARGET_SAMPLING_RATE).item()
            if speech_probability > VAD_THRESHOLD:
                return True
        except Exception as e:
            print(f"VAD error: {e}")
            return False
    return False


def processing_loop():
    """ Processing loop which reads audio chunks from the queue, runs ASR → LLM → TTS upon voice
    activity detection (VAD).
    """
    started_speaking = False
    silence_counter = 0

    while True:
        # Retrieve latest audio chunk which is in 2D (frames, channels=1)
        audio_chunk = audio_queue.get()

        # Convert it to 1D (samples, ) and change dtype for Whisper & Silero
        audio_array = audio_chunk.flatten().astype(DTYPE)

        # Check if there is speech in the audio chunk
        is_speech = chunk_has_speech_silero(audio_array)

        if is_speech:
            if not started_speaking:
                print("Detected first voice input. Starting speech buffering...")
                started_speaking = True

            speech_buffer.append(audio_array)
            silence_counter = 0  # reset
        else:
            if not started_speaking:
                continue  # skip initial silence

            silence_counter += 1
            if silence_counter >= MAX_SILENCE_CHUNKS and speech_buffer:
                print("End of speech segment. Now processing this speech segment...")

                # Concatenate and clear buffer immediately
                full_audio = np.concatenate(speech_buffer)
                speech_buffer.clear()
                started_speaking = False
                silence_counter = 0

                try:
                    transcription = run_stream_asr(full_audio)
                    if transcription:
                        print(f"User: {transcription}")
                        try:
                            response = run_stream_llm(transcription)
                            if response:
                                print(f"Assistant: {response}")
                                try:
                                    run_stream_tts(response)
                                except Exception as e:
                                    print(f"TTS error: {e}")

                        except Exception as e:
                            print(f"LLM error: {e}")
                            continue

                except Exception as e:
                    print(f"ASR error: {e}")
                    continue  # Skip further processing


def main():
    """ Real time streaming voice assistant """

    print("Real time streaming voice assistant:")
    threading.Thread(target=audio_loop, daemon=True).start()
    processing_loop()


if __name__ == "__main__":
    main()
