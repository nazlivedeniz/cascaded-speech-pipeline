""" WEB DEMO:
    Complete cascaded/sequential pipeline that:
    - Accepts raw speech input (via mic or upload)
    - Transcribes it into text via speech-to-text
    - Feeds transcription into a pretrained LLM for response generation
    - Converts LLM response back to audio via text-to-speech
"""
import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from TTS.api import TTS

from cascaded_pipeline import (
    ASR_MODEL_ID,
    DEVICE,
    LLM_MODEL_ID,
    TORCH_DTYPE,
    TTS_MODEL_ID,
    process_pipeline,
)

# Model initialization
print(f"Loading ASR model and processor: {ASR_MODEL_ID}")
ASR_MODEL = AutoModelForSpeechSeq2Seq.from_pretrained(ASR_MODEL_ID,
                                                      torch_dtype=TORCH_DTYPE,
                                                      low_cpu_mem_usage=True,
                                                      use_safetensors=True).to(DEVICE)
ASR_PROCESSOR = AutoProcessor.from_pretrained(ASR_MODEL_ID)

print(f"Loading LLM tokenizer and model: {LLM_MODEL_ID}")
LLM_TOKENIZER = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
LLM_MODEL = AutoModelForCausalLM.from_pretrained(LLM_MODEL_ID).to(DEVICE)

print(f"Loading TTS model: {TTS_MODEL_ID}")
TTS_MODEL = TTS(model_name=TTS_MODEL_ID, progress_bar=True)


def process_pipeline_gradio(audio_file_path):
    """ Gradio-compatible wrapper to run the full ASR -> LLM -> TTS pipeline.

    :param audio_file_path: File path from gr.Audio input
    :return: Path to the generated TTS audio response
    """
    return process_pipeline(
        input_audio_file_path=audio_file_path,
        asr_model=ASR_MODEL,
        asr_processor=ASR_PROCESSOR,
        llm_model=LLM_MODEL,
        llm_tokenizer=LLM_TOKENIZER,
        tts_model=TTS_MODEL
    )


# Gradio Interface
demo = gr.Interface(
    fn=process_pipeline_gradio,
    inputs=gr.Audio(type="filepath", label="Speak or upload here"),
    outputs=[
        gr.Text(label="Transcription"),
        gr.Text(label="LLM Response"),
        gr.Audio(label="Synthesized Response"),
    ],
    title="Cascaded Pipeline Demo",
    description="Open-source pipeline using Whisper, DialoGPT, Coqui TTS."
                " - Record your voice or upload an audio file, and get a response back :)"
)

if __name__ == "__main__":
    demo.launch()
