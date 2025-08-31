## Design Decisions

## Cascaded Pipeline Overview

### Advantages

- It is easy to implement.
- It is straightforward to interpret, since it is easier to debug.
- It is highly modular. Different pretrained components can be used for each step depending on
  the need, without retraining.
- No need to train the entire system end-to-end.
- High performance with a good quality ASR block.

### Challenges

- Performance highly depends on the quality of the initial ASR block. Errors in the 
  transcription propagate through the other blocks.
- Noise has a big impact on the transcription quality.
- Relatively high latency since blocks operate sequentially and LLM waits for the transcription.
- LLM block has no access to acoustic features, hence unable to process non-textual information
  (such as audio nuances) -> No multi-modal fusion!

## A - Model Selection for Cascaded Pipeline Demo

### A1- Speech-to-Text (Automatic Speech Recognition) Model

OpenAI's `whisper-small` is selected as the STT model, since

- It is open-source.
- It is available via HuggingFace.
- It has high quality even with small models.
- It is easy to use (no big audio preprocessing needed).
- It is robust to noise.

Further notes:

- Since `Whisper` requires **mono** audio **at 16kHz**, a data preprocessing step was added to
  resample the audio and convert it to mono if needed.
- `Whisper` is trained on 30-second audio segments. On very short segments (under ~1–2s), it may 
  produce incomplete or low-confidence transcriptions, or it may hallucinate. But it is mostly 
  capable of handling 5-10 seconds of audio without much latency and performance degradation  
  introduced.

### A2- Large Language Model

Based on a quick research on conversational language models, `microsoft/DialoGPT-medium` is 
selected, since

- It is available via HuggingFace.
- It is easy to use (no special preprocessing or fine-tuning needed for basic dialog pipelines).
- It is fast and lightweight (`GPT2` based), hence good for demo purposes.
- It is expected to give a natural conversational flow due to being trained
  specifically on multi-turn conversations (from Reddit).
- It is expected to generate more natural and human-like responses compared to general purpose
  LLMs trained on mixed sources.
- It is possible to get more context-aware responses when a dialog history (in the form of
  alternating User/Assistant turns) is provided.

`medium` version of `DialoGPT` is selected for this project as a balanced option between
conversational quality and efficiency, so that there will be some room to scale down to `small` for
faster inference or scale up to `large` for improved contextual awareness and more coherent
responses.

Although this model

- might fail with up-to-date facts,
- might produce unexpectedly casual outputs,
- cannot follow instructions, and
- can only handle a short history,

it can still be considered a good starter model.

Further notes on LLM parameters:

- Temperature parameter controls the creativity and randomness, while top_p and top_k control 
  how wide the model's options can be.
- For this demo, a more balanced mode for the LLM is selected. See `LLM_MAX_NEW_TOKENS`, 
  `LLM_TEMPERATURE`, `LLM_TOP_P`, `LLM_TOP_K` params set in the script.
- Known issues: From time to time it can take a while to generate the response. It usually ends up 
  with a short answer like 'Yes/No" or an emoji or a punctuation like "...", and the 
  resulting synthesized speech might sound weird.  
- DialoGPT has a maximum context window of 1024 tokens. Since dialog history is passed as text  
  (User: ... Assistant: ...), it can fill up quickly. It can be further optimized or a turn-based 
  approach can be implemented which only keeps the last N rounds.

### A3- Text-to-Speech Model

CoquiTTS' `tts_models/en/ljspeech/tacotron2-DDC` is selected as the TTS model, since

- It is open-source.
- It has good audio quality thanks to **tacotron2-DDC** and **hifigan_v2** combination.
- It is easy to install and use.
- It is fast.
- It has light memory footprint (can run on GPU and CPU).
- It is good for real-time demos.

Although this model

- tends to mispronounce contractions (e.g. you are -> you're) or acronyms,
- struggles to handle punctuation, and
- only supports English with a single pre-trained female voice,

it still serves quite well as a starter model.

Note that adding a simple utility script that normalizes contractions, removes symbols, emojis,
etc., prior to synthesis will help with the output audio quality.

### A4- Demo Tool

`Gradio UI` is used for demo, since

- It is easy to use with very few lines of code required.
- It provides a clean demo interface.

## B - Model Selection for Cascaded Pipeline Streaming Demo

### B1- Speech-to-Text (Automatic Speech Recognition) Model

For the original demo, `whisper-small` from HuggingFace was selected. Since it cannot be used 
for the streaming version of the pipeline, `faster-whisper` library is used.

### B2- Voice Activity Detection Model

For the streaming pipeline version, a VAD block is added in front of the pipeline in order to
detect the speech and end of speech segments. Upon detection of end of speech by VAD, the speech
segment is sent to the cascaded ASR -> LLM -> TTS pipeline.

`WebRTC` is commonly used for VAD problems, but its performance is already outperformed by
`SileroVAD`. `SileroVAD` performs significantly better than `WebRTC` in terms of accurately  
rejecting silence and non-speech (i.e., improving true negatives).

Further notes:

- VAD models operate on shorter frames than our streaming audio chunks. For example, `SileroVAD` 
  recommends using frame sizes of 10, 20, or 30 ms (160, 320, 480 samples respectively at 16kHz).
  Hence for each audio chunk, sliding windowing is applied to check if there is speech in each frame.
