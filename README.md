# whatdisay

# "What'd I Say?!"

A python utility to generate a diarized transcript from an audio file leveraging Open-AI's Whisper module for transcription and Pyannote for diarization.  

## Setup

The project depends on [OpenAI's Whisper model](https://github.com/openai/whisper/tree/main) for speech recognition.  The following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/samjhecht/whatdisay.git


## CLI Usage

The following command will take an audio file and generate a diarized transcription:

    whatdisay --from-existing audio_filename.wav

By default, it will use Whisper's `medium` model.  Currently only wav files are supported as inputs.

## Python Usage

...coming soon.

## TODOs:
- add setup utility that allows u to save ur huggingface token to .env.
- finish building the record utility
- make the transcription and diarization faster for longer files
- add tool that assists in labeling which 'SPEAKER_' maps to human names
- add support for other file types for input audio besides wav