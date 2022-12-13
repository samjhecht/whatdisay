# whatdisay

# "What'd I Say?!"

A python utility to generate a diarized transcript from an audio file leveraging [Open-AI's Whisper](https://github.com/openai/whisper/tree/main) module for transcription and [Deepgram](https://deepgram.com/) for diarization.  

The project leverages the following libraries/APIs:
* [OpenAI's Whisper model](https://github.com/openai/whisper/tree/main): Used for speech recognition.  The following command will pull and install the latest commit from this repository, along with its Python dependencies:
* [Deepgram](https://deepgram.com/): The default solution for speaker diarization.  You'll need to create an account and get an API key if you want to leverage the speaker diarization capabilities of this library.  Deepgram also provides transcription functionality, but it's not as good as Whisper, so this library just leverages Deepgram's diarization function and then uses Whisper to generating the transcriptions.  (Deepgram does have a 'beta' version of functionality to allow you to set your model to "whisper" for transcription, but at this time it does not support diarization)
* [Pyannote](https://github.com/pyannote/pyannote-audio): While Deepgram is the default, the library also supports using Pyannote instead for speaker diarization.  This option is best if you would like to leverage [Pyannote's solution](https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/prodigy.md) for annotating your own dataset to improve accuracy of speaker diarization.

## Setup

The following command will pull and install the latest commit from this repository, along with its Python dependencies: 

    pip install git+https://github.com/samjhecht/whatdisay.git

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/samjhecht/whatdisay.git

## CLI Usage

First, run the `--configure` command to configure the library.  If you don't plan to leverage speaker diarization features, you can simply leave the config properties blank, but you'll still be required to create a `config.yaml` file the first time you run the CLI.  You'll be prompt to update it later if you attempt to use functionality that requires a property that was not set up front.

    whatdisay --configure

The following command will take an audio file and generate a transcription using OpenAI Whisper:

    whatdisay --transcript audio_filename.wav

To generate a diarized transcript:

    whatdisay --transcript audio_filename.wav --diarize

Currently only wav files are supported for audio file inputs.

By default, it will use Whisper's `large` model and Deepgram's "Enhanced" tier `meeting` model.  If you would like to change either to use other available models, you can do so via your `config.yaml` file.  Documentation on available models found [here](https://developers.deepgram.com/documentation/features/model/) for Deepgram and [here](https://github.com/openai/whisper) for Whisper.


## TODOs:
- Add functionality to allow for customization of location for transcription output directory.
- add support for other file types for input audio besides wav
- make the transcription and diarization faster for longer files by using asyncio for whisper transcription step
- add tool that assists in a cleanup step after the diarization is complete to allow the user to assign human names to replace the values for 'SPEAKER_1','SPEAKER_2', etc.
- potentially add option to parallelize whisper transcription when someone just runs transcript w/o diarize, by chopping up big file and running multiple async whisper tasks