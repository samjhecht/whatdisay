#!/usr/bin/env python3

from whatdisay.utils import FileNameHelper, millisec
from pydub import AudioSegment
from whatdisay.diarize import Diarize
from pathlib import Path
from whatdisay.audio import truncateAudio
import asyncio
import os
import re
import json
import webvtt
import whisper
from whisper.utils import write_txt,write_vtt


def generateWhisperTranscript(wav_file, filename_helper: FileNameHelper, model="medium", custom_name=""):
    """
    Uses OpenAI Whisper to generate a transcription from an audio file.

    Parameters
    ----------
    wav_file: str
        The path to the audio file that you want a transcription of.

    filename_helper: FileNameHelper
        An instantiated utils.FileNameHelper class that provides all the necessary directory names.
    
    model: str
        The whisper model instance (tiny, base, small, medium, large).   Will default to 'medium' if not set.

    custom_name: str
        Optional input to pass a desired filename prefix for the resulting whisper transcription files.  Needed for the diarization functions.
    """
    if not type(filename_helper) == FileNameHelper:
        raise ValueError('Parameter filename_helper must be of type FileNameHelper.')

    print(f'Beginning Whisper transcription from {wav_file}')
    model = whisper.load_model(model)
    result = model.transcribe(wav_file)

    if custom_name:
        whisper_filename = str(custom_name)
    else: 
        whisper_filename = str(Path(wav_file).stem)

    # TODO: do i actually need to use whisper's file writer for text? 

    # save TXT
    with open(os.path.join(filename_helper.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.txt'), "w", encoding="utf-8") as txt:
        write_txt(result["segments"], file=txt)
    print('Saved TXT file of whisper transcription at: {}'.format(os.path.join(
        filename_helper.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.txt')))

    # save VTT
    with open(os.path.join(filename_helper.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.vtt'), "w", encoding="utf-8") as vtt:
        write_vtt(result["segments"], file=vtt)
    print('Saved VTT file of whisper transcription at: {}'.format(os.path.join(
        filename_helper.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.vtt')))



def diarizedTranscriptPyannote(wav_file, filename_helper: FileNameHelper, model="medium"):
    """
    Run the whole shebang. Use Pyannote to create diarization and OpenAI Whisper to generate a transcription from an audio file.
    Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wav_file: str
        The path to the audio file from which you would like a diarized transcription.

    filename_helper: FileNameHelper
        An instantiated utils.FileNameHelper class that provides all the necessary directory names.
    
    model: str
        The whisper model instance (tiny, base, small, medium, large).   Will default to 'medium' if not set.
    """

    dz = Diarize(filename_helper,3).diarize_pyannote(wav_file)
    groups = dz[0]
    gidx = dz[1]
    
    # Make sure there's a directory to save the audio segment files in
    filename_helper.createTaskDir(filename_helper.whisper_transcriptions_dir)

    for i in range(gidx+1):
        segment_audio_filename = filename_helper.dia_segments_dir, str(gidx) + '.wav'
        generateWhisperTranscript(segment_audio_filename, filename_helper,'large', i)
    
    gidx = -1

    final_output_file = os.path.join(filename_helper.diarized_transcriptions_dir, filename_helper.task_name + ".txt")

    with open(final_output_file, "w", encoding="utf-8") as text_file:
        for g in groups:
            shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0] # the start time in the original video
            shift = millisec(shift) - 2000
            shift = max(shift, 0)

            gidx += 1

            vtt_file = os.path.join(filename_helper.tmp_file_dir, 'whisper_transcriptions/' + str(gidx) + '_whisper.vtt')
            captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(vtt_file)]

            if captions:
                speaker = g[0].split()[-1]

                for c in captions:
                    text_file.write(f'{speaker}: {c[2]}')

    text_file.close()
    
    print(f'Saved diarized transcript at location: {final_output_file}')


def getWhisperTxt(wav_file, model="medium"):

    print(f'Beginning Whisper transcription from {wav_file}')
    model = whisper.load_model(model)
    w = model.transcribe(wav_file)    
    transcript_txt: str = w["text"]
    
    return transcript_txt


def diarizedTranscriptDeepgram(
    wav_file, 
    fh: FileNameHelper, 
    model="large"
    ):
    """
    Run the whole shebang. Use Deepgram to get the speaker diarization segments and then run OpenAI Whisper
    over each segment to generate a transcription from an audio file. Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wav_file: str
        The path to the audio file from which you would like a diarized transcription.

    fh: FileNameHelper
        An instantiated utils.FileNameHelper class that provides all the necessary directory names.
    
    model: str
        The Whisper model instance (tiny, base, small, medium, large).   Will default to 'large' if not set.
    """

    dz = asyncio.run(Diarize(fh).diarize_deepgram(wav_file))

    audio = AudioSegment.from_wav(wav_file)

    # Make sure there's a directory to save the audio segment files in
    fh.createTaskDir(fh.dia_segments_dir)

    idx = 0
    for segment in dz:
        start = float(segment[0]) * 1000
        end = float(segment[1]) * 1000

        output_af_name = os.path.join(fh.dia_segments_dir + str(idx) + '.wav')
        audio[start:end].export(output_af_name, format='wav')
        idx += 1
        

    final_output_file = os.path.join(fh.diarized_transcriptions_dir, fh.task_name + ".txt")

    with open(final_output_file, "w", encoding="utf-8") as text_file:

        for i in range(len(dz)):
            segment_audio = os.path.join(fh.dia_segments_dir, str(i) + '.wav')
            speaker = 'Speaker_' + str(dz[i][2])
            w = getWhisperTxt(segment_audio,"large")

            text_file.write(f'{speaker}: {w}\n')
            print(f'{speaker}: {w}')

    text_file.close()

    print(f'Saved diarized transcript at location: {final_output_file}')

        