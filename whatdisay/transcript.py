#!/usr/bin/env python3

from .utils import BinPaths, millisec
from .diarize import Diarize
import argparse
import time
import os
import webvtt
import whisper
import logging
from pathlib import Path


def generateWhisperTranscript(wavFile, binpaths: BinPaths, model="medium"):
    """
    Uses OpenAI Whisper to generate a transcription from an audio file.

    Parameters
    ----------
    wavFile: str
        The path to the audio file that you want a transcription of.

    binpaths: BinPaths
        An instantiated utils.BinPaths class that provides all the necessary directory names.
    
    model: str
        The whisper model instance (tiny, base, small, medium, large).   Will default to 'medium' if not set.
    """
    if not type(binpaths) == BinPaths:
        raise ValueError('Parameter binpaths must be of type BinPaths.')

    print('Beginning Whisper transcption from {}'.format(str(wavFile)))
    model = whisper.load_model("medium")
    result = model.transcribe(wavFile)

    task_name = binpaths.task_name
    
    # save TXT
    with open(os.path.join(binpaths.task_dir, task_name + "_whisper.txt"), "w", encoding="utf-8") as txt:
        whisper.write_txt(result["segments"], file=txt)
    print('Saved TXT file of whisper transcription at: {}'.format(os.path.join(binpaths.task_dir, task_name + "_whisper.txt")))
    
    # save VTT
    with open(os.path.join(binpaths.task_dir, task_name + "_whisper.vtt"), "w", encoding="utf-8") as vtt:
        whisper.write_vtt(result["segments"], file=vtt)
    print('Saved VTT file of whisper transcription at: {}'.format(os.path.join(binpaths.task_dir, task_name + "_whisper.vtt")))


def generateDiarizedTranscript(wavFile, binpaths: BinPaths, model="medium"):
    """
    Run the whole shebang. Use Pyannote to create diarization and OpenAI Whisper to generate a transcription from an audio file.
    Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wavFile: str
        The path to the audio file from which you would like a diarized transcription.

    binpaths: BinPaths
        An instantiated utils.BinPaths class that provides all the necessary directory names.
    
    model: str
        The whisper model instance (tiny, base, small, medium, large).   Will default to 'medium' if not set.
    """

    dz = Diarize(binpaths).create_diarization(wavFile)
    segments = dz[0]
    dzList = dz[1]
    dz_wav = os.path.join(binpaths.tmp_file_dir,'dz.wav')
    generateWhisperTranscript(dz_wav, binpaths)

    final_output_file = os.path.join(binpaths.transcriptions_dir, binpaths.event_name + ".txt")
    vtt_file = os.path.join(binpaths.task_dir,binpaths.task_name + "_whisper.vtt")

    # Use webvtt-py to read the transcription file generated using whisper
    captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(vtt_file)]
    # print(*captions[:8], sep='\n')

    with open(final_output_file, "w", encoding="utf-8") as text_file:
        
        for i in range(len(segments)):
            idx = 0
            for idx in range(len(captions)):
                if captions[idx][0] >= (segments[i] - 2000):
                    break;
            
            while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
                c = captions[idx]  
                
                start = dzList[i][0] + (c[0] -segments[i])

                if start < 0: 
                    start = 0
                idx += 1

                start = start / 1000.0
                
                speaker_name = dzList[i][2]
                caption = c[2]
                formatted_entry = '{}: {}\n'.format(speaker_name,caption)

                text_file.write(formatted_entry)

def cli():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fe', '--from-existing', type=str, required=True, help="Generated diarized transcriptiion from an existing recording.")
    parser.add_argument('-r','--new-recording', action="store_true", help="Will kick off a new recording.")
    parser.add_argument('-reset','--reset-pipeline', help="Re-pull pyannote's speaker diarization pipeline.")
    parser.add_argument('--debug', action="store_true", help="Enable debug mode.")
    args = parser.parse_args()

    ts_task = str(int(time.time() * 1000))

    if args.l:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.debug("Here we go!")

    event_name = input("Provide a name for this event: ")
    if not event_name:
        event_name = 'new_task'
    else:
        event_name = event_name.replace(' ', '_')

    
    TASK_NAME = str(event_name) + '_' + ts_task
    print('the task name is: {}'.format(TASK_NAME))

    bin_paths = BinPaths(TASK_NAME)
    bin_paths.createTaskDirectories()

    if args.reset:
        Diarize(bin_paths).reset_pretrained_pipeline()

    if args.fe:
        wav_file = args.fe
        try:
            Path(wav_file).resolve(strict=True)
        except FileNotFoundError:
            print('You did not pass a valid file name.')
        else:
            # people = input("Input comma-separated list of participants: ")
            tags = input("Input comma-separated list of tags to add to markdown for Obsidian: ")

            generateDiarizedTranscript(wav_file,bin_paths)
            
            # Todo add something to parse input tags and then append them in a format obsidian will like.

            # Now that the job is done, delete the tmp files unless debug mode is on, in which case we'll save them for troubleshoting.
            if not args.l:
                bin_paths.cleanupTask()


if __name__ == '__main__':
    cli()