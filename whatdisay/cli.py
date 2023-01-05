#!/usr/bin/env python3

from whatdisay.utils import TaskProps, MdFileUtil, millisec, check_file_is_valid, getTaskName
from whatdisay.config import Config
from whatdisay.diarize import Diarize
import whatdisay.transcribe as transcribe
from whatdisay.audio import truncateAudio
from datetime import datetime
import asyncio
import time
import argparse
import re
import logging
import shutil, os
import sys

def enableDebugMode():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.debug("Debug mode enabled.")

def resetPyannotePipe(tp):
    Diarize(tp).reset_pretrained_pipeline()

def runTruncateAudio(args, tp):
    
    t = args.pop('truncate_audio')
    af = t.truncate[0]
    start = t.truncate[1]
    end = t.truncate[2]

    # Check that the time is in the format "HH:mm:SS"
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", start):
        raise argparse.ArgumentTypeError("start timestamp must be in the format 'HH:mm:SS'")
    if not re.match(r"^\d{2}:\d{2}:\d{2}$", end):
        raise argparse.ArgumentTypeError("end timestamp must be in the format 'HH:mm:SS'")

    if check_file_is_valid(af):
        t1 = millisec(start)
        t2 = millisec(end)
        truncateAudio(af, t1, t2, tp)

def runTranscription(args, tp: TaskProps):

    start_time = time.time()
    
    debug_mode = args.get('debug')
    get_transcript = args.pop('transcript')
    diarize = args.pop('diarize')
    generate_md = args.pop('generate_markdown')

    tp.createAllTaskDirectories()

    # If diarization model specified, enforce that it's either 'deepgram' or 'pyannote'.  
    if diarize:
        if diarize not in ["pyannote","deepgram","whisper_local"]:
            raise ValueError("Invalid value for --diarize argument.  Must be 'deepgram' or 'pyannote'.")

    if get_transcript:
        wav_file = get_transcript
        if check_file_is_valid(wav_file):
            
            if generate_md:
                md_title = input("Input title for markdown file: ")
                tags = input("Input comma-separated list of tags to add to markdown for Obsidian: ")

            if diarize:
                print(f'Generating transcript using {diarize} for diarization...')
                start_time = time.time()
                if diarize == 'pyannote':
                    transcribe.diarizedTranscriptPyannote(wav_file,tp)
                elif diarize == 'whisper_local':
                    whisper_model = Config().get_param('WHISPER_MODEL')
                    asyncio.run(transcribe.diarizedTranscriptDeepgramWhisperLocal(wav_file, whisper_model, tp))
                    run_time = time.time() - start_time
                    print(f'whisper local run time: {run_time}')
                else:
                    asyncio.run(transcribe.diarizedTranscriptAllDeepgram(wav_file,tp))
                    run_time = time.time() - start_time
                    print(f'async deepgram run time: {run_time}')
            else:
                whisper_model = Config().get_param('WHISPER_MODEL')
                transcribe.generateWhisperTranscript(wav_file,tp, whisper_model)
                
                # Move the whisper transcription to the transcriptions directory before the tmp_dir gets deleted later
                tmp_whisper = os.path.join(tp.whisper_transcriptions_dir, tp.task_name + "_whisper.txt")
                final_whisper = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")
                shutil.copy(tmp_whisper, final_whisper)
                print(f'Saved whisper transcription at: {final_whisper}')

            if generate_md:
                output_file_txt = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")
                output_file_md = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".md")

                md_file = MdFileUtil(output_file_txt, tags, md_title, tp)

                with open(output_file_txt, "r") as txt_file:
                    for line in txt_file:
                        md_file.append_line(line.strip())

                af_ctime = datetime.fromtimestamp(os.path.getctime(wav_file)).strftime('%Y-%m-%dT%H:%M:%S')
                md_file.append_line(f'\n\n\nTranscript generated from audio file originally created at: {af_ctime}')
                print(f'Saved Markdown file at location: {output_file_md}')
                    
            # Now that the job is done, delete the tmp files unless debug mode is on, in which case we'll save them for troubleshoting.
            if not debug_mode:
                tp.cleanupTask()

def cli():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument('--configure', action='store_true', help="Configure the CLI and create or update config yaml file.")
    exclusive_group.add_argument('--transcript', type=str, required=False, help="Generated diarized transcriptiion from an existing recording. Requires the path to the audio file you need a transcript of.")
    exclusive_group.add_argument('--truncate_audio', nargs=3, required=False, help="Trim an audio file using timestamps provided.")
    parser.add_argument('--diarize', nargs='?', const='deepgram', type=str, help="Diarize the transcript. Defaults to Deepgram for diarization model unless 'pyannote' is passed as a value.")
    parser.add_argument('--event_name', type=str, required=False)
    parser.add_argument('--reset_pipeline', help="Re-pull pyannote's speaker diarization pipeline.")
    parser.add_argument('--debug', action="store_true", help="Enable debug mode.")
    parser.add_argument('-md', '--generate_markdown',action="store_true", help="Generate a markdown version of the final transcript and add tags for Obsidian.")
    args = parser.parse_args().__dict__

    if args.get('debug'):
        enableDebugMode()

    if args.get('configure'):
        Config().configure()
        sys.exit(1)
    else:
        Config().get_config()

    task_name: str = getTaskName(args)
    tp = TaskProps(task_name)

    if args.get('reset_pipeline'):
        resetPyannotePipe(tp)
    elif args.get('transcript'):
        runTranscription(args, tp)
    elif args.get('truncate_audio'):
        runTruncateAudio(args, tp)
    else:
        parser.print_help()

if __name__ == '__main__':
    cli()

