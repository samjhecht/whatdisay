#!/usr/bin/env python3

from utils import TaskProps, millisec
from diarize import Diarize
import transcribe
from audio import truncateAudio
import argparse
import time
import logging
from pathlib import Path
import shutil, os

def cli():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-fe', '--from_existing', type=str, required=True, help="Generated diarized transcriptiion from an existing recording.")
    parser.add_argument('-t','--raw_transcript_only',action="store_true", help="Set this flag if you just want the transcript without a diarization.")
    parser.add_argument('--event_name', type=str, required=False)
    parser.add_argument('-r','--new_recording', action="store_true", help="Will kick off a new recording.")
    parser.add_argument('--reset','--reset_pipeline', help="Re-pull pyannote's speaker diarization pipeline.")
    parser.add_argument('--truncate_audio',type=int) # todo - add more handling for default and errors
    parser.add_argument('--debug', action="store_true", help="Enable debug mode.")
    parser.add_argument('-dm','--diarization_model', type=str, help="Optionally choose whether to use Deepgram or Pyannote for diarization.")
    parser.add_argument('-md', '--generate_markdown',action="store_true", help="Generate a markdown version of the final transcript and add tags for Obsidian.")
    args = parser.parse_args().__dict__

    debug_mode = args.pop('debug')
    run_from_existing = args.pop('from_existing')
    just_transcript = args.pop('raw_transcript_only')
    event_name = args.pop('event_name')
    reset_pipeline = args.pop('reset')
    truncate_audio = args.pop('truncate_audio')
    dia_model = args.pop('diarization_model')
    generate_md = args.pop('generate_markdown')

    ts_task = str(int(time.time() * 1000))

    if debug_mode:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        logging.debug("Debug mode enabled.")

    if event_name:
        print('Running task for event with name: {}'.format(event_name))
    else:
        event_name = input("Provide a name for this event: ")
        if event_name == '':
            event_name = 'new_task'
        else:
            event_name = event_name
    event_name = event_name.replace(' ', '_')

    tp = TaskProps(event_name,ts_task)
    tp.createAllTaskDirectories()

    if reset_pipeline:
        Diarize(tp).reset_pretrained_pipeline()

    # If diarization model specified, enforce that it's either 'd' or 'p'.  Otherwise, default to deepgram.
    if dia_model:
        if dia_model != 'p' and dia_model != 'd':
            raise ValueError("Diarization model must value must be either 'd' or 'p'.")
    else:
        dia_model = 'd'

    if run_from_existing:
        wav_file = run_from_existing
        try:
            Path(wav_file).resolve(strict=True)
        except FileNotFoundError:
            print('You did not pass a valid file name.')
        else:
            
            if generate_md:
                md_title = input("Input title for markdown file: ")
                tags = input("Input comma-separated list of tags to add to markdown for Obsidian: ")

            if just_transcript:
                transcribe.generateWhisperTranscript(wav_file,tp)
                
                # Move the whisper transcription to the transcriptions directory before the tmp_dir gets deleted later
                tmp_whisper = os.path.join(tp.tmp_file_dir, tp.task_name + "_whisper.txt")
                final_whisper = os.path.join(tp.whisper_transcriptions_dir, tp.task_name + "_whisper.txt")
                shutil.copy(tmp_whisper, final_whisper)
                print(f'Saved whisper transcription at: {final_whisper}')
            
            else:
                if dia_model == 'p':
                    transcribe.diarizedTranscriptPyannote(wav_file,tp)
                else:
                    transcribe.diarizedTranscriptDeepgram(wav_file,tp)
            
            # Todo add something to parse input tags and then append them in a format obsidian will like.
            if generate_md:
                output_file_txt = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")
                output_file_md = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".md")

                with open(output_file_md, "w", encoding="utf-8") as md_file:

                    tag_list = tags.replace(' ', '-').split(',')
                    
                    obsidian_yaml_block = """---\ntags:"""

                    for tag in tag_list:
                        obsidian_yaml_block = obsidian_yaml_block + f'\n- {tag}'

                    obsidian_yaml_block = obsidian_yaml_block + '\n---\n'

                    md_file.write(obsidian_yaml_block)
                    md_file.write(f'# {md_title}\n')
                    md_file.write('\n')

                    with open(output_file_txt,"r") as txt_file:
                        for line in txt_file:
                            md_file.write(line.strip())

                    
            # Now that the job is done, delete the tmp files unless debug mode is on, in which case we'll save them for troubleshoting.
            if not debug_mode:
                tp.cleanupTask()

    # if truncate_audio:
    #     wav_file = run_from_existing
    #     try:
    #         Path(wav_file).resolve(strict=True)
    #     except FileNotFoundError:
    #         print('You did not pass a valid file name.')
    #     else:
    #         t2: int = 3 * 60 * 1000
    #         truncateAudio(0,t2,wav_file,tp)

if __name__ == '__main__':
    cli()