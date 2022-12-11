#!/usr/bin/env python3

from whatdisay.utils import TaskProps, millisec, check_file_is_valid, getTaskName
from whatdisay.diarize import Diarize
import whatdisay.transcribe as transcribe
from whatdisay.audio import truncateAudio
import argparse, yaml
import re
import logging
import shutil, os


def configure(args):
    config = {}
    config['OBSIDIAN_DIR'] = input("Enter the location for obsidian markdown files to be saved for Obsidian.")
    config['DEEPGRAM_API_KEY'] = input("Enter your Deepgram API Key.  https://deepgram.com/ ...")
    config['HUGGINGFACE_TOKEN'] = input("Enter your Huggingface API Token. https://huggingface.co/ ...")
    config['OUTPUT_DIR'] = input("Optionally specify a directory for all task output to be saved...")

    # Save to a yaml file in the project root directory.
    BASE_DIR = os.path.abspath(os.path.dirname(__name__))
    config_file =os.path.join(BASE_DIR, "config.yaml")
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

def checkConfig(args: dict):

    try:
        BASE_DIR = os.path.abspath(os.path.dirname(__name__))
        config_file =os.path.join(BASE_DIR, "config.yaml")
        with open(config_file, 'r') as f:
            config = yaml.load(f)
    except FileNotFoundError:

        print("Library not configured.  Run the CLI with the 'configure' argument first.")
        return

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

    debug_mode = args.get('debug')
    get_transcript = args.pop('transcript')
    diarize = args.pop('diarize')
    generate_md = args.pop('generate_markdown')

    tp.createAllTaskDirectories()

    # If diarization model specified, enforce that it's either 'deepgram' or 'pyannote'.  
    if diarize:
        if diarize not in ["pyannote","deepgram"]:
            raise ValueError("Invalid value for --diarize argument.  Must be 'deepgram' or 'pyannote'.")

    if get_transcript:
        wav_file = get_transcript
        if check_file_is_valid(wav_file):
            
            if generate_md:
                md_title = input("Input title for markdown file: ")
                tags = input("Input comma-separated list of tags to add to markdown for Obsidian: ")

            if diarize:
                print(f'Generating transcript using {diarize} for diarization...')
                if diarize == 'pyannote':
                    transcribe.diarizedTranscriptPyannote(wav_file,tp)
                else:
                    transcribe.diarizedTranscriptDeepgram(wav_file,tp)
            else:
                transcribe.generateWhisperTranscript(wav_file,tp)
                
                # Move the whisper transcription to the transcriptions directory before the tmp_dir gets deleted later
                tmp_whisper = os.path.join(tp.whisper_transcriptions_dir, tp.task_name + "_whisper.txt")
                final_whisper = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + "_whisper.txt")
                shutil.copy(tmp_whisper, final_whisper)
                print(f'Saved whisper transcription at: {final_whisper}')

            if generate_md:
                output_file_txt = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")
                output_file_md = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".md")

                with open(output_file_md, "w", encoding="utf-8") as md_file:

                    tag_list = tags.replace(' ', '-').split(',')
                    
                    obsidian_yaml_block = """---\ntags:"""

                    for tag in tag_list:
                        obsidian_yaml_block = obsidian_yaml_block + f'\n- {tag}'

                    obsidian_yaml_block = obsidian_yaml_block + '\n---'
                    md_file.write('\n')

                    md_file.write(obsidian_yaml_block)
                    md_file.write(f'# {md_title}\n')
                    md_file.write('\n')

                    with open(output_file_txt,"r") as txt_file:
                        for line in txt_file:
                            md_file.write(line.strip())
                            md_file.write('\n')
                    af_ctime = os.path.getctime(wav_file)
                    md_file.write(f'\n\n\nTranscript generated from audio file originally created at: {af_ctime}')
                    print(f'Saved Markdown file at location: {output_file_md}')
                    
            # Now that the job is done, delete the tmp files unless debug mode is on, in which case we'll save them for troubleshoting.
            if not debug_mode:
                tp.cleanupTask()

def cli():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    exclusive_group = parser.add_mutually_exclusive_group(required=False)
    exclusive_group.add_argument('--configure', action='store_true', help="Configure the CLI and create or update config yaml file.")
    exclusive_group.add_argument('--transcript', type=str, required=False, help="Generated diarized transcriptiion from an existing recording. Requires the path to the audio file you need a transcript of.")
    exclusive_group.add_argument('--new_recording', help="Will kick off a new recording.")
    exclusive_group.add_argument('--truncate_audio', nargs=3, required=False, help="Trim an audio file using timestamps provided.")
    parser.add_argument('--diarize', nargs='?', const='deepgram', type=str, help="Diarize the transcript. Defaults to Deepgram for diarization model unless 'pyannote' is passed as a value.")
    # parser.add_argument('--dia_model', type=str, choices=["deepgram","pyannote"])
    parser.add_argument('--event_name', type=str, required=False)
    parser.add_argument('--reset_pipeline', help="Re-pull pyannote's speaker diarization pipeline.")
    parser.add_argument('--debug', action="store_true", help="Enable debug mode.")
    parser.add_argument('-md', '--generate_markdown',action="store_true", help="Generate a markdown version of the final transcript and add tags for Obsidian.")
    args = parser.parse_args().__dict__

    if args.get('debug'):
        enableDebugMode()
    
    task_name: str = getTaskName(args)
    tp = TaskProps(task_name)

    if args.get('configure'):
        configure(args)
    else:
        checkConfig(args)

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