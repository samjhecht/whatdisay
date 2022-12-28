#!/usr/bin/env python3

import os
import sys
import shutil
from pathlib import Path
from pydub import AudioSegment
import time

def getTaskName(args:dict) -> str:

    event_name = args.get('event_name')
    ts_task: str = str(int(time.time() * 1000))

    if not event_name:
        event_name = input("Provide a name for this event: ")
        if event_name == '':
            event_name = ''
        else:
            event_name = event_name.replace(' ', '_')
            event_name = event_name + '_'
    else:
        event_name = event_name.replace(' ', '_')
        event_name = event_name + '_'
    
    event_name = event_name.lower() + ts_task
    
    print(f'Running task for event with name: {event_name}')
    return event_name

def millisec(timeStr: str) -> int:
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

def check_file_is_valid(f: str) -> bool:
    if not Path(f).resolve(strict=True):
        raise FileNotFoundError(f"Audio file not found at: {f}")
    
    filename, file_extension = os.path.splitext(f)
    if file_extension != ".wav":
        raise ValueError("Audio file must be '.wav' format.  Other filetypes not yet supported.")

    return True

# Add 0.5 second buffer to beginning of audio file to avoid loss on transcription.  accepts audio file name.
def addIntroSpacer(input_audio,output_dir):
    audio = AudioSegment.from_wav(input_audio)
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = spacer.append(audio, crossfade=0)

    filename_spacer = output_dir + 'audio_intro_spacer.wav'
    audio.export(filename_spacer, format='wav')
    print('Saved new copy of {} at {}'.format(input_audio,filename_spacer))

class TaskProps:

    def __init__(self, task_name):
        self.task_name = task_name
        self.output_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__name__))) + '/output/'
        self.diarized_transcriptions_dir = self.output_dir + 'diarized_transcriptions/'
        self.tasks_dir = self.output_dir + 'tasks/'
        self.task_dir = self.tasks_dir + str(self.task_name) + '/'
        self.tmp_file_dir = self.task_dir + 'tmp_files/' 
        self.dia_segments_dir = self.tmp_file_dir + 'audio_segments/' 
        self.whisper_transcriptions_dir = self.tmp_file_dir + 'whisper_transcriptions/'
        self.pipelines_dir = self.output_dir + 'pipelines/'
        self.sd_pipeline = self.pipelines_dir + 'models--pyannote--speaker-diarization'
        self.new_recordings_dir = self.output_dir + 'new_recordings/'
        self.pyannote_diarization_txt = os.path.join(self.tmp_file_dir,'diarization.txt')
        self.diarized_audio_file = os.path.join(self.tmp_file_dir, 'dz.wav')

    def createTaskDir(self,dir):
        if not os.path.exists(dir):
            print('Output directory not detected.  creating one..')
            os.makedirs(dir)

    def createAllTaskDirectories(self):
        if not os.path.exists(self.output_dir):
            print('Output directory not detected.  creating one..')
            os.makedirs(self.output_dir)
        if not os.path.exists(self.tasks_dir):
            print('Output directory not detected.  creating one..')
            os.makedirs(self.tasks_dir)
        if not os.path.exists(self.task_dir):
            print('Output directory not detected.  creating one..')
            os.makedirs(self.task_dir)
        if not os.path.exists(self.tmp_file_dir):
            print('Tmp file directory not detected.  creating one..')
            os.makedirs(self.tmp_file_dir)
        if not os.path.exists(self.task_dir):
            print('Task directory not detected.  creating one..')
            os.makedirs(self.task_dir)
        if not os.path.exists(self.diarized_transcriptions_dir):
            print('Directory for final transcription files not detected.  creating one..')
            os.makedirs(self.diarized_transcriptions_dir)
        if not os.path.exists(self.whisper_transcriptions_dir):
            print('Directory for whisper transcription files not detected.  creating one..')
            os.makedirs(self.whisper_transcriptions_dir)
        if not os.path.exists(self.dia_segments_dir):
            print('Directory for diarization segments files not detected.  creating one..')
            os.makedirs(self.dia_segments_dir)

    def cleanupTask(self):
        # remove the task directory and delete all intermediate files used in creating the diarized transcript
        p = self.task_dir
        if os.path.exists(p):
            print('Deleting task tmp file directory at: {}'.format(p))
            shutil.rmtree(self.task_dir)
        else:
            print('No task tmp directory found for task: {}'.format(self.task_name))

    def cleanupAllTasks(self):
        # remove all tasks by deleting entire tmp directory
        p = self.tmp_file_dir
        if os.path.exists(p):
            print('Deleting tmp file directory at: {}'.format(p))
            shutil.rmtree(self.tmp_file_dir)
        else:
            print('No tmp directory found for task: {}'.format(self.task_name))


class MdFileUtil:

    def __init__(self, in_file: str, tags: str, title: str, tp: TaskProps):
        self.in_file = in_file
        self.tags = tags.replace(' ', '-').split(',')
        self.title = title
        self.diarized_transcriptions_dir = tp.diarized_transcriptions_dir
        self.out_file = self.get_md_file()
        self.add_tags = self.add_tags_and_title()
    
    def get_md_file(self):

        try:
            if Path(self.in_file).resolve(strict=True):
                n = Path(self.in_file).stem
                outfile_path = os.path.join(self.diarized_transcriptions_dir, n + '.md')
                md_file = open(outfile_path, 'w+', encoding="utf-8")
                md_file.close()
                print(f'Creating new markdown file at: {outfile_path}')
        
        except FileNotFoundError:

            print(f"No input file found at {self.in_file}.")
            sys.exit(1)
        
        return outfile_path

    def add_tags_and_title(self):
        
        if self.tags:

            tag_list = self.tags
            obsidian_yaml_block = """---\ntags:"""

            for tag in tag_list:
                obsidian_yaml_block = obsidian_yaml_block + f"\n- {tag}"

            obsidian_yaml_block = obsidian_yaml_block + "\n---"
            
            with open(self.out_file, 'r+', encoding="utf-8") as md_file:
                existing_data = md_file.read()
                md_file.seek(0,0)
                md_file.write(obsidian_yaml_block)
                md_file.write("\n")
                md_file.write(f'# {self.title}')
                md_file.write("\n" + existing_data)

    def append_line(self, t):

        with open(self.out_file, 'a', encoding="utf-8") as md_file:
            md_file.write(t)
            md_file.write("\n")


    


