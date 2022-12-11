#!/usr/bin/env python3

import os
import shutil
from pathlib import Path
from pydub import AudioSegment
import time
from dotenv import load_dotenv

def getTaskName(args:dict) -> str:

    event_name = args.pop('event_name')
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
    
    event_name = event_name + ts_task
    
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
        self.obsidian_dir = os.environ.get('OBSIDIAN_DIRECTORY_LOCATION')

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


class ObsidianHelper:

    def __init__(self, base_file: str, tp: TaskProps):
        self.base_file = base_file
        self.md_file = self.get_md_file()
        self.task_name: str = tp.task_name
        self.obsidian_dir: str = tp.obsidian_dir
    
    def get_md_file(self):
        if os.path.exists(self.base_file) and self.base_file.endswith('.md'):
            md_file = open(self.base_file, 'w+', encoding="utf-8")
            md_file.close()
        else:
            raise Exception(f'No markdown file found at {self.base_file}')
        return md_file

    def add_tags(self, tags:str):
        tag_list = tags.replace(' ', '-').split(',')
        
        obsidian_yaml_block = """---\ntags:"""

        for tag in tag_list:
            obsidian_yaml_block = obsidian_yaml_block + f'\n- {tag}'

        obsidian_yaml_block = obsidian_yaml_block + '\n---'
        
        self.append_at_beginning(obsidian_yaml_block)

    def append_at_beginning(self, new_data):
        with open(self.base_file, 'r+', encoding="utf-8") as self.md_file:
            existing_data = self.md_file.read()
            self.md_file.seek(0,0)
            self.md_file.write(new_data)
            self.md_file.write('\n' + existing_data)
        self.md_file.close()


