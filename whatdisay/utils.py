#!/usr/bin/env python3

import os
import shutil
from pydub import AudioSegment


# def cli():
#     TODO

def millisec(timeStr):
  spl = timeStr.split(":")
  s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
  return s

# Add 0.5 second buffer to beginning of audio file to avoid loss on transcription.  accepts audio file name.
def addIntroSpacer(input_audio,output_dir):
    audio = AudioSegment.from_wav(input_audio)
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = spacer.append(audio, crossfade=0)

    filename_spacer = output_dir + 'audio_intro_spacer.wav'
    audio.export(filename_spacer, format='wav')
    print('Saved new copy of {} at {}'.format(input_audio,filename_spacer))



class BinPaths:

    def __init__(self, event_name, task_ts):
        self.task_name = event_name + '_' + task_ts
        self.event_name = event_name
        self.bin_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__name__))) + '/bin/'
        self.task_dir = self.bin_dir + str(self.task_name) + '/'
        self.tmp_file_dir = self.task_dir + 'tmp_files/' 
        self.pipelines_dir = self.bin_dir + 'pipelines/'
        self.sd_pipeline = self.pipelines_dir + 'models--pyannote--speaker-diarization'
        self.new_recordings_dir = self.bin_dir + 'new_recordings/'

    def createTaskDirectories(self):
        if not os.path.exists(self.bin_dir):
            print('bin directory not detected.  creating one..')
            os.makedirs(self.bin_dir)
        if not os.path.exists(self.task_dir):
            print('Task directory not detected.  creating one..')
            os.makedirs(self.task_dir)
        if not os.path.exists(self.tmp_file_dir):
            print('Tmp file directory not detected.  creating one..')
            os.makedirs(self.tmp_file_dir)

    def cleanup(self):
        # remove the tmp_files
        p = self.tmp_file_dir
        if os.path.exists(p):
            print('Deleting tmp file directory at: {}'.format(p))
            shutil.rmtree(self.tmp_file_dir)
        else:
            print('No tmp directory found for task: {}'.format(self.task_name))

