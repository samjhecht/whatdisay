#!/usr/bin/env python3

from pyannote.audio import Pipeline
from whatdisay.utils import BinPaths
from pydub import AudioSegment
from dotenv import load_dotenv
import os, shutil
import re

class Diarize():

    def __init__(self, binpaths: BinPaths):
        if not type(binpaths) == BinPaths:
            raise ValueError('Parameter binpaths must be of type BinPaths.')
        self.tmp_file_dir = binpaths.tmp_file_dir
        self.pipelines_cash_dir = binpaths.pipelines_dir
        self.sd_pipe_cash_dir = binpaths.sd_pipeline
        self.spacermilli = 2000
    
    # Instantiate pretrained speaker diarization pipeline
    def create_diarization(
        self,
        audio_file,
        num_speakers = None
        ):

        load_dotenv()
        huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

        print('instantiating pretrained pipeline')
        cd = self.pipelines_cash_dir
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token=huggingface_token,cache_dir=cd)

        # apply the pipeline to an audio file
        print('Applying the pipeline to audio file')
        diarization = pipeline(audio_file)
        # print(*list(diarization.itertracks(yield_label = True))[:10], sep="\n")

        # dump the diarization output to disk using text format
        print('Dumping the diarization output to disk using text format')
        d = self.tmp_file_dir + 'diarization.txt'
        
        with open(os.path.join(self.tmp_file_dir, "diarization.txt"), "w", encoding="utf-8") as text_file:
            text_file.write(str(diarization))
        print('Generated diarization file at {}'.format(d))

        audio = AudioSegment.from_wav(audio_file)
        spacer = AudioSegment.silent(duration=self.spacermilli)

        ## Now it's time to create the diarization
        def millisec(timeStr):
            spl = timeStr.split(":")
            s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
            return s


        dz = open(os.path.join(self.tmp_file_dir,"diarization.txt")).read().splitlines()
        dzList = []
        for l in dz:
            start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
            start = millisec(start) - self.spacermilli
            end = millisec(end)  - self.spacermilli
            speaker = re.findall('SPEAKER_\w+',string=l)[0] # extract which speaker it is
            dzList.append([start, end, speaker])

        # print(*dzList[:10], sep='\n')

        ## Attaching audio segements according to the diarization, with a spacer as the delimiter.
        sounds = spacer
        segments = []

        dz = open(os.path.join(self.tmp_file_dir,"diarization.txt")).read().splitlines()
        for l in dz:
            start, end =  tuple(re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
            start = int(millisec(start)) #milliseconds
            end = int(millisec(end))  #milliseconds
            
            segments.append(len(sounds))
            sounds = sounds.append(audio[start:end], crossfade=0)
            sounds = sounds.append(spacer, crossfade=0)

        diarization_af = os.path.join(self.tmp_file_dir, 'dz.wav')
        sounds.export(diarization_af, format="wav") #Exports to a wav file with spacers.
        print('Saved audio file spaced according to diarization: {}'.format(diarization_af))

        ## Free up some memory cuz we've been at if for a while
        print('Freeing up some memory because we\'ve been at it a while!')
        del sounds, spacer, audio, dz

        return segments, dzList


    def reset_pretrained_pipeline(self):
        # delete previously cached model if it exists
        s = self.sd_pipe_cash_dir
        print(s)
        if os.path.exists(s):
            print('Deleting previously cached pipeline...')
            shutil.rmtree(s)
        else:
            print('No cached pipeline found.  proceeding to download one.')
        
        cd = self.pipelines_cash_dir
        
        try:
            pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token='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',cache_dir=cd)
        except OSError:
            print('Something went wrong trying to get pyannote pipeline.')
        else:
            print('Successfully downloaded pyannote/speaker-diarization pre-trained model.')




