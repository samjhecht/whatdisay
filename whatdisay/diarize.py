#!/usr/bin/env python3

from pyannote.audio import Pipeline
from whatdisay.utils import TaskProps
from pydub import AudioSegment
from dotenv import load_dotenv, find_dotenv
import os, shutil
import re
from deepgram import Deepgram
import json
import asyncio


class Diarize():

    def __init__(self, tp: TaskProps):
        if not type(tp) == TaskProps:
            raise ValueError('Parameter tp must be of type TaskProps.')
        self.tp = tp
        self.tmp_file_dir = tp.tmp_file_dir
        self.pipelines_cash_dir = tp.pipelines_dir
        self.sd_pipe_cash_dir = tp.sd_pipeline
        self.pyannote_dia_txt_file = tp.pyannote_diarization_txt
        self.diarized_audio_file = tp.diarized_audio_file
        self.spacermilli = 2000

    def add_intro_spacer(self,audio_file):
        raw_audio = AudioSegment.from_wav(audio_file)
        spacer = AudioSegment.silent(duration=self.spacermilli)
        audio = spacer.append(raw_audio,crossfade=0)
        return audio

    def load_pipeline(self):
        '''
        Instantiate pretrained speaker diarization pipeline
        '''
        
        pipe_type: str = 'pyannote/speaker-diarization'

        load_dotenv()
        huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

        print('instantiating pretrained pipeline')
        cd = self.pipelines_cash_dir
        pipeline = Pipeline.from_pretrained(pipe_type,use_auth_token=huggingface_token,cache_dir=cd)
        
        return pipeline

    def apply_pipeline(self, audio_file):
        
        # pyannote.audio apparently misses the first 0.5 seconds of the audio, so we'll add a spacer at the beginning to compensate
        audio_intro_spacer = self.add_intro_spacer(audio_file)
        new_audio = os.path.join(self.tmp_file_dir,'spaced_audio.wav')
        audio_intro_spacer.export(new_audio,format="wav")
        print('Added intro spacer to audio_file and saved new wav file at: {}'.format(new_audio))

        pipeline = self.load_pipeline()
        # apply the pipeline to an audio file
        print('Applying the pipeline to audio file')
        diarization = pipeline(new_audio)
        # print(*list(diarization.itertracks(yield_label = True))[:10], sep="\n")
        print('Finished applying pipeline to audio_file named: {}'.format(new_audio))
        return diarization


    def diarize_pyannote(
        self,
        audio_file,
        num_speaker=None
    ):

        diarization = self.apply_pipeline(audio_file)
        
        # dump the diarization output to disk using text format
        print('Dumping the diarization output to a text file for now.')
        
        with open(self.pyannote_dia_txt_file, "w", encoding="utf-8") as text_file:
            text_file.write(str(diarization))
        print('Generated diarization file at {}'.format(self.pyannote_dia_txt_file))

        af_w_intro_spacer = os.path.join(self.tmp_file_dir,'spaced_audio.wav')
        audio = AudioSegment.from_wav(af_w_intro_spacer)

        ## Now it's time to create the diarization
        def millisec(timeStr):
            spl = timeStr.split(":")
            s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2]) )* 1000)
            return s
        
        raw_dz = open(self.pyannote_dia_txt_file).read().splitlines()
        groups = []
        g = []
        previous_end = 0

        for d in raw_dz: 
            if g and (g[0].split()[-1] != d.split()[-1]):      #same speaker
                groups.append(g)
                g = []
  
            g.append(d)
            
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=d)[1]
            end = millisec(end)
            if (previous_end > end):       #segment engulfed by a previous segment
                groups.append(g)
                g = [] 
            else:
                previous_end = end
        
        if g:
            groups.append(g)     

        # Make sure there's a directory to save the audio segment files in
        self.tp.createTaskDir(self.tp.dia_segments_dir)

        af_w_intro_spacer = os.path.join(self.tmp_file_dir,'spaced_audio.wav')
        audio = AudioSegment.from_wav(af_w_intro_spacer)
        gidx = -1
        for g in groups:
            start = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0]
            end = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[-1])[1]
            start = millisec(start) #- spacermilli
            end = millisec(end)  #- spacermilli
            print(start, end)
            gidx += 1
            output_af_name = os.path.join(self.tmp_file_dir, 'audio_segments/' + str(gidx) + '.wav')
            audio[start:end].export(output_af_name, format='wav')
            print(f'Saved segment audio file at: {output_af_name}')

        return groups, gidx


    async def diarize_deepgram(self, audio_file):

        BASE_DIR = os.path.abspath(os.path.dirname(__name__))
        load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
        deepgram_api_key = os.environ.get('DEEPGRAM_API_KEY')

        # Initialize the Deepgram SDK
        deepgram = Deepgram(deepgram_api_key)

        with open(audio_file,'rb') as audio:
            source = {'buffer': audio, 'mimetype': 'audio/wav'}

            response = await asyncio.create_task(
                deepgram.transcription.prerecorded(
                    source,
                    {
                        'punctuate': True, 
                        'diarize': True, 
                        'utterances': True,
                        'tier': 'enhanced', 
                        # 'model': 'whisper'}
                        'model': 'meeting'}
                )
            )

        output_json = json.dumps(response)
        j = json.loads(output_json)
        utterances = j['results']['utterances']

        # deepgram_to_file = os.path.join(self.tmp_file_dir,'deepgram_output.json')
        # with open(deepgram_to_file,'w',encoding='utf-8') as f:
        #     json.dump(response,f, ensure_ascii=False, indent=4)
            
        segments = []
        if utterances:
            previous_speaker = -1
            current_caption = ""
            previous_start = 0
            previous_end = 0
            
            for u in utterances:
                speaker  = u['speaker']
                start = u['start']
                end = u['end']

                # confidence = u['confidence']
                caption = u['transcript']                

                if speaker == previous_speaker:
                    current_caption += "  " + caption
                    previous_end = end
                else:
                    if previous_speaker != -1:
                        segments.append([previous_start, previous_end, previous_speaker, current_caption])
                    current_caption = caption
                    previous_speaker = speaker
                    previous_start = start
                    previous_end = end

            # add the last row
            segments.append([previous_start, previous_end, previous_speaker, current_caption])
        
        return segments


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
        load_dotenv()
        huggingface_token = os.environ.get('HUGGINGFACE_TOKEN')

        try:
            pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token=huggingface_token,cache_dir=cd)
        except OSError:
            print('Something went wrong trying to get pyannote pipeline.')
        else:
            print('Successfully downloaded pyannote/speaker-diarization pre-trained model.')
