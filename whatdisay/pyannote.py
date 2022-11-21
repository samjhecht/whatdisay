#!/usr/bin/env python3

import logging
from pyannote.audio import Pipeline
import os, shutil

class PyannoteHelpers():

    def __init__(self):
        self.pipelines_cash_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__name__))) + '/bin/pipelines',
        self.sd_pipe_cash_dir = self.pipelines_cash_dir[0] + '/models--pyannote--speaker-diarization/'
    
    # Instantiate pretrained speaker diarization pipeline
    def getPipeline(
        self,
        audio_file,
        num_speakers = None
        ):

        cd = self.pipelines_cash_dir[0]
        pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token='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',cache_dir=cd)
        diarization = pipeline

        # apply the pipeline to an audio file
        print(f'Applying the pipeline to audio file')
        diarization = pipeline(audio_file)
        # print(*list(diarization.itertracks(yield_label = True))[:10], sep="\n")

        # dump the diarization output to disk using text format
        print(f'Dumping the diarization output to disk using text format')
        f = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__name__))) + '/bin/transcriptions_raw/diarization.txt'
        print(f)
        with open(f, "w") as text_file:
            text_file.write(str(diarization))


    def reset_pretrained_pipeline(self):
        # delete previously cached model if it exists
        s = self.sd_pipe_cash_dir
        print(s)
        if os.path.exists(s):
            print(f'Deleting previously cached pipeline...')
            shutil.rmtree(s)
        else:
            print(f'No cached pipeline found.  proceeding to download one.')
        
        cd = self.pipelines_cash_dir[0]
        print(cd)
        try:
            pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization',use_auth_token='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',cache_dir=cd)
        except OSError:
            print(f'Something went wrong trying to get pyannote pipeline.')
        else:
            print(f'Successfully downloaded pyannote/speaker-diarization pre-trained model.')




