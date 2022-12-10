#!/usr/bin/env python3


from whatdisay.transcribe import diarizedTranscriptDeepgram
from whatdisay.utils import TaskProps
from whatdisay.diarize import Diarize
from pydub import AudioSegment
import whisper
import time
import logging
import os
import json
from deepgram import Deepgram
import webvtt
import asyncio
from dotenv import load_dotenv, find_dotenv

# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# logging.debug("Debug mode enabled.")

# event_name = 'test_event'
# ts_task = 'wombat'
# # ts_task = str(int(time.time() * 1000))

# tp = TaskProps(event_name,ts_task)

# wav_file = f'/Users/sam/code/whisper/prodigy-env/audio/three_speakers_1.wav'


# diarizedTranscriptDeepgram(wav_file,tp)


#########################################
######### Syncronous Whispering #########
#########################################

# dz = asyncio.run(Diarize(tp).diarize_deepgram(wav_file))

# # # Make sure there's a directory to save the audio segment files in
# tp.createTaskDir(tp.whisper_transcriptions_dir)

# audio = AudioSegment.from_wav(wav_file)

# # Make sure there's a directory to save the audio segment files in
# tp.createTaskDir(tp.dia_segments_dir)
# tp.createTaskDir(tp.whisper_transcriptions_dir)

# idx = 0
# for segment in dz:
#     start = float(segment[0]) * 1000
#     end = float(segment[1]) * 1000

#     output_af_name = os.path.join(tp.dia_segments_dir + str(idx) + '.wav')
#     audio[start:end].export(output_af_name, format='wav')
#     idx += 1
    

# final_output_file = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")

# async def getWhisperTxt(wav_file, model="medium"):

#     print(f'Beginning Whisper transcription from {wav_file}')
#     model = whisper.load_model(model)
#     response = await asyncio.create_task(
#         model.transcribe(wav_file)    
#     )
#     transcript_txt = response["text"]
    
#     return transcript_txt


# with open(final_output_file, "w", encoding="utf-8") as text_file:

#     for i in range(len(dz)):
#         segment_audio = os.path.join(tp.dia_segments_dir, str(i) + '.wav')
#         speaker = 'Speaker_' + str(dz[i][2])
#         w = getWhisperTxt(segment_audio,"large")

#         text_file.write(f'{speaker}: {w}\n')
#         print(f'{speaker}: {w}')


# text_file.close()

#########################################
######### Asyncronous Whispering ########
#########################################

# async def getWhisperTxt(wav_file, model="medium"):

#     print(f'Beginning Whisper transcription from {wav_file}')
#     model = whisper.load_model(model)
#     response = await asyncio.create_task(
#         model.transcribe(wav_file)    
#     )
#     transcript_txt = response["text"]
    
#     return transcript_txt

# async def getWhispers(dz):
    
#     coroutines = []

#     for i in range(len(dz)):
#         segment_audio = os.path.join(tp.dia_segments_dir, str(i) + '.wav')
        
#         l = asyncio.create_task(getWhisperTxt(segment_audio,"large"))
#         coroutines.append(l)
    
#     whispers = await asyncio.gather(*coroutines)

#     return whispers


# with open(final_output_file, "w", encoding="utf-8") as text_file:

#     whispers = asyncio.run(getWhispers(dz))

#     for i in range(len(dz)):
        
#         speaker = 'Speaker_' + str(dz[i][2])
#         w = whispers[i]

#         text_file.write(f'{speaker}: {w}\n')
#         print(f'{speaker}: {w}')


# text_file.close()


# async def getWhisperTxt(wav_file, model="medium"):

#     print(f'Beginning Whisper transcription from {wav_file}')
#     model = whisper.load_model(model)
#     response = await asyncio.create_task(
#         model.transcribe(wav_file)    
#     )
#     output_json = json.dumps(response)
#     j = json.loads(output_json)
#     transcript_txt: str = j["text"]
    
#     return transcript_txt