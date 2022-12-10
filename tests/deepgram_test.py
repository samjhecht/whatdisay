


from dotenv import load_dotenv
import os
import logging
from deepgram import Deepgram
import asyncio
import json

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.debug("Debug mode enabled.")


async def deepgram_test():

    load_dotenv()
    deepgram_api_key = os.environ.get('DEEPGRAM_API_KEY')

    # Initialize the Deepgram SDK
    deepgram = Deepgram(deepgram_api_key)

    audio_source = f'/Users/sam/code/whisper/prodigy-env/audio/three_speakers_1.wav'

    with open(audio_source,'rb') as audio:
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
    print(output_json)
    j = json.loads(output_json)
    utterances = j['results']['utterances']
    with open('outputblabla.json','w',encoding='utf-8') as f:
        json.dump(response,f, ensure_ascii=False, indent=4)
        
    if utterances:
        for u in utterances:
            start = u['start']
            end = u['end']
            speaker = u['speaker']
            caption = u['transcript']
            confidence = u['confidence']
            print(f'{start} - {end} | {speaker} | {caption}')

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

            confidence = u['confidence']
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

    # if utterances:
    #     previous_speaker = -1
    #     current_caption = ""
        
    #     for u in utterances:
    #         speaker  = u['speaker']
    #         start = u['start']
    #         end = u['end']

    #         confidence = u['confidence']
    #         caption = u['transcript']                

    #         if speaker == previous_speaker:
    #             current_caption += "  " + caption
    #         else:
    #             if previous_speaker != -1:
    #                 segments.append([start, end, previous_speaker, current_caption])
    #             current_caption = caption
    #             previous_speaker = speaker

    #     # add the last row
    #     segments.append([start, end, previous_speaker, current_caption])
    print('-----')
    for s in segments:

        print(f'{s[0]} - {s[1]} | {s[2]} | {s[3]}')



# asyncio.run(maind())