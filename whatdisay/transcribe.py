#!/usr/bin/env python3

from whatdisay.utils import TaskProps, millisec
from whatdisay.config import Config
from pydub import AudioSegment
from whatdisay.diarize import Diarize
from deepgram import Deepgram
import aiofiles
import asyncio
import os
import json
import re
import webvtt
import whisper
from whisper.utils import write_txt,write_vtt


def generateWhisperTranscript(wav_file, tp: TaskProps, model="large", custom_name=""):
    """
    Uses OpenAI Whisper to generate a transcription from an audio file.

    Parameters
    ----------
    wav_file: str
        The path to the audio file that you want a transcription of.

    tp: TaskProps
        An instantiated utils.TaskProps class that provides all the necessary directory names.
    
    model: str
        The whisper model instance (tiny, base, small, medium, large).   Will default to 'medium' if not set.

    custom_name: str
        Optional input to pass a desired filename prefix for the resulting whisper transcription files.  Needed for the diarization functions.
    """
    if not type(tp) == TaskProps:
        raise ValueError('Parameter tp must be of type TaskProps.')

    print(f'Beginning Whisper transcription from {wav_file}')
    model = whisper.load_model(model)
    result = model.transcribe(wav_file)

    if custom_name:
        whisper_filename = str(custom_name)
    else: 
        whisper_filename = tp.task_name

    # TODO: do i actually need to use whisper's file writer for text? 

    # save TXT
    with open(os.path.join(tp.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.txt'), "w", encoding="utf-8") as txt:
        write_txt(result["segments"], file=txt)
    print('Saved TXT file of whisper transcription at: {}'.format(os.path.join(
        tp.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.txt')))

    # save VTT
    with open(os.path.join(tp.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.vtt'), "w", encoding="utf-8") as vtt:
        write_vtt(result["segments"], file=vtt)
    print('Saved VTT file of whisper transcription at: {}'.format(os.path.join(
        tp.whisper_transcriptions_dir, str(whisper_filename) + '_whisper.vtt')))



def diarizedTranscriptPyannote(wav_file, tp: TaskProps):
    """
    Run the whole shebang. Use Pyannote to create diarization and OpenAI Whisper to generate a transcription from an audio file.
    Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wav_file: str
        The path to the audio file from which you would like a diarized transcription.

    tp: TaskProps
        An instantiated utils.TaskProps class that provides all the necessary directory names.
    
    """
    whisper_model = Config().get_param('WHISPER_MODEL')
    dz = Diarize(tp,3).diarize_pyannote(wav_file)
    groups = dz[0]
    gidx = dz[1]
    
    # Make sure there's a directory to save the audio segment files in
    tp.createTaskDir(tp.whisper_transcriptions_dir)

    for i in range(gidx+1):
        segment_audio_filename = tp.dia_segments_dir, str(gidx) + '.wav'
        generateWhisperTranscript(segment_audio_filename, tp, whisper_model, i)
    
    gidx = -1

    final_output_file = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")

    with open(final_output_file, "w", encoding="utf-8") as text_file:
        for g in groups:
            shift = re.findall('[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=g[0])[0] # the start time in the original video
            shift = millisec(shift) - 2000
            shift = max(shift, 0)

            gidx += 1

            vtt_file = os.path.join(tp.tmp_file_dir, 'whisper_transcriptions/' + str(gidx) + '_whisper.vtt')
            captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)),  caption.text] for caption in webvtt.read(vtt_file)]

            if captions:
                speaker = g[0].split()[-1]

                for c in captions:
                    text_file.write(f'{speaker}: {c[2]}')
    
    print(f'Saved diarized transcript at location: {final_output_file}')


def getWhisperTxt(wav_file, model="large") -> str:

    print(f'Beginning Whisper transcription from {wav_file}')
    model = whisper.load_model(model)
    w = model.transcribe(wav_file)    
    transcript_txt: str = w["text"]
    
    return transcript_txt


async def diarizedTranscriptDeepgramWhisperLocal(
    wav_file,
    whisper_model: str,
    tp: TaskProps
    ):
    """
    Run the whole shebang. Use Deepgram to get the speaker diarization segments and then run OpenAI Whisper
    over each segment to generate a transcription from an audio file. Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wav_file: str
        The path to the audio file from which you would like a diarized transcription.

    tp: TaskProps
        An instantiated utils.TaskProps class that provides all the necessary directory names.

    """
    
    dz = await Diarize(tp).diarize_deepgram(wav_file)

    audio = AudioSegment.from_wav(wav_file)

    # Make sure there's a directory to save the audio segment files in
    tp.createTaskDir(tp.dia_segments_dir)

    idx = 0
    for segment in dz:
        start = float(segment[0]) * 1000
        end = float(segment[1]) * 1000

        output_af_name = os.path.join(tp.dia_segments_dir + str(idx) + '.wav')
        audio[start:end].export(output_af_name, format='wav')
        idx += 1
        
    final_output_file = os.path.join(tp.diarized_transcriptions_dir, tp.task_name + ".txt")

    with open(final_output_file, "w", encoding="utf-8") as text_file:

        for i in range(len(dz)):
            segment_audio = os.path.join(tp.dia_segments_dir, str(i) + '.wav')
            speaker = 'Speaker_' + str(dz[i][2])
            w = getWhisperTxt(segment_audio, whisper_model)

            if w:
                text_file.write(f'{speaker}: {w}\n')
                print(f'{speaker}: {w}')

    print(f'Saved diarized transcript at location: {final_output_file}')

async def getWhisperTxtDeepgram(wav_file) -> str:

    deepgram_api_key = Config().get_param('DEEPGRAM_API_KEY')

    # Initialize the Deepgram SDK
    deepgram = Deepgram(deepgram_api_key)

    # with open(wav_file,'rb') as audio:
        # source = {'buffer': audio, 'mimetype': 'audio/wav'}

    async with aiofiles.open(wav_file, mode='rb') as audio:
        # audio_data = await f.read()
        source = {'buffer': audio, 'mimetype': 'audio/wav'}

        response = await asyncio.create_task(
            deepgram.transcription.prerecorded(
                source,
                {
                    'punctuate': True, 
                    'tier': 'enhanced', 
                    'model': 'whisper'}
            )
        )

    # output_json = json.dumps(response)
    j = json.loads(response)
    transcript = j["results"]["channels"][0]["alternatives"][0]["transcript"]

    return transcript


async def diarizedTranscriptAllDeepgram(
    wav_file, 
    tp: TaskProps
    ):
    """
    Run the whole shebang. Use Deepgram to get the speaker diarization segments and then leverage Deepgram's API torun OpenAI Whisper
    over each segment to generate a transcription from an audio file instead of doing the whisper transcription locally.
    Then combine the results to create diarized transcript. 

    Parameters
    ----------
    wav_file: str
        The path to the audio file from which you would like a diarized transcription.

    tp: TaskProps
        An instantiated utils.TaskProps class that provides all the necessary directory names.

    """
    dz = await Diarize(tp).diarize_deepgram(wav_file)
    
    audio = AudioSegment.from_wav(wav_file)

    # Make sure there's a directory to save the audio segment files in
    tp.createTaskDir(tp.dia_segments_dir)

    print('Creating audio segments based on the diarization...')
    idx = 0
    for segment in dz:
        start = float(segment[0]) * 1000
        end = float(segment[1]) * 1000

        output_af_name = os.path.join(tp.dia_segments_dir + str(idx) + '.wav')
        audio[start:end].export(output_af_name, format='wav')
        idx += 1

    deepgram_api_key = Config().get_param('DEEPGRAM_API_KEY')
    
    # Initialize the Deepgram SDK
    deepgram = Deepgram(deepgram_api_key)

    async def gather_with_concurrency_limit(n, *coros):
        semaphore = asyncio.Semaphore(n)

        async def sem_coro(coro):
            async with semaphore:
                return await coro
        return await asyncio.gather(*(sem_coro(c) for c in coros))

    async def get_transcript(i, s, af):
        print(f'Starting task: {i}')
        try:
            async with aiofiles.open(af, mode='rb') as f:
                audio = await f.read()
                source = {'buffer': audio, 'mimetype': 'audio/wav'}
                response = await asyncio.create_task(
                    deepgram.transcription.prerecorded(
                        source,
                        {
                            'punctuate': True, 
                            'tier': 'enhanced', 
                            'model': 'whisper'}
                    )
                )
                output_json = json.dumps(response)
                j = json.loads(output_json)
                transcript = j["results"]["channels"][0]["alternatives"][0]["transcript"]
                speaker = 'Speaker_' + s

                if transcript:
                    result = f'{speaker}: {transcript}'
                    print(result)
                else:
                    result = ""
        except Exception as e:
            print('Error while sending: ', + str(e))
            raise

        return result

    coroutines = []
    for i in range(len(dz)):
        speaker = str(dz[i][2])
        af = f'{tp.dia_segments_dir}/{str(i)}.wav'
        coroutines.append(get_transcript(i,speaker,af))

    print('Getting whisper transcripts from Deepgram...')
    result_list = await gather_with_concurrency_limit(10, *coroutines)
    
    final_output_file = f'{tp.diarized_transcriptions_dir}/{tp.task_name}.txt'

    async with aiofiles.open(final_output_file, "w", encoding="utf-8") as text_file:
        for r in result_list:
            if r:
                await text_file.write(f'{r}\n')
        
    print(f'Saved diarized transcript at location: {final_output_file}')
