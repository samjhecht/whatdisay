import pyaudio
import wave
import sys
import os
import re
from pydub import AudioSegment

from whatdisay.utils import BinPaths, millisec
from whatdisay.diarize import Diarize
import os



def recordAudio(wf):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    RECORD_SECONDS = 5

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Starting recording...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(wf, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


    # recordAudio(WAVE_OUTPUT_FILENAME)
    # print('Recording saved at: {}'.format(WAVE_OUTPUT_FILENAME))


def truncateAudio(t1,t2,audio_file):

    newAudio = AudioSegment.from_wav(audio_file)
    a = newAudio[t1:t2]
    trunc_filename = str(os.path.splitext(audio_file)) + '_trunc'
    a.export(trunc_filename, format="wav") 
    
    return trunc_filename + '.wav'

