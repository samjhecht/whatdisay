import pyaudio
import wave
from pydub import AudioSegment
from pathlib import Path
from whatdisay.utils import TaskProps, millisec



def convertAudio(m4a_file):

    # n = Path(m4a_file).stem
    # track = AudioSegment.from_file(audio_file,  format= 'm4a')
    # file_handle = track.export(wav_filename, format='wav')
    return str('Make this function work.')

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


def truncateAudio(audio_file, t1: int, t2: int, file_names: TaskProps):

    newAudio = AudioSegment.from_wav(audio_file)
    a = newAudio[t1:t2]
    trunc_filename = file_names.task_dir + file_names.event_name + '_trunc.wav'
    a.export(trunc_filename, format="wav") 

    print('Saved truncated version of audio file at location: {}'.format(trunc_filename))

