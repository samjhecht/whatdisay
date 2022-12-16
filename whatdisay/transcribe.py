#!/usr/bin/env python3

from whatdisay.utils import TaskProps, millisec
from whatdisay.config import Config
from pydub import AudioSegment
from whatdisay.diarize import Diarize
from deepgram import Deepgram
import aiofiles
import concurrent.futures
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

async def gather_with_concurrency_limit(n, *coros):
    semaphore = asyncio.Semaphore(n)

    async def sem_coro(coro):
        async with semaphore:
            return await coro
    return await asyncio.gather(*(sem_coro(c) for c in coros))

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
    # dz = await Diarize(tp).diarize_deepgram(wav_file)
    dz = [[3.2382812, 11.9609375, 0, "Yeah. Actually,  that'll probably be better if we use yours because I think mine doesn't quite pick it up in the same volume."], [12.375, 13.359375, 1, 'Mhmm. Okay.'], [13.6953125, 23.5625, 0, "Anyway,  yeah, work works in a we're finally announcing my departure  on Monday to the broader company."], [24.21875, 28.75, 2, "So it didn't the the negotiations  you decided not to take what they were offering?"], [29.140625, 244.875, 0, "Well,  it's funny, gosh.  So I agreed with our CTO to, like,  put some time into working up, hashing out, some more of the details of what a role might look like. And I haven't done that yet partly because, like,  last week was Thanksgiving, and this week, I'm, like,  finding it hard to make room for that while I'm in this awkward in between phase where, like, they kept delaying, like, letting my team know, but we'd let some of them know,  but I'm we haven't like, they're finalizing today, like,  permit like, the  offer for one of my top lieutenants to take my role. So I haven't been able to transition stuff. So I still am in this awkward place where I have, like, my workload and  Oh. I was kinda hoping to my plan was,  like, get this done  and let go of, like,  the role, and then I can spend some time  thinking about, you know, knowledge transfer, but also  mapping that out.  But  I don't know. Over probably the last three weeks, I've  feel more like  I gotta get out of there.  Like,  it's been  kind of burnt out inducing even to I actually branded to our my HR business partner last week. Like,  I can't even quit without it being burned out in DC. Because they're just, like,  holding the messaging hostage  and, like, being hard to communicate with  and they're busy, but, like,  it's just like, oh my gosh.  I'm like, I was working really hard to try to help them pull off, like, minimizing the impact of my news and, like, having a good narrative.  And but I like, they have to meet me in the middle on that. Like, I can't just go, like, solve that alone,  and  and they were  I've got to focus, like, you know what I give up on trying to make this  go as well as it could go because  and I'll just  come to terms with however  the  messaging unfolds and whatever the impact is that, like, I can't I can't  solve that problem if they're not gonna be in any middle. Anyway,  I yesterday,  my current boss who hadn't who hadn't talked to in, like, two and a half weeks  calls me and and  was like, oh, we're gonna move forward with  the offer for Alex who's, like, my amaire leader who will be promoted.  But Erika, who's who's, like, top boss,  she called me about it and said, yeah. Let's move forward. But, like, more importantly, what what about Sam? Have you managed to,  like, make sure we we're keeping him, and we can miss him to stay yet? And I guess, that's even more important, and he's telling me that. And he's like, so how about this? I told her, like,  you'll just you can just report to me directly  with, like,  know, we'll just  figure out what your responsibilities  are, like, once you, like, figure out what you wanna work on and, like,  it it was on I came away from that call being, like, wow. He's basically offering me, like, you still want your high paying  Jot salary,  but you can just do nothing until you decide what you wanna do. Like, blank check on a roll. It's almost, like, kind of a turn off on this.  Why?"], [249.25, 249.75, 1, 'Well,'], [251.375, 469.75, 0, "I wouldn't be happy in that type of dynamic anyway. Like, I'm not capable of Well, it wouldn't be indefinite. Right? It it would be an indefinite? No. But it's either It's kind of a turn off is, like, they they shouldn't be offering that, frankly. Like,  we've  got tight budgets with the economy getting tougher and, like,  I don't know. It it's also it's also, like, kind of lazy on their part of, like,  Erica, who's the president of field operations, you know, I  I went to the CEO in July and said, hey.  We gotta put a time line on me not being in this role. We can explore other things, but I can't do this one anymore.  And I don't think they, like, totally took me serious or whatever, and then  weren't meeting me on the middle in the middle on, like, figuring that out. Proactively. And so I was in September, I was just like, alright, folks. I'm letting HR know that my last day will be November, and then they freaked out and called me. Erica calls me, and it's like, oh, know, first of all, keep please take through the end of the quarter. I say yes, and she's like, and you can take time off. You can either take a sabbatical and keep getting paid while you do it and take take a R and R for a few months and come back, or you can you can quit and leave.  But, like, at least let us, like,  do this homework, and I wrote, like, a five page doc for whatever. I sent you the the first draft for that. Mhmm. Mhmm. And  like, tell us what you wanna be when you grow up and all that stuff, and give us a chance to surprise you with something that convinced you you'd wanna stay. And then I do that in, like, sent it to her first, ended up sending the CEO to. The CTO is the only one who, like,  actually can tell he read it, and he came back with, like,  Really, I was I've kind of impressed with  how little I've gotten to know him because he's newer.  And how  like,  he just had very like,  he clearly read it and had really thoughtful ideas  that made a lot of sense and that aligned to what I was talking about. And  And so, like, Ray, who's my current boss, is the new guy. He's five weeks in. He's  like,  totally  overwhelmed with all the stuff he's trying to come up to speed on. He's got a million things to do, and it's our biggest quarter of the year. It's got a big number to hit, and just all these other things. And  it's a bit of a turn off that, like, Erica asked me to go through this process with her, and then she's like,  She might have read the doc. I don't know. But either way, she was like, alright. Let me delegate this to the brand new guy who doesn't know Sam or have any context on this or have any idea of what it doesn't know the business in the organization well enough to, like, really be in a position to make your own So it gives you it undermines your confidence that things won't go any differently if you stay.  Yeah. I mean, the at the  I'm I'm probably over indexing on on  like, I I I'm not even really actually that that's kinda like an offhanded comment because I'm not really spending much time thinking about this particular variable because, like, I definitely would not wanna stay in this or, like, this side of the organization. If I were to stay, it would be like the engineering side. Like, this side of anything is a mess and nothing, it definitely won't get better.  I mean,  yeah, probably"], [471.0, 509.75, 2, "Okay. Let me let me just so we have two things.  One from a while ago on the table, which is the relationship end of things.  And  the other is this  activation from this meeting.  So just give yourself a mean, maybe there's something else that's happened in between. I don't know. That feels, you know,  emotionally alive.  What's the question? The question is,  what do you wanna focus on?  What what feels most important to you right now? Mhmm."], [510.5, 537.75, 0, "And alive. I think it might be good, Dick. And I don't wanna lose the other conversation  indefinitely, but given my timing of, like, you  know, hitting a threshold point of, you know, messaging going out next week, and then that's gonna then, you know, we're one month away from the end date. It's  probably  worth  spending some time on -- Sure. Sure. -- checking back in on on the work stuff and Mhmm."], [540.5, 545.5, 1, "Does that sound good? That's fine. I just wanna be -- No."], [546.0, 549.75, 2, '-- intentional.  Together.  Makes sense. Yeah.'], [557.5, 573.5, 0, "Another thing that's probably worth I mean, that I've I've, like, journaled a bunch on over the last two weeks.  Well, I told you last Monday when we talked at the very beginning about how, like, that Saturday and Sunday had been, like, amazing."], [574.0, 574.5, 1, 'And'], [576.0, 662.0, 0, "Wednesday through Sunday the following week or, like, the rest of that week was  literally the same playbook every day. And I was, like, head moments of, like, oh my gosh. This  seems like a really clear signal that the plan to like, this is what I need to do for some amount of time because this is, like, basically wake up in the morning, work on  stuff  on my computer that I'm passionate about whether that's coding or  learning how to  produce music with logic or whatever, like, that type of productive stuff and and and songwriting stuff, then, like, vigorous exercise in the afternoon, and then, like, painting in the evening  just really, like  felt like I was glow in last week.  Yeah. The other thing that I  Feel  like I  like I I guess thinking about I had I've kind of I've put a pin and thinking about, like,  talking to them about what role a role might look like and stuff.  And I'm about to, like, now get back to needing to think about that. I just kinda, like, kick that can a little bit down the road for a few weeks there.  But I don't know if"], [663.25, 663.75, 1, 'I'], [665.0, 669.5, 0, "I don't I guess if I think about it, I I can't imagine it being possible for me to make"], [671.0, 671.5, 1, 'to'], [674.0, 689.5, 0, "there's it seems very unlikely that any role would come up that I would be able to decide on  before  taking, like, space on whether that was something I wanted to do. Oh, I see. Like"], [690.5, 694.5, 2, 'No worries. Do you need to not feel committed in order to decide if you wanna commit'], [696.0, 702.0, 0, 'and have this space to not be doing it, to give you that Yeah. I just I I guess maybe I just feel like I need'], [705.92523, 706.42523, 1, 'Maybe'], [708.0, 721.75, 0, "I I two things. Like, one, I need to find out if, like, the burnout factors, like, that feeling I was having in this meeting this morning of, like, oh my god. I'm just done meeting on meetings like this with Jay freaking out and yelling at somebody."], [722.68524, 723.18524, 1, 'I'], [724.5, 788.0, 0, "I need to see if that can be possibly recharge on a reasonable time line if I just disengage from it for a few months. And then two, I'm assuming there'll be  a bunch of realizations and and, like,  you know, given three months of not being on these front lines, like, I I don't know what I'll I'll, like, get clarity on. You know, it might be that  I  decide I'd really need to lean into  continuing to pursue some of these creativity related things for a while longer, or I might decide something might happen that gives me total clarity that I need to try to go become an astronaut or whatever. You know, like, something totally different than this. And  Maybe that's  maybe that  to a certain extent, getting my hopes up on getting clarity at some point, but"], [788.5, 807.75, 2, "Can I ask you -- Yeah. -- have you  in any major, like, decision  Have you had that it sounds like a clarity  maybe means, like, a vision that has no ambivalence?  Attached to it? Yeah. I've had two"], [813.5, 898.5, 0, "points.  I can remember where I had, like, oh, man. I don't need to worry. I can this is just the right thing, and I'm all in on it.  And  it was it's just this magical feeling of, like And what was it about? The first one was when I was in high school, I  wanted to go to the United States Military Academy in West Point, and I spent a whole year. I ended up not getting in at the very I got two Congressional Nominations  and ran a five thirty mile and ace the grades and all that, but  two inhalers in my medical record for childhood asthma to the age of thirteen kept me out in the end. So it ended up also being, like, this kinda soul crushing disappointment.  But I remember in the run up to that, I was like, wanna go do this so bad. I went to UVA  and,  ultimately, but I which was awesome. I loved UVA, but it was like a Well, I guess I'll go to this in state school that'll be cheap for my parents and that I clearly, you know, it's easy to get into.  But not with I wasn't, like, super pumped about that. I've been coming off this, like, one of these other thing. But the other one was I I took a job after college, not really knowing what I wanted to do, but, like, a friend's brother got me an interview, and I got the job.  And I I literally I was in investment consulting  and"], [900.5, 902.21027, 1, 'which  not'], [903.0, 977.5, 0, "my jam, but it's, like, advising institutional investors on how to  allocate their money.  And I read an article  well, I was already starting to, like,  get  it just get passionate about, like, the yeah. Like, some programming stuff and learning learning more about that on my own. And  and data related stuff trying to go do more in terms of analytics that we could do in that role. And then I read an article in The Economist about big data  And I just had this clarity moment of, like,  I don't care that I was an econ Spanish major and didn't do CS.  I need to get commit go work at one of these companies at this, like, new  in this big data space and a start up that will let me  at least spend part of my time learning to to be a software developer,  and I need to move to San Francisco, and that's gonna I I'm gonna make that happen. I had this, like, ridiculous clarity.  And  then I was able to just, like, pour my energy into  pulling that off and, like, made that transition happen in, like, three months."], [978.0, 979.5, 2, 'Uh-huh. Wow. Okay.'], [980.0, 1052.5, 0, "So I don't know if I'll  I don't know that I'll have that level of clarity,  but I just think it's really hard when I'm in the weeds  at  this job, like,  to get perspective.  And once I get perspective,  I suspect that  it may be that I after getting  perspective that  and and I think another thing that will help  inform it is, like,  having some other conversations with potential other options.  To get some data points on what that might look like, and what options might be out there, and and what alternatives could be like. I'm just so in this, like I see. How about world, and it's hard to, like, look. Over the you know, and and get perspective.  And I may decide  in three or four months after leaving that  Yeah. You know what? I I I am really  excited. I do really wanna go back and recommit and do that role reporting to the  CTO. But  there's just I can't imagine a scenario in the next month where we get me to that level of"], [1053.0, 1060.75, 2, "conviction Mhmm. Okay. Just wanna get a sense of what you're picturing  as, you know, what  you need."], [1061.2803, 1063.6401, 1, 'Yeah.  And'], [1065.0, 1066.5, 0, "that's  probably"], [1067.0, 1067.5, 1, 'helpful'], [1068.0, 1070.75, 0, "to keep articulating more because I'm not"], [1071.1453, 1073.5, 1, 'To  a'], [1074.0, 1074.75, 0, 'certain extent,'], [1078.1603, 1078.6603, 1, 'I'], [1079.0, 1138.5, 0, "know I need I feel like really I I was sort of having those clarity moments,  not  Guys  waffling a lit I waffle a little bit. Oh, it's not quite as much conviction as the big data  thing, but last week, like,  when doing that routine, I was talking about like, I I was feeling like, oh my  gosh.  I don't know what would come after or where it will go, but I need to spend three months  giving myself the space to do this every day. So I I feel like I have conviction on that. But in terms of what I need after that,  I am kind of  I haven't really articulated that from or or  form that up in my head. So it is helpful probably  to push me to try  to unravel that  in this session and and others going forward."], [1138.875, 1141.4153, 2, 'Can  I ask you Is'], [1141.9576, 1142.4576, 1, 'there'], [1143.0, 1157.5, 2, "any  downside  to  you  I don't know, committing yourself to giving yourself three months of  this, of what you're of, you know, what has been lighting you up. I think I'm decided."], [1158.0, 1161.5, 0, "I'm I'm, like, I'm ready to to just say that's"], [1162.0, 1168.5, 2, 'happening. Okay. So you that feels how is it for me to ask you that and for you to say that?  Releaving,'], [1169.0, 1185.5, 0, "like,  tension goes away, you know, I'm like Okay. So I'm not getting any any reason not to do that. I mean, the the reasons not to would be like, last week, I went to a friend's giving, and one of my friends was, like,  well,"], [1186.0, 1186.75, 1, 'I mean,'], [1187.1453, 1200.5, 0, 'you know, once you leave a job and have been the longer you go, being unemployed,  the the more sure negotiating doors close and you you might have to take step back and and, like Is that true in'], [1201.0, 1206.8334, 1, "tech? I mean, it's  at your level? I don't"], [1208.0, 1234.75, 0, "I  if you went really long, it can become true because,  like,  you're just, like, getting out of date on the tech ecosystem and, like, what's relevant and because it just moves so fast. But for me and my skill set of, like, what I would do, I don't think so. And  it's  beside the point anyway because"], [1235.2953, 1239.5, 1, "I'm  like, you know, the other another"], [1240.0, 1267.5, 0, "naysay or reason why it could be bad to take three months or more is, like, woah, we're about to potentially go into a huge recession and the, you know, tech tech companies are doing layoffs, and there may not be Like, once again though, I don't think that like, at my level, they're  there will still be opportunities  that that that, you know, is different than, like, Amazon doing lay offs of, you know, people way lower done individual contributor"], [1268.0, 1269.7427, 1, 'roles. But that'], [1270.3334, 1281.5, 0, "that's also beside the point too because, like, I'm at a point where I'm willing like, if if the  economy totally crashes, and I and there let's say there are no and I decide  that,"], [1282.0, 1283.5, 1, 'you know, I think I,'], [1284.0, 1290.75, 0, "like, Well, let me try to go back and get a job, you know, on this track and and tech and, oh, no. It's not possible."], [1291.5702, 1293.2852, 1, "I'd  be"], [1294.0, 1328.0, 0, "okay, like you you know what? I gotta go do this other  quite  different thing making quite a bit less in salary for a while, but it's something that, like,  you know, wouldn't make sense on paper to do next  like, going and working for my brother on his company for a year just to for no for, like, take no cash, just equity, and my trying to help prop him up and get him further along, you know, with my skills and experience and or, you know, something like  that. I'm just at a point where"], [1329.0, 1331.5, 1, 'that  so'], [1332.0, 1334.5, 0, "no. To your to your point, there's no downside"], [1335.0, 1337.5, 2, "of this. Just Okay. That's that's yeah. That's what it's happened."], [1338.0, 1339.5, 1, 'Yeah.  Okay.'], [1340.0, 1340.75, 0, 'I think'], [1342.1503, 1342.6503, 1, 'I'], [1344.0, 1351.5, 0, 'guess, really the only question  standing  between now and the end of the year when that date comes around  is'], [1352.1353, 1354.5, 1, 'how do I wanna'], [1356.0, 1402.5, 0, 'comport myself with you know, how do I wanna leave  things  with you know, the the folks about me here.  Like,  like, are there things I might wanna do that to, like, keep a door open in a certain way? Like, how do I do I wanna, like, optimize for that? Or  And  I think I I think the answer is'], [1404.0, 1404.5, 1, 'I'], [1406.0, 1407.75, 0, 'do wanna go through with'], [1408.9204, 1409.4204, 1, 'doing'], [1410.0, 1432.5, 0, "what I talked about with that CTO, which is  spending a couple it's probably just a couple days work to refine it. The, like, mapping out, like, what what is this  problem that that he and I talked about high level that the business needs to solve? And what would a role look like in his work that would solve it and  frame  it as,"], [1433.0, 1433.5, 1, 'like,'], [1434.0, 1453.5, 0, "I already framed it as like, wow. I'll map that out and, you know,  whether I end up in that position or not, like, he'll come away with the work of mapping this out being done for him, which is a, you know, a benefit for him  either way.  And then I think I should just"], [1459.2305, 1462.2435, 1, 'Otherwise,  I  think'], [1463.0, 1495.5, 0, "I should shut down the conversations  probably with them and  just be clear so it doesn't string out anymore. What would that look like? Can sound like?  Like, for yesterday, that guy called me and was like, hey. You know, Erica's just dying to know, like, she delegated to me to make sure you're staying, and she's asking me for updates on whether I'm insured  you're staying. And how about I just, you know, blank check role and, like,  decide when you want what you want, like, you'll report to me and figure it out whenever you're ready to figure it out."], [1496.0, 1496.5, 1, 'You'], [1497.0, 1517.5, 0, "know, I didn't say  no. Like, tell Erica no, like, we're not gonna do that, and she can stop asking me for updates on it. Like, I that's what I mean by, like, Okay. And so picture doing that, and picture how that would feel and how how you imagine them reacting and how that would affect you. Just to roll"], [1518.0, 1518.6854, 2, 'the video,'], [1520.0, 1520.5, 0, 'Connect.'], [1525.7004, 1530.925, 1, 'Yeah.  I  mean,  like,'], [1532.0, 1572.75, 0, "part of me, I I I have, like, a little bit of an impulse to like, they're gonna probably come back and be like, oh, wait.  Like, thought you were gonna, like, consider  roles with us, and I have the simple stuff like, yeah. But you didn't fucking read the doc  and reply fly to me.  And so, frankly, like, if anything, this kinda hurt  the the odds of  me saying, But I'm also at a point where I'm, like, closer to the end where I'm, like, I don't even it's not really in my worth my while to  and to to  give  them that feedback."], [1582.8153, 1584.4077, 1, 'Yeah.  I'], [1585.0, 1596.0, 0, "guess I'm not gonna think through how exactly I'll frame it because I'm because  starting next week, I'm they're gonna come back around after we announce it,  like,  to checking in on it. And"], [1596.3334, 1598.5, 2, "What's happening in your body as you pictured"], [1599.0, 1599.5, 0, 'it?'], [1606.6202, 1608.81, 1, "I  don't"], [1610.0, 1630.5, 0, "know. It's it's hard it's hard to discern what is happening in my body from this versus, like, my body to side that we were gonna wake up at three thirty this morning and not be able to go back to sleep. So I've got, like, a a low sleep headache  Actually, that's been happening every day this week except usually for it. Can I ask you, are you waking up anxious"], [1631.0, 1632.5, 2, 'or  energized?'], [1633.0353, 1634.4987, 1, 'And I I'], [1635.0, 1641.5, 0, "basically,  last  week, it didn't happen. I was able to sleep until eight if I wanted"], [1642.0, 1643.2502, 1, 'to. This'], [1646.0, 1666.75, 0, "week started every day. It's been, like,  four AM on the dot Yeah. I'm just like, up.  Up. Yeah. Here we go. Let me look at the clock. I bet I know what time it's gonna be up four zero one. Here we go. And this is Adi. This morning this morning, my mind was racing  a bit on"], [1668.4154, 1675.5, 1, "What  it was a  can't  remember what the topics"], [1676.0, 1676.5, 0, 'were.'], [1676.9904, 1681.5, 2, 'Anxiety what was it? Emotion with it? Anxiety, interest, a little'], [1682.0, 1711.5, 0, "above. Today, there was, like, a little anxiety  mixed  in.  The other days this week, it's just it's just, like, purely like a  I'm awake.  I'm, like, I'm not thinking about anything in particular except that, like, I'm, you know, trying to get back to sleep, and it's not happening. Because I just  I used I in the past, I I had a long phase of this For  years,"], [1712.0, 1713.5, 1, 'there  would'], [1714.0, 1742.75, 0, "be  stents where I would go through a cycle where the four AM wake ups for everyday for months that were very tied to anxiety about work. It would be like a crushing.  Like, I'm awake. I'm  tired, but I'm so stressed out, and my chest is so tight. Because, like, am I gonna fail at my job? And in my career, am I letting everybody down. Like and this is not this is different. It's, like, it's similar it's very similar  in the"], [1743.7654, 1745.8827, 1, 'in  one'], [1747.0, 1775.5, 0, "sense that it's the  it's  clearly, like, tied to  the same, like, cortisol,  like, some kind of, like, biological  process that literally is happening, cycling if this is accent,  you know, I I look at the clock. It's it's almost to the minute, same time every  day. But it's there's not that in  clear mapping to anxiety for why the wake ups are happening."], [1776.0, 1777.5, 1, 'Okay. Are are you on medication'], [1778.0, 1862.5, 0, "now? Yeah. I have I'm on Zoloft.  Okay.  Yeah. I don't know what to make of it this  week. I also had, like, earlier this year, I had  a I got COVID, and I had a I don't remember if I told you this, actually. I had a really rough  few months  in from, like,  April through I got better for a few weeks and then, like, long COVID, basically. I know I get diagnosed with that, and I had, like, really bad mental fog, and I my sleep was just  destroyed for, like, four months.  Like, I it it was  it was way worse even. It was like, I would wake  up once or twice an hour, and I'd be able to go back to sleep  sometimes. It would be really consistent. It would be, like, Sometimes I would get two hours and wake up and go back to sleep for an hour and wake up, go back to sleep. Some days, it'd be, like, get three hours and then, like, I'm just gonna be up from two AM  on. So that really sucked.  And this  and that was similar wake ups  to these.  In that, like, when I would wake up, I'd be like, well, I'm not anxious, but I blame my awake and not gonna be able to go back to sleep. And that's kind of the feeling at four AM right now. Okay. So you're having now. It could be a transient thing just happening this week,"], [1863.0, 1865.9376, 1, 'but  I  really'], [1867.0, 1868.5, 0, "wish it didn't happen because"], [1870.0, 1878.75, 2, "Yeah.  Yeah.  It doesn't seem associated with a a decrease in the energy or motivation or confidence or anything like that."], [1880.0, 1880.5, 0, 'No.'], [1881.0, 1883.5, 1, 'Okay. No. Not at all right now.'], [1884.0, 1886.75, 2, 'Is it associated with any increase in those things?'], [1896.0, 1897.5, 1, 'No.  I'], [1898.0, 1901.75, 0, "don't I don't think so. Not that I know of."], [1902.1903, 1903.5, 2, "Okay. I wouldn't say"], [1904.0, 1905.5, 0, 'increasing  confidence.'], [1905.875, 1907.5, 1, 'And  energy and'], [1908.0, 1908.5, 2, 'motivation.'], [1910.1653, 1912.5826, 1, 'I  do'], [1914.0, 1919.5, 0, 'have a lot of energy right  now.  I would say. And is that unusual?'], [1920.0, 1922.5, 2, 'Or does it come and go? Or'], [1924.3752, 1940.0453, 0, "I should probably have sent you these things that are these journal entries I wrote last week. It was a lot of I  Last week, I actually ended up texting with one of my good friends. He's also a doctor, like, No."], [1941.0302, 1943.0151, 1, 'Actually,  what'], [1944.0, 1951.5, 0, 'happened was I this was, like, last Tuesday. I  went upstairs. Some  of my upstairs neighbor is a good friend and was telling'], [1952.0, 1952.75, 1, 'him, I'], [1954.6252, 1995.5, 0, "you know, I I told you about it last Monday when we met. Like, I have been having these, like, profound  like,  just delightful, like, mental feeling like this mental energy is there, and I'm, like, mentally more sharp and capable than I've been  potentially ever,  and  it's I was like, it even makes me wonder, like, in in  my  Am I going crazy? Is this what people who are losing their mind think that they're having profound, you know,  experiences that really aren't that profound?  And he was like, alright. Sounds like you might be a little bit  manic.  And"], [1996.25, 1997.5, 2, 'I was wondering about'], [1998.0, 2006.75, 0, 'that. Yeah. And then the next morning so, like, Wednesday morning or Thursday when I cover which day it was, I was like, I woke up and I was thinking about it. I was like, ugh.'], [2008.0, 2016.5, 2, "Yeah. I mean Let me just say those are not mutually exclusive, but you're a little manic and also  profoundly"], [2017.0, 2031.75, 0, "creative. They do. Yeah. Yeah. Yeah. And so I texted my friend and was, like, my doctor friend asking him, and he's an artist. Too. He, like, splits his time between  he he'll do night shifts for a week and then take a month off to just go all in on his art."], [2033.0, 2037.5, 1, "And  it's always, like, really encouraging me to"], [2038.0, 2074.6765, 0, "embrace the  up session with  Art.  But, yeah, he basically said the same thing. He was like, yeah. I mean, you might be a little manic, but  it sounds like it's it's good, Manik. And I would say go all in on embrace the mania and see what art comes out of it. But, yeah, there were points last week where I was like, wow. I'm, like,  amped  on on, like,  this does feel like an unnatural amount of  excitement and wondering, like, oh, is there"], [2075.195, 2080.5, 2, "a downswing Oh, that's that's the that's the thing to track. Right? That's the only I mean, if"], [2081.0, 2081.5, 1, 'if'], [2082.0, 2094.75, 2, "there's no downside, that's the problem. No problem.  And if you start to feel like you're,  you  know, you know, coming down in a way that feels"], [2096.0, 2114.5, 0, 'that. Yeah. I was wondering another thing I was thinking about last week is  the mental sharpness  feeling, felt similar to when my brother was in the hospital  for  those two months. I  like, the first month, I flew in and was, like,'], [2116.0, 2116.5, 1, 'in'], [2117.5498, 2220.5, 0, "charge of coordinating, like, his whole situation and my grandma died at the same time. I was using one phone to  organize that funeral and get my mom there and the other to deal with my brother, almost dying. What was his what was his happened with him again? What was his diagnosis? I don't remember what. I ultimately diagnosed him with adult onset stills disease, which is, like, an auto immune thing. It's it's, like,  Crohn's, it is your immune system fired up and attacking your gut, and your resistance is up. Stills is a similar process, but it attacks like your liver or your spleen, your heart, and your joints. And it can be triggered by a virus. I  I think it was probably triggered by COVID, and they just never had a test for that screen or whatever because he never had evidence for it. But if he had one single  test that indicated COVID in the weeks or months before the year to explain all this. But now he's, like, probably  it permanently damages joints. He came within a centimeter  dime and  he still injects himself once a day with  with  Anna Ken run.  But, anyway, like, at the beginning of that, I've just gotten  to Costa Rica  I actually got jumped and beat up in Costa Rica. And then, like, the next morning, my brother called him. He was like, they said, okay. Surn. I'm dying. Can you please come help? Get mom and daddy to save you by? Like, because the doctor mess not been told to me at terminal blood cancer prematurely.  Was ridiculous. But, anyway, so I, like, flew up there, and then I was just in this, like,"], [2222.0, 2226.5, 1, "adrenaline I found, like, one of the things that I'm thankful for is I was I was"], [2226.875, 2251.8333, 0, "born  with, like, the  Oh, the house is burning down. Here we go. Column collected gonna work the problem, and I  like, my  if a campus or whatever nooks my brain with adrenaline and I have extra mental powers. And I felt there's mental superpowers  during that process. Like, it was I I have crazy memories of"], [2252.4167, 2258.3765, 1, 'my mom and I just got back from the hospital, and  my mom'], [2258.7822, 2384.5, 0, "we're we're sitting down to eat this pizza we got, and my mom's phone starts ringing. On the table, and I looked to the left, and I see caller ID, unknown, Lynchburg, Virginia in my brain. Before my mom even reached for it, I was like,  grandma's dying. And I got up. I didn't even know she was in Lynchburg. I just knew she was, like, medical issues and that's near that it's an hour away from where She lived. And so before my mom even answers, I'm already walking over to my phone and starting to get ready to call my aunt Julie. And, like, was just coordinating  I I coordinated getting in touch  with the nurses and waking up my aunts and uncles so they could say goodbye and, like,  anyway, my at the time, my boss was like, dude, you guys take care of yourself. You're clearly jacked on in Berlin, and there's gonna be a crash. And I was like, wait. I feel great. And boy, was he right? Oh my gosh. Like, after three weeks of running like  that,  it was a epic fall. I was like, the mental fall guy, like, couldn't function every day for it took, like, it took, like, a month probably to  recover from that.  And  so there were, like, little similarities last week of, like, oh, man. I feel some of those same mental power. Use their commas also.  And then this is not a traumatic thing you're dealing with. Yeah. This is a very, very different in terms of the -- Yeah. -- stimulus  I I don't know. I haven't figured out what the stimulus of this cycle is. Like but it it just gave me pause, like, oh, I need to keep an eye on it because I really don't want that kinda  crash again. But at the same time, like, oh, no. I don't wanna admit that it could be a temporary thing because I love it so much. And I  wanna rationalize an excuse for how this is maybe just gonna be something that continues.  Mhmm. Okay. But I'm gonna I'm gonna have to count on you to let me know if I actually am losing my mind and and the"], [2386.0, 2408.5, 2, "mania is negative media. Well, I'm not getting that right now. And you will and if you start to get super irritable,  Or I mean, honestly, this isn't a  you know, I know what people with my level of education know, but it's not a specialty.  But, you know, it's  I don't think it'll be hard to track."], [2410.0, 2410.5, 0, 'Yeah.'], [2412.3948, 2412.806, 2, 'So'], [2413.217, 2414.5, 0, "I'll I'll keep an eye on it. Yeah."], [2414.8, 2418.4375, 1, 'But  it is  actually  interest'], [2418.875, 2434.5, 0, "I hadn't thought about that. Connection  between, like, not the early wake ups this week, which is clearly some kind of biological process  and the manic feelings last week.  And"], [2435.9297, 2446.5, 2, "I don't know what could be causing that though. Yeah. Well, you know, I mean, I don't I think a lot of times, it's never known  what caused it. Yeah."], [2447.0, 2447.5, 1, 'And'], [2448.0, 2450.75, 2, 'sometimes it is very clear. And'], [2453.4697, 2453.9697, 0, 'Yeah.'], [2456.0, 2475.5, 2, "So anyway  yeah. I think also just tracking micro  things about irritability if it starts to sort of shift into that. So you're tracking it on your own. That's great. Yeah.  How is it for me to ask you this? All  these"], [2476.0, 2498.863, 0, "questions? Well, I mean, I actually feel like I I through more at you without you asking all that many questions. But No. It's good.  It's good. I  I started with asking you about the wake ups and what you felt when you moved on. I'm I'm glad we dug in on that because it is something that, like, I've been kind of  Clearly,  you're"], [2499.1814, 2499.6814, 2, 'right.'], [2500.0, 2510.75, 0, "Exactly. Worried about, and I but I hadn't, like, been able to get it out there and make the connections and come up with a plan for how I'm gonna keep an eye on it and not  worry about it."], [2512.0, 2512.5, 2, 'Right.'], [2514.0, 2520.75, 0, "I know we're in wrap up time here. Yeah. In terms of taking with me,"], [2521.0793, 2523.5264, 1, 'I  think  I'], [2524.0, 2534.5, 0, "have a plan now of, like, how to approach  the conversations. Sounds good to know my bosses. Yeah.  So  I'll take that with me and go execute on"], [2536.0, 2538.5, 1, 'it. And I do think it would be helpful'], [2538.875, 2587.5, 0, "in  future sessions to  start unpacking a little bit. Like, kinda like we have been doing for,  like, relationships like, what I want I know what I wanna do for a two or three months, but I don't know I don't have  a framework for  really approaching what it comes after  that. So I I think digging in  on  that a bit in breaking it down. And I don't mean necessarily,  like,  trying to break down, like, what is the clarity thing I need to have clarity on  and do next, but, like, more of the framework for how I'm gonna  approach finding that.  If that makes sense.  It"], [2588.0, 2591.1667, 1, 'does.  So  the'], [2592.0, 2594.75, 2, 'process  versus  the goal? Yeah.'], [2595.1343, 2595.6343, 1, 'And'], [2596.5671, 2639.375, 0, "maybe,  like,  understanding high level requirements of the goal  Yes. Basically, like, this  just like we said, we're not we haven't said, like, well, let's sit down and search who the person you should marry and live with forever is  But we've said, like, high level, like, what are some things that we should look for there? And I think I need a similar approach when it comes to, like, figuring out what I should do next. It's not, like, how do I find a specific role, but, like Right. Exactly. I'm not sure I totally know, like,  what my criteria are for what would have been a successful use of my  next"], [2639.75, 2640.75, 1, 'eight years. Right.'], [2641.8193, 2652.5, 2, "So I have I have a plan on what I'm doing for the next two to three months. So I wanna dig into that in terms of how I'm going to a approach finding it, i. E. The process of finding it versus knowing"], [2653.0, 2655.375, 1, 'the  goal.  Is'], [2655.75, 2666.5, 0, "that accurate? Yep.  Think that's it. Okay. That's it. Anything else you wanna add? Those are some big ones, though. I feel like They are. Yeah. Yeah. That is."], [2668.0, 2670.5, 1, 'Okay. Alright. Have a have a great rest of your'], [2670.875, 2671.375, 0, 'Thursday,'], [2671.75, 2678.75, 2, "and you're great. Too. I just wanna let you know, I will be out for the week between Christmas and New Years, but here, otherwise."], [2679.5642, 2683.375, 0, 'Okay. Cool. Okay. Alright. You talk to you next  Thursday.'], [2683.75, 2684.75, 1, 'Okay. Bye bye.'], [3459.7776, 3460.2776, 0, 'Are'], [3464.0, 3467.5, 1, 'you recording?  Oh,  oh,'], [3468.0, 3470.6943, 2, 'jeez. It just said recording in progress.']]
    
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

    async def get_transcript(i, s, af):
        print(f'Starting task: {i}')
        try:
            # af = f'{tp.dia_segments_dir}/{str(i)}.wav'
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
    result_list = await gather_with_concurrency_limit(100,*coroutines)
    
    final_output_file = f'{tp.diarized_transcriptions_dir}/{tp.task_name}.txt'

    async with aiofiles.open(final_output_file, "w", encoding="utf-8") as text_file:
        for r in result_list:
            if r:
                await text_file.write(f'{r}\n')
        
    print(f'Saved diarized transcript at location: {final_output_file}')

