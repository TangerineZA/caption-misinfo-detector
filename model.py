import moviepy.editor as mp
import speech_recognition as sr
import keras_ocr

from pydub import AudioSegment

SKIP_FRAMES = 4
TIME_STEP = 0.5     # seconds between captured frames
AUDIO_DELTA = 1000  # milliseconds around frame capture to capture audio

class caption_guided_visual_representation_learning:

    # VARS:
    # input_video : video   -> the video upon which the detector is operating
    # frames : list         -> set of all video frames in input_video
    # time_delta : int      -> number of seconds used as delta with which to interpret words from audio
    # frame_captions : list<frame_caption>
    #                        -> set of all frame_caption objects generated through video analysis
    # frame_caption : list<words> 
    #                         -> a list of words found in the audio chunk pertaining to a frame k's timestamp
    # vector_word_sequence : list 
    #                         -> list of vector representations of word recognised in captions of frame k
    # recoded_word_sequence : list 
    #                         -> process vector_word_sequence with BiGRU and then apply maxpooling layer


    # PROCEDURE;
    # 1. Break audio into chunks
    # 2. Interpret words from chunks ->
    #       2.1. Inter

    def __init__(self, video_file : str = None, time_delta_ms = AUDIO_DELTA) -> None:
        self.video_file : str = video_file
        self.video : mp.VideoClip = mp.VideoFileClip(video_file)
        self.audio : AudioSegment = AudioSegment.from_file(video_file)
        self.time_delta_ms : int = time_delta_ms
        self.framerate : float = self.video.fps
        self.time_per_frame_ms : float = 1 / self.framerate
        self.total_frames : int = self.video.reader.nframes
        self.video_frames_segemented : list = [] # used to store sparsely collected video frames
        self.audio_segments : list = []
        self.audio_segment_times_seconds : list = [] # to keep track of which frame each audio segment is related to
        self.combined_captions : list = [] # combined captions from video OCR and audio recognition
    

    def current_position_in_audio(self, current_segment):
        current_ms = current_segment + TIME_STEP

    def extract_audio_captions(self) -> list:
        # In particular, for each frame sampled from the video,
        # we extract the words recognized in the audio chunk that
        # corresponds to the frame and the text superimposed on
        # the frame

        # SECTION: GET AUDIO SEGMENT FOR EACH FRAME
        current_time_seconds = 0
        while current_time_seconds > self.video.duration:
            next_segment = None

            start_time_seconds = current_time_seconds - (self.time_delta_ms / 1000)
            end_time_seconds = current_time_seconds + (self.time_delta_ms / 1000)
            start_time_ms = start_time_seconds * 1000
            end_time_ms = end_time_seconds * 1000

            if start_time_seconds < 0:
                pass
            if end_time_seconds > self.video.duration:
                pass

            next_segment = self.audio[start_time_ms:end_time_ms]
            self.audio_segments.append(next_segment)
            self.audio_segment_times_seconds.append(current_time_seconds)
            current_time_seconds = current_time_seconds + TIME_STEP

        # SECTION: GET SPEECH/MEANING FROM EACH AUDIO SEGMENT
        

        pass

    def extract_video_captions(self) -> list:
        for current_time_seconds in self.audio_segment_times_seconds:
            next_frame = self.video.get_frame(current_time_seconds)

        # do OCR

        # add to caption object


    def encode_captions():
        # First transform each word in the word sequence to a 
        # fixed-length vector representation with the pre-trained
        # GloVe embedding

        # Further develop a bidirectional gated recurrent unit (BiGRU)
        # encoder to encode the semantic information in each word 
        # embedding from both directions of the word sequence
        pass

    def get_visual_frame_representation():
        # caption-guided design starts with identifying a set of
        # main object regions in each video frame using the
        # advanced Fast R-CNN method

        # compute attention weight

        # frame visual representation fk of frame ck is generated 
        # as the weighted sum of the object region features using
        # the attention weight of ri
        pass

class acoustic_aware_speech_representation_learning:
    pass

class visual_speech_coattentive_information_fusion:
    pass

class supervised_misleading_video_detection:
    pass