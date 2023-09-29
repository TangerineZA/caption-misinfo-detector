import moviepy.editor as mp
import speech_recognition as sr
import keras_ocr
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# BiGRU imports
import time
import pandas as pd
import tensorflow as tf
import itertools as it
from sklearn.model_selection import train_test_split

from pydub import AudioSegment

SKIP_FRAMES = 4
TIME_STEP = 0.5     # seconds between captured frames
AUDIO_DELTA = 1000  # milliseconds around frame capture to capture audio
SEED = 26032013

np.random.seed()

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
        self.video_ocr_captions : list = []
        self.audio_segments : list = []
        self.audio_segment_times_seconds : list = [] # to keep track of which frame each audio segment is related to
        self.audio_captions : list = []
        self.combined_captions : list = [] # combined captions from video OCR and audio recognition
        self.combined_embeddings : list = [] # contains 2d array of GloVe embeddings - each successive entry is for an individual frame, so [i][j] will have the embedding of word j from frame i
    

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
        r = sr.Recognizer()
        for segment in self.audio_segments:
            with sr.AudioSource(segment) as source:
                audio_text = r.record(source)
            text = r.recognize_google(audio_text)
            self.audio_captions.append(text)

    def extract_video_captions(self) -> list:
        for current_time_seconds in self.audio_segment_times_seconds:
            next_frame = self.video.get_frame(current_time_seconds)

        # do OCR TODO FIX
        pipeline = keras_ocr.pipeline.Pipeline()
        predictions = pipeline.recognize(self.video_frames_segemented)
        self.video_ocr_captions.append(predictions)


    def encode_captions(self):
        # let's put both caption forms into one list for now, following the paper instructions
        combined_caption_list : list = []
        for i in len(self.video_ocr_captions):
            audio = self.audio_captions[i]
            visual = self.video_ocr_captions[i]
            combined_entry : list = []
            combined_entry.append(visual, audio)
            combined_caption_list.append(combined_entry)


        # First transform each word in the word sequence to a 
        # fixed-length vector representation with the pre-trained
        # GloVe embedding
        embedding_dict : dict = load_glove_dict()

        embeddings_2d_array : list = []
        for caption_list in self.combined_captions:
            frame_embedding = get_vectors_for_frame_caption(embedding_dict, caption_list)
        self.combined_embeddings = embeddings_2d_array

        

        # Further develop a bidirectional gated recurrent unit (BiGRU)
        # encoder to encode the semantic information in each word 
        # embedding from both directions of the word sequence

        # TEMP SKIP #TODO if want exact replica of Shang et al.
        

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
    

    def __init__(self, encoded_data : list) -> None:
        self.encoded_data_np = np.array(encoded_data)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(activation="relu"),
            tf.keras.layers.Dense(activation="relu"),
            tf.keras.layers.Softmax()
        ])

        # TODO - should go from_logits, but softmax layer already part of model
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        optimiser = tf.keras.optimizers.Adam(0.001)

        model.compile(
            optimizer=optimiser,
            loss=loss
        )
    
    def train(test_data, train_data):
        pass



def load_glove_dict() -> dict :
    i = 0
    embeddings_dict = {}
    with open("glove.42B.300d.txt", 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector

            i = i + 1
            if 1 % 10 == 0:
                print("Loaded line " + i + " from embeddings")
    return embeddings_dict

def get_vectors_for_frame_caption(embeddings_dict : dict, frame_captions_list : list) -> list:
    embeddings : list = []
    for word in frame_captions_list:
        try:
            vector_representation = embeddings_dict[word]
            embeddings.append(vector_representation)
        except:
            pass
    return embeddings