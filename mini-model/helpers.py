from transformers import pipeline
import os
import cv2
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer
import wave
import json
import math
from pydub.silence import split_on_silence
import speech_recognition as sr
import numpy as np
import tensorflow as tf
import gensim

# TODO make video-loader that split all videos into frames, to pass into image-loader and get captions from

# order: use Video_loader to load video and break into frames, then use Image_loader to load images and get back captions.

class Video_loader:
    def __init__(self, folder_name, frame_frequency = 5) -> None:
        print("Init video loader")
        self.folder_name = folder_name
        self.frame_frequency = frame_frequency
        # for each video, make a folder with the name of the video, and fill that with frames

    def process(self):
        self.make_frames()
        return self.start_image_loading()

    def make_frames(self) -> None:
        print("Make frames")
        for filename in os.listdir(self.folder_name):
            video_file = os.path.join(self.folder_name, filename)
            video_frames_folder_name = filename + "_images"
            video_frames_folder_path = os.path.join(self.folder_name, video_frames_folder_name)
            try:
                os.mkdir(video_frames_folder_path)
            except OSError as e:
                print(e)
            print("Storing in " + video_frames_folder_path)

            # split into frames and save them
            cam = cv2.VideoCapture(video_file)
            frameno = 0
            while(True):
                ret, frame = cam.read()
                if ret:
                    name = str(frameno) + ".jpg"
                    # cv2.imwrite(name, frame)
                    filepath_and_name = os.path.join(video_frames_folder_path, name)
                    # print("Filepath and name: " + filepath_and_name)
                    if not os.path.isdir(video_frames_folder_path):
                        print("No such a directory: {}".format(video_frames_folder_path))
                        exit(1)
                    if frameno % self.frame_frequency == 0:
                        cv2.imwrite(filepath_and_name, frame)
                    frameno = frameno + 1
                else:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
    
    def start_image_loading(self) -> dict:
        print("Start image loading")
        list_subfolders_with_paths = [f.path for f in os.scandir(self.folder_name) if f.is_dir()]
        
        captions_dict : dict = dict()
        for subfolder in list_subfolders_with_paths:
            # print("Captioning subfolder: " + subfolder)
            image_loader = Image_loader(subfolder)
            # how will I save captions from each video?
            # make a dict!
            caption = image_loader.get_captions()
            captions_dict[subfolder] = caption
            # print(caption)
        
        return captions_dict



class Image_loader:
    def __init__(self, folder_name) -> None:
        print("Init image loader")
        self.folder_name = folder_name
        self.captioner : Captioner = Captioner()
        self.captions : list = []

    def get_captions(self) -> list:
        print("Start image get captions")
        for filename in os.listdir(self.folder_name):
            f = os.path.join(self.folder_name, filename)
            self.captions.append(self.captioner.get_caption(f))
        return self.captions

class Captioner:
    def __init__(self) -> None:
        print("Init captioner")
        self.captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")

    def get_caption(self, image_path) -> str:
        # print("Get caption")
        caption = self.captioner(image_path)
        # print(caption)
        return caption

class Audio_transcriber:
    def __init__(self, video_folder : str = None) -> None:
        self.video_folder = video_folder

    def transcribe_video_folder(self, folder_name):
        print("Transcribing video folder: " + folder_name)
        filenames = [f for f in os.scandir(folder_name) if os.path.isfile(f)]
        print("Filenames: ")
        print(filenames)
        audio_captions_dict : dict = dict()

        print("Make folders and save audio")
        for filename in filenames:
            video_file = filename
            audio_file_folder_name = filename.name + "_audio"
            audio_folder_path = os.path.join(folder_name, audio_file_folder_name)
            try:
                os.mkdir(audio_folder_path)
            except OSError as e:
                print(e)
            print("Storing in " + audio_folder_path)

            audio : AudioSegment = AudioSegment.from_file(video_file, format="mp4")
            print("Transcribing video " + str(video_file))
            audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

            filepath_and_name = os.path.join(audio_folder_path, "audio.wav")
            audio.export(filepath_and_name, format="wav")
            
            text = self.transcribe_using_cloud(filepath_and_name, audio_folder_path)
            audio_captions_dict[filename] = text



    def transcribe_video(self, audio_file: str, output_folder : str) -> str:
        wf = wave.open(audio_file, 'r')

        # get transcriptions going
        model = Model(r"vosk-model-en-us-0.22")
        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetWords(True)

        textResults = []
        results = ""

        print("Model and recogniser initialised...")

        while True:
            data = wf.readframes(4096)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                recognizerResult = recognizer.Result()
                print(recognizerResult)
                results = results + recognizerResult
                # convert the recognizerResult string into a dictionary  
                resultDict = json.loads(recognizerResult)
                # save the 'text' value from the dictionary into a list
                textResults.append(resultDict.get("text", ""))

        return results
    
    def transcribe_using_cloud(self, audio_file, output_folder):
        r = sr.Recognizer()
        sound = AudioSegment.from_file(audio_file)
        # split audio
        major_chunks = self.split_seconds(sound, 50)
        print("Audio split!")

        chunks = []
        for major_chunk in major_chunks:
            for sub_chunk in split_on_silence(major_chunk,
            # experiment with this value for your target audio file
            min_silence_len = 500,
            # adjust this per requirement
            silence_thresh = major_chunk.dBFS-14,
            # keep the silence for 1 second, adjustable as well
            keep_silence=500,):
                chunks.append(sub_chunk)

        # for chunk in chunks:
            # print(type(chunk))

        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
            print("Made folder " + output_folder)
        whole_text = ""
        for i, audio_chunk in enumerate(chunks, start=1):
            # print(type(audio_chunk))
            # export audio chunk and save it in
            # the `folder_name` directory.
            chunk_filename = os.path.join(output_folder, f"chunk{i}.wav")
            audio_chunk.export(chunk_filename, format="wav")
            # print("Exported " + chunk_filename + " to " + output_folder)
            # recognize the chunk
            with sr.AudioFile(chunk_filename) as source:
                    audio_listened = r.record(source)
                    # try converting it to text
                    try:
                        # print("Doing recognition!")
                        text = r.recognize_google(audio_listened)
                        # print(text)
                    except sr.UnknownValueError as e:
                        print("Error:", str(e))
                    else:
                        text = f"{text.capitalize()}. "
                        print(chunk_filename, ":", text)
                        whole_text += text
                        pass
        # return the text for all chunks detected
        return whole_text
    
    def split_seconds(self, audiofile: AudioSegment, max_seconds):
        max_milliseconds = max_seconds * 1000
        num_segments = math.ceil(len(audiofile) / max_milliseconds)
        chunks = []

        current_ms = 0
        for i in range (0, num_segments):
            chunk = audiofile[current_ms:current_ms+max_milliseconds]
            chunks.append(chunk)
            current_ms = current_ms + max_milliseconds

        return chunks
    

class fusion_helper:
    def __init__(self) -> None:
        self.embedding_dimension = 300
        self.filepath = 'glove.42B.300d.txt'

    def get_dicts_text(self, dict1 : dict, dict2 : dict) -> list:
        combined_values : list = []
        combined_values.append(dict1.values(), dict2.values())
        return combined_values
    
    def get_embeddings_from_wordlist(self, wordlist : list) -> list:
        word_dictionary : dict = dict.fromkeys(wordlist)
        vocab_size = len(word_dictionary) + 1
        embedding_matrix_vocab = np.zeros((
            vocab_size, self.embedding_dimension
        ))

        with open(self.filepath, encoding='utf8') as f:
            for line in f:
                word, *vector = line.split()
                if word in word_dictionary:
                    idx = word_dictionary[word]
                    embedding_matrix_vocab[idx] = np.array(
                        vector, dtype=np.float32
                        )[:self.embedding_dimension]
        return embedding_matrix_vocab
                    