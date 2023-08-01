from transformers import pipeline
import os
import cv2
import speech_recognition as sr
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer

# TODO make video-loader that split all videos into frames, to pass into image-loader and get captions from

# order: use Video_loader to load video and break into frames, then use Image_loader to load images and get back captions.

class Video_loader:
    def __init__(self, folder_name, frame_frequency = 5) -> None:
        self.folder_name = folder_name
        self.frame_frequency = frame_frequency
        # for each video, make a folder with the name of the video, and fill that with frames

    def make_frames(self) -> None:
        for filename in os.listdir(self.folder_name):
            video_file = os.path.join(self.folder_name, filename)
            video_frames_folder_name = filename + "_images"
            video_frames_folder_path = os.path.join(self.folder_name, video_frames_folder_name)

            # TODO check that folder is created

            # split into frames and save them
            cam = cv2.VideoCapture(video_file)
            frameno = 0
            while(True):
                ret, frame = cam.read()
                if ret:
                    name = str(frameno) + ".jpg"
                    cv2.imwrite(name, frame)
                    frameno = frameno + self.frame_frequency
                else:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
    
    def start_image_loading(self) -> dict:
        list_subfolders_with_paths = [f.path for f in os.scandir(self.folder_name) if f.is_dir()]
        
        captions_dict : dict = dict()
        for subfolder in list_subfolders_with_paths:
            image_loader = Image_loader(subfolder)
            # how will I save captions from each video?
            # make a dict!
            captions_dict[subfolder] = image_loader.get_captions()
        
        return captions_dict



class Image_loader:
    def __init__(self, folder_name) -> None:
        self.folder_name = folder_name
        self.captioner : Captioner = Captioner()
        self.captions : list = []

    def get_captions(self) -> list:
        for filename in os.listdir(self.folder_name):
            f = os.path.join(self.folder_name, filename)
            self.captions.append(self.captioner.get_caption(f))
        return self.captions

class Captioner:
    def __init__(self) -> None:
        self.captioner = pipeline("image-to-text",model="Salesforce/blip-image-captioning-base")

    def get_caption(self, image_path) -> str:
        self.captioner(image_path)

class Audio_transcriber:
    def __init__(self, video_file : str = None) -> None:
        self.video_file = video_file


    # TODO fix file reading issues
    def transcribe_video(video_name: str) -> str:
        video = AudioSegment.from_file("video.mp4", format="mp4")
        audio = video.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        # TODO fix the path
        audio.export("audio.wav", format="wav")

        # get transcriptions going
        model = Model(r"vosk-model-small-en-us-0.15")
        recogniser = KaldiRecognizer(model, 16000)
        # read audio data into "data" TODO
        data = None
        if recogniser.AcceptWaveform(data):
            text = recogniser.Result()
            return text
        else:
            return None



