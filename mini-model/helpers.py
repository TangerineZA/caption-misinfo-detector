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
    def __init__(self, vide_folder : str = None) -> None:
        self.video_folder = vide_folder

    def transcribe_video_folder(self, folder_name):
        filenames = [f for f in os.scandir(folder_name) if os.path.isfile(os.path.join(folder_name, f))]
        audio_captions_dict : dict = dict()
        for filename in filenames:
            text = self.transcribe_video(filename)
            audio_captions_dict[filename] = text

    # TODO fix file reading issues
    def transcribe_video(video_file: str) -> str:
        video = AudioSegment.from_file(video_file, format="mp4")
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



