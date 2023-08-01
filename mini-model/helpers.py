from transformers import pipeline
import os
import cv2

# TODO make video-loader that split all videos into frames, to pass into image-loader and get captions from

# order: use Video_loader to load video and break into frames, then use Image_loader to load images and get back captions.

class Video_loader:
    def __init__(self, folder_name) -> None:
        self.folder_name = folder_name
        # for each video, make a folder with the name of the video, and fill that with frames

    def make_frames(self, frame_frequency : int) -> None:
        for filename in os.listdir(self.folder_name):
            video_file = os.path.join(self.folder_name, filename)
            video_frames_folder_name = filename + "_images"
            video_frames_folder_folder = os.path.join(self.folder_name, video_frames_folder_name)

            # TODO check that folder is created

            # split into frames and save them
            cam = cv2.VideoCapture(video_file)
            frameno = 0
            while(True):
                ret, frame = cam.read()
                if ret:
                    name = str(frameno) + ".jpg"
                    cv2.imwrite(name, frame)
                    frameno = frameno + frame_frequency
                else:
                    break
            
            cam.release()
            cv2.destroyAllWindows()


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