import torch
import cv2
import os
import time
import numpy as np

from PyQt5 import QtCore
from torchvision import transforms
from torch.utils.data import DataLoader

import sys
sys.path.append('..')
from model import MattingNetwork
from inference_utils import VideoReader, ImageSequenceReader

class Camera(QtCore.QThread):
    ori_data = QtCore.pyqtSignal(np.ndarray)
    matting_data = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, variant: str, checkpoint: str, device: str ,parent=None, background_source='../content/video/video_test.mp4'):
        super().__init__(parent)
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device

        self.cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        
        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.cam is None or not self.cam.isOpened():
            self.connect = False
            self.running = False
        else:
            self.connect = True
            self.running = False
        self.frame_counter = 0

        self.background_img = None
        self.background_cap = None

        """
        record-related variables
        """
        self.record_flag = False
        self.video_writter = None
        
        """
        model info
        """
        self.model_dict = {
            "mobilenetv3": "rvm_mobilenetv3.pth",
            "resnet50": "rvm_resnet50.pth",
        }

        """
        input type
            - 0: webcam
            - 1: video
        """
        self.input_type = 0
        self.video_input = None

        if os.path.isfile(background_source):
            if background_source.lower().endswith(('.mp4', '.avi', '.mov')):
                self.background_cap = cv2.VideoCapture(background_source)
                if not self.background_cap.isOpened():
                    raise IOError("Cannot open background video")
            else:
                self.background_img = cv2.imread(background_source)
                if self.background_img is None:
                    raise IOError("Cannot open background image")
                self.background_img = cv2.cvtColor(self.background_img, cv2.COLOR_BGR2RGB)
                self.background_img = cv2.resize(self.background_img, (self.frame_width,self.frame_height))
                background_tensor = torch.tensor(self.background_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255).to(device)
        else:
            raise ValueError("Invalid background source")


    def convert(self, *args, **kwargs):
        self.convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)

    def convert_video(self, model, device, dtype, downsample_ratio = 0.25):

        transform = transforms.ToTensor()

        # Initialize reader
        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        if not self.cam.isOpened():
            raise IOError("Cannot open webcam")

        model = model.eval()
        if device is None or dtype is None:
            param = next(model.parameters())
            dtype = param.dtype
            device = param.device

        try:
            with torch.no_grad():
                    rec = [None] * 4
                    downsample_ratio = 0.25
                    while self.running and self.connect:
                        if self.input_type == 0:
                            ret, frame = self.cam.read()
                            if not ret:
                                break
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            self.ori_data.emit(frame)
                            frame = transform(frame).unsqueeze(0).to(device, dtype)

                            fgr, pha, *rec = model(frame.cuda(), *rec, downsample_ratio)

                            if self.background_img is not None:
                                background_tensor = torch.tensor(self.background_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255).to(device)
                            elif self.background_cap is not None:
                                ret, bg_frame = self.background_cap.read()
                                if not ret:
                                    self.background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    ret, bg_frame = self.background_cap.read()
                                bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
                                bg_frame = cv2.resize(bg_frame, (self.frame_width, self.frame_height))
                                background_tensor = torch.tensor(bg_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255).to(device)

                            if self.record_flag:
                                """
                                record frontground video
                                """
                                green_background = torch.tensor([0, 1, 0], dtype=torch.float32).view(1, 3, 1, 1).to(device)
                                com_green = fgr * pha + green_background * (1 - pha)
                                com_green = com_green.squeeze().permute(1, 2, 0).cpu().numpy()
                                com_green = cv2.cvtColor(com_green, cv2.COLOR_RGB2BGR)
                                self.video_writter.write((com_green * 255).astype('uint8'))

                            com = fgr * pha + background_tensor * (1 - pha)
                            com = com.squeeze().permute(1, 2, 0).cpu().numpy()
                            com = (com * 255).astype('uint8')
                            self.matting_data.emit(com)
                        elif self.input_type == 1 and self.video_input is not None:
                            ret, frame = self.video_input.read()
                            if not ret:
                                self.video_input.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, frame = self.video_input.read()
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                            self.ori_data.emit(frame)
                            frame = transform(frame).unsqueeze(0).to(device, dtype)

                            fgr, pha, *rec = model(frame.cuda(), *rec, downsample_ratio)

                            if self.background_img is not None:
                                background_tensor = torch.tensor(self.background_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255).to(device)
                            elif self.background_cap is not None:
                                ret, bg_frame = self.background_cap.read()
                                if not ret:
                                    self.background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                    ret, bg_frame = self.background_cap.read()
                                bg_frame = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB)
                                bg_frame = cv2.resize(bg_frame, (self.frame_width, self.frame_height))
                                background_tensor = torch.tensor(bg_frame, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).div(255).to(device)

                            if self.record_flag:
                                """
                                record frontground video
                                """
                                green_background = torch.tensor([0, 1, 0], dtype=torch.float32).view(1, 3, 1, 1).to(device)
                                com_green = fgr * pha + green_background * (1 - pha)
                                com_green = com_green.squeeze().permute(1, 2, 0).cpu().numpy()
                                com_green = cv2.cvtColor(com_green, cv2.COLOR_RGB2BGR)
                                self.video_writter.write((com_green * 255).astype('uint8'))

                            com = fgr * pha + background_tensor * (1 - pha)
                            com = com.squeeze().permute(1, 2, 0).cpu().numpy()
                            com = (com * 255).astype('uint8')
                            self.matting_data.emit(com)

        finally:
            pass
    def run(self):
        self.convert_video(model = self.model, device=self.device, dtype=torch.float32)

    def open(self):
        if self.connect:
            self.running = True

    def stop(self):
        if self.connect:
            self.running = False

    def close(self):
        if self.connect:
            self.running = False
            time.sleep(1)

    def change_model(self, model_name):
        """
        change model depend on model_index
        """
        checkpoint = "../pretrained_model/" + self.model_dict[model_name]
        self.model = MattingNetwork(model_name).eval().to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)