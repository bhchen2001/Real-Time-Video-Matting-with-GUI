import cv2
import torch
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QTimer

from main_window_ui.video_matting_window import Ui_MainWindow
from camera import Camera
import numpy as np
import math

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.view_data.setScaledContents(True)
        self.view_data2.setScaledContents(True)
        self.device = "cuda"

        """
        Camera-related variables
        """
        self.view_x = self.view.horizontalScrollBar()
        self.view_y = self.view.verticalScrollBar()
        self.view.installEventFilter(self)
        self.last_move_x = 0
        self.last_move_y = 0

        self.view_x2 = self.view2.horizontalScrollBar()
        self.view_y2 = self.view2.verticalScrollBar()
        self.view2.installEventFilter(self)
        self.last_move_x2 = 0
        self.last_move_y2 = 0

        """
        Camera-related functions
        """
        self.ProcessCam = Camera('mobilenetv3', "../pretrained_model/rvm_mobilenetv3.pth", self.device)
        if self.ProcessCam.connect:
            """
            Link the signal to the slot function
                - show the original image and the matting image
            """
            self.ProcessCam.ori_data.connect(self.getRaw)
            self.ProcessCam.matting_data.connect(self.getRaw2)
            self.cam_start.setEnabled(True)
        else:
            self.cam_start.setEnabled(False)
        
        """
        Other functions of camera
        """
        self.cam_stop.setEnabled(False)
        self.viewRoi.setEnabled(False)
        self.viewRoi2.setEnabled(False)
        
        """
        Basic button functions
        """
        self.cam_start.clicked.connect(self.openCam)
        self.cam_stop.clicked.connect(self.stopCam)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.browser_button.clicked.connect(self.replace_background)
        self.bg_select.currentIndexChanged.connect(self.change_BG)


        """
        Record-related functions
        """
        self.ProcessCam.record_flag = False
        self.record_start.clicked.connect(self.set_record_flag_start)
        self.record_stop.clicked.connect(self.set_record_flag_stop)
        self.record_timer = QTimer()
        self.record_counter = 0

        self.record_timer.timeout.connect(self.update_record_counter)
        
        """
        Model-select functions
        """
        self.model_select.currentIndexChanged.connect(lambda index: self.ProcessCam.change_model(self.model_select.currentText()))

        """
        Input video functions
        """
        self.input_select.activated.connect(self.input_type_change)

    def change_BG(self):
        if self.bg_select.currentIndex() == 0:
            self.browser_button.setEnabled(True)
        elif self.bg_select.currentIndex() == 1:
            self.browser_button.setEnabled(False)
            self.browser_path.setText("The other Camera")
            self.ProcessCam.background_img = None
            if self.ProcessCam.cam2.isOpened():
                self.ProcessCam.background_cap = self.ProcessCam.cam2

    def getRaw(self, data):
        """
        Get the original image
        """
        self.showData(data)

    def getRaw2(self, data):
        """
        Get the matting image
        """
        self.showData2(data)

    def openCam(self):
        """
        Activate the camera recording
        """
        if self.ProcessCam.connect:
            self.ProcessCam.open()
            self.ProcessCam.start()
            
            self.cam_start.setEnabled(False)
            self.cam_stop.setEnabled(True)
            self.viewRoi.setEnabled(True)
            self.viewRoi2.setEnabled(True)
    def stopCam(self):
        """
        Freeze the camera image
        """
        if self.ProcessCam.connect:
            self.ProcessCam.stop()
            
            self.cam_start.setEnabled(True)
            self.cam_stop.setEnabled(False)
            self.viewRoi.setEnabled(False)
            self.viewRoi2.setEnabled(False)

    def showData(self, img):
        """
        Show the original image
        """
        self.Ny, self.Nx, _ = img.shape

        img_data = img.tobytes()
        qimg = QtGui.QImage(img_data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        self.view_data.setScaledContents(True)
        self.view_data.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if self.viewRoi.currentIndex() == 0: roi_rate = 0.5
        elif self.viewRoi.currentIndex() == 1: roi_rate = 0.75
        elif self.viewRoi.currentIndex() == 2: roi_rate = 1
        elif self.viewRoi.currentIndex() == 3: roi_rate = 1.25
        elif self.viewRoi.currentIndex() == 4: roi_rate = 1.5
        else: pass
        self.scrollAreaWidgetContents.setMinimumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.scrollAreaWidgetContents.setMaximumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.view_data.setMinimumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.view_data.setMaximumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        

    def showData2(self, img):
        """
        Show the matting image
        """
        if self.viewRoi2.currentIndex() == 0:
            """
            No effect
            """
            img = img
        elif self.viewRoi2.currentIndex() == 1:
            """
            Retro effect
            """
            B = 0.272 * img[:, :, 2] + 0.534 * img[:, :, 1] + 0.131 * img[:, :, 0]
            G = 0.349 * img[:, :, 2] + 0.686 * img[:, :, 1] + 0.168 * img[:, :, 0]
            R = 0.393 * img[:, :, 2] + 0.769 * img[:, :, 1] + 0.189 * img[:, :, 0]

            B = np.clip(B, 0, 255)
            G = np.clip(G, 0, 255)
            R = np.clip(R, 0, 255)

            img = np.stack((R, G, B), axis=-1).astype(np.uint8)

        elif self.viewRoi2.currentIndex() == 2:
            """
            Sketch effect
            """
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, (5,5), 0)
            canny = cv2.Canny(gaussian, 25, 100)
            ret, result = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)
            img = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        elif self.viewRoi2.currentIndex() == 3:
            """
            Animation effect
            """
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            smooth = cv2.bilateralFilter(img, 9, 300, 300)
            edges = cv2.Canny(gray, 30, 100)
            cartoon = cv2.bitwise_and(smooth, smooth, mask=edges)
            cartoon = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
            cartoon[:, :, 1] = np.clip(cartoon[:, :, 1] * 1.5, 0, 255)
            img = cv2.cvtColor(cartoon, cv2.COLOR_HSV2BGR)

        elif self.viewRoi2.currentIndex() == 4:
            """
            Watercolor effect
            """
            scale_percent = 75
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            small_image = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
            image_blur = cv2.GaussianBlur(small_image, (15, 15), 0)

            image_smooth = cv2.bilateralFilter(image_blur, d=5, sigmaColor=75, sigmaSpace=75)

            gray = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.adaptiveThreshold(cv2.medianBlur(gray, 5), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
            watercolor = cv2.bitwise_and(image_smooth, edges_colored)
            
            watercolor = cv2.cvtColor(watercolor, cv2.COLOR_BGR2HSV)
            watercolor[:, :, 1] = np.clip(watercolor[:, :, 1] * 1.5, 0, 255)
            watercolor = cv2.resize(watercolor, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            img= cv2.cvtColor(watercolor, cv2.COLOR_HSV2BGR)


        self.Ny, self.Nx, _ = img.shape
        img_data = img.tobytes()
        qimg = QtGui.QImage(img_data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        self.view_data2.setScaledContents(True)
        self.view_data2.setPixmap(QtGui.QPixmap.fromImage(qimg))

        if self.viewRoi.currentIndex() == 0: roi_rate = 0.5
        elif self.viewRoi.currentIndex() == 1: roi_rate = 0.75
        elif self.viewRoi.currentIndex() == 2: roi_rate = 1
        elif self.viewRoi.currentIndex() == 3: roi_rate = 1.25
        elif self.viewRoi.currentIndex() == 4: roi_rate = 1.5
        else: pass
        self.scrollAreaWidgetContents_3.setMinimumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.scrollAreaWidgetContents_3.setMaximumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.view_data2.setMinimumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))
        self.view_data2.setMaximumSize(int(self.Nx*roi_rate), int(self.Ny*roi_rate))

        

    def eventFilter(self, source, event):
        """
        Unknown event filter
        """
        if source == self.view:
            if event.type() == QtCore.QEvent.MouseMove:  # 定義滑鼠點擊移動動作
                # 找到滑鼠移動位置
                if self.last_move_x == 0 or self.last_move_y == 0:
                    self.last_move_x = event.pos().x()
                    self.last_move_y = event.pos().y()
                # 計算滑鼠移動量
                distance_x = self.last_move_x - event.pos().x()
                distance_y = self.last_move_y - event.pos().y()
                # 設置 view 的視窗移動
                self.view_x.setValue(self.view_x.value() + distance_x)
                self.view_y.setValue(self.view_y.value() + distance_y)
                # 儲存滑鼠最後移動的位置
                self.last_move_x = event.pos().x()
                self.last_move_y = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:  # 定義滑鼠放開動作
                # 滑鼠放開過後，最後位置重置
                self.last_move_x = 0
                self.last_move_y = 0
            return QtWidgets.QWidget.eventFilter(self, source, event)
        elif source == self.view2:
            if event.type() == QtCore.QEvent.MouseMove:  # 定義滑鼠點擊移動動作
                # 找到滑鼠移動位置
                if self.last_move_x2 == 0 or self.last_move_y2 == 0:
                    self.last_move_x2 = event.pos().x()
                    self.last_move_y2 = event.pos().y()
                # 計算滑鼠移動量
                distance_x = self.last_move_x2 - event.pos().x()
                distance_y = self.last_move_y2 - event.pos().y()
                # 設置 view 的視窗移動
                self.view_x2.setValue(self.view_x2.value() + distance_x)
                self.view_y2.setValue(self.view_y2.value() + distance_y)
                # 儲存滑鼠最後移動的位置
                self.last_move_x2 = event.pos().x()
                self.last_move_y2 = event.pos().y()
            elif event.type() == QtCore.QEvent.MouseButtonRelease:  # 定義滑鼠放開動作
                # 滑鼠放開過後，最後位置重置
                self.last_move_x2 = 0
                self.last_move_y2 = 0
            return QtWidgets.QWidget.eventFilter(self, source, event)

    def closeEvent(self, event):
        """
        Closing window event
        """
        if self.ProcessCam.running:
            self.ProcessCam.close()
            time.sleep(1)
            self.ProcessCam.terminate()
        QtWidgets.QApplication.closeAllWindows()

    def keyPressEvent(self, event):
        """
        Closing window by keyboard event
        """
        if event.key() == QtCore.Qt.Key_Q: 
            if self.ProcessCam.running:
                self.ProcessCam.close()
                time.sleep(1)
                self.ProcessCam.terminate()
            QtWidgets.QApplication.closeAllWindows()

    def replace_background(self):
        background_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '../content/image', 'Image Files (*.jpg *.jpeg *.png *.mp4)')[0]
        if not background_path:
            return
        self.browser_path.setText(background_path)

        if background_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            self.ProcessCam.background_img = cv2.imread(background_path)
            if self.ProcessCam.background_img is None:
                raise IOError("Cannot open background image")
            self.ProcessCam.background_img = cv2.cvtColor(self.ProcessCam.background_img, cv2.COLOR_BGR2RGB)
            self.ProcessCam.background_img = cv2.resize(self.ProcessCam.background_img, (self.ProcessCam.frame_width,self.ProcessCam.frame_height))
            self.ProcessCam.background_cap = None
        else:
            self.ProcessCam.background_img = None
            self.ProcessCam.background_cap = cv2.VideoCapture(background_path)
            if not self.ProcessCam.background_cap.isOpened():
                raise IOError("Cannot open background video")
            
    def set_record_flag_start(self):
        """
        set record flag when clicking the button
        """
        if not self.ProcessCam.record_flag:
            self.ProcessCam.record_flag = True
            self.ProcessCam.video_writter = cv2.VideoWriter('../output/output.avi', cv2.VideoWriter_fourcc(*'mp4v'), 20, (self.ProcessCam.frame_width,self.ProcessCam.frame_height))

            self.record_timer.start(1000)

    def set_record_flag_stop(self):
        """
        set record flag when clicking the button
        """
        if self.ProcessCam.record_flag:
            self.ProcessCam.record_flag = False
            self.ProcessCam.video_writter.release()
            time.sleep(1)
            self.ProcessCam.video_writter = None

            # Stop the QTimer
            self.record_timer.stop()
            
            # Reset the counter
            self.record_counter = 0
            self.record_time.setText("00:00:00")

    def update_record_counter(self):
        self.record_counter += 1
        self.record_time.setText("{:02d}:{:02d}:{:02d}".format(self.record_counter//3600, self.record_counter//60, self.record_counter%60))

    def input_type_change(self):
        """
        input type change
        """
        self.ProcessCam.input_type = self.input_select.currentIndex()
        if self.ProcessCam.input_type == 0:
            self.ProcessCam.video_input = None
        elif self.ProcessCam.input_type == 1:
            video_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '../content/video', 'Video Files (*.mp4 *.avi *.mov)')[0]
            if not video_path:
                return
            self.ProcessCam.video_input = cv2.VideoCapture(video_path)
            self.Process.video_fps = self.ProcessCam.video_input.get(cv2.CAP_PROP_FPS)
            if not self.ProcessCam.video_input.isOpened():
                raise IOError("Cannot open video file")