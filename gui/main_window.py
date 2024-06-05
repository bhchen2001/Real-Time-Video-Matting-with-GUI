import cv2
import torch
import time

from PyQt5 import QtCore, QtGui, QtWidgets

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

        # view 屬性設置，為了能讓滑鼠進行操控，
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

        # 設定相機功能
        self.ProcessCam = Camera('mobilenetv3', "../pretrained_model/rvm_mobilenetv3.pth", self.device)  # 建立相機物件
        if self.ProcessCam.connect:
            # 連接影像訊號 (ori_data) 至 getRaw()
            self.ProcessCam.ori_data.connect(self.getRaw)  # 槽功能：取得並顯示影像
            self.ProcessCam.matting_data.connect(self.getRaw2)  # 槽功能：取得並顯示影像
            # 攝影機啟動按鈕的狀態：ON
            self.cam_start.setEnabled(True)
        else:
            # 攝影機啟動按鈕的狀態：OFF
            self.cam_start.setEnabled(False)
        # 攝影機的其他功能狀態：OFF
        self.cam_stop.setEnabled(False)
        self.viewRoi.setEnabled(False)
        self.viewRoi2.setEnabled(False)
        # 連接按鍵
        self.cam_start.clicked.connect(self.openCam)  # 槽功能：開啟攝影機
        self.cam_stop.clicked.connect(self.stopCam)  # 槽功能：暫停讀取影像
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.browser_button.clicked.connect(self.replace_background)
        self.bg_select.currentIndexChanged.connect(self.change_BG)


        """
        Record-related functions
        """
        self.ProcessCam.record_flag = False
        self.record_start.clicked.connect(self.set_record_flag_start)
        self.record_stop.clicked.connect(self.set_record_flag_stop)
        
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

    def getRaw(self, data):  # data 為接收到的影像
        """ 取得影像 """
        self.showData(data)  # 將影像傳入至 showData()

    def getRaw2(self, data):  # data 為接收到的影像
        """ 取得影像 """
        self.showData2(data)  # 將影像傳入至 showData()

    def openCam(self):
        """ 啟動攝影機的影像讀取 """
        if self.ProcessCam.connect:  # 判斷攝影機是否可用
            self.ProcessCam.open()   # 影像讀取功能開啟
            self.ProcessCam.start()  # 在子緒啟動影像讀取
            # 按鈕的狀態：啟動 OFF、暫停 ON、視窗大小 ON
            self.cam_start.setEnabled(False)
            self.cam_stop.setEnabled(True)
            self.viewRoi.setEnabled(True)
            self.viewRoi2.setEnabled(True)
    def stopCam(self):
        """ 凍結攝影機的影像 """
        if self.ProcessCam.connect:  # 判斷攝影機是否可用
            self.ProcessCam.stop()   # 影像讀取功能關閉
            # 按鈕的狀態：啟動 ON、暫停 OFF、視窗大小 OFF
            self.cam_start.setEnabled(True)
            self.cam_stop.setEnabled(False)
            self.viewRoi.setEnabled(False)
            self.viewRoi2.setEnabled(False)
    def showData(self, img):
        """ 顯示攝影機的影像 """
        self.Ny, self.Nx, _ = img.shape  # 取得影像尺寸

        # 建立 Qimage 物件 (RGB格式)
        img_data = img.tobytes()
        qimg = QtGui.QImage(img_data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        # view_data 的顯示設定
        self.view_data.setScaledContents(True)  # 尺度可變
        ### 將 Qimage 物件設置到 view_data 上
        self.view_data.setPixmap(QtGui.QPixmap.fromImage(qimg))
        ### 顯示大小設定
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
        """ 顯示攝影機的影像 """
        # 建立 Qimage 物件 (RGB格式)
        if self.viewRoi2.currentIndex() == 0:
            img = img
        elif self.viewRoi2.currentIndex() == 1:
            # 图像怀旧特效 (矢量化操作)
            B = 0.272 * img[:, :, 2] + 0.534 * img[:, :, 1] + 0.131 * img[:, :, 0]
            G = 0.349 * img[:, :, 2] + 0.686 * img[:, :, 1] + 0.168 * img[:, :, 0]
            R = 0.393 * img[:, :, 2] + 0.769 * img[:, :, 1] + 0.189 * img[:, :, 0]

            # 限制值的范围在0-255之间
            B = np.clip(B, 0, 255)
            G = np.clip(G, 0, 255)
            R = np.clip(R, 0, 255)

            dst = np.stack((R, G, B), axis=-1).astype(np.uint8)
            img = dst

        elif self.viewRoi2.currentIndex() == 2:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #高斯滤波降噪
            gaussian = cv2.GaussianBlur(gray, (5,5), 0)
            #Canny算子
            canny = cv2.Canny(gaussian, 25, 100)
            #阈值化处理
            ret, result = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)
            img = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            # img = result

        self.Ny, self.Nx, _ = img.shape  # 取得影像尺寸
        img_data = img.tobytes()
        qimg = QtGui.QImage(img_data, self.Nx, self.Ny, QtGui.QImage.Format_RGB888)

        # view_data 的顯示設定
        self.view_data2.setScaledContents(True)  # 尺度可變
        ### 將 Qimage 物件設置到 view_data 上
        self.view_data2.setPixmap(QtGui.QPixmap.fromImage(qimg))
        ### 顯示大小設定
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
        """ 事件過濾 (找到對應物件並定義滑鼠動作) """
        if source == self.view:  # 找到 view 來源
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
        """ 視窗應用程式關閉事件 """
        if self.ProcessCam.running:
            self.ProcessCam.close()      # 關閉攝影機
            time.sleep(1)
            self.ProcessCam.terminate()  # 關閉子緒
        QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗

    def keyPressEvent(self, event):
        """ 鍵盤事件 """
        if event.key() == QtCore.Qt.Key_Q:   # 偵測是否按下鍵盤 Q
            if self.ProcessCam.running:
                self.ProcessCam.close()      # 關閉攝影機
                time.sleep(1)
                self.ProcessCam.terminate()  # 關閉子緒
            QtWidgets.QApplication.closeAllWindows()  # 關閉所有視窗

    def replace_background(self):
        background_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '../content/image', 'Image Files (*.jpg *.jpeg *.png *.mp4)')[0]
        if not background_path:
            return
        self.browser_path.setText(background_path)

        # if the file is image file
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

    def set_record_flag_stop(self):
        """
        set record flag when clicking the button
        """
        if self.ProcessCam.record_flag:
            self.ProcessCam.record_flag = False
            self.ProcessCam.video_writter.release()
            time.sleep(1)
            self.ProcessCam.video_writter = None

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
            if not self.ProcessCam.video_input.isOpened():
                raise IOError("Cannot open video file")