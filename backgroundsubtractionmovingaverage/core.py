import numpy as np
import cv2

T_BLUR_PARAM = (4, 4)

class BackgroundSubtractMovingAverage:
    def __init__(self, minArea=2500, updateWeight=0.01):
        self.minArea = minArea
        self.updateWeight = updateWeight
        self.avg = None
        self.avg_float = None

    def update(self, frame):
        '''
        a motion detection, input image and return moving box info
        :param frame: a bgr frame array
        :return:
            xywhs: a list, like [{'x': x, 'y': y, 'w': w, 'h': h}]
            cnts: a list, a output from cv2.findContours
        '''
        if self.avg is None:
            self.avg = cv2.blur(frame, T_BLUR_PARAM)
            self.avg_float = np.float32(self.avg)
            return [], []
        else:
            # 模糊處理
            blur = cv2.blur(frame, T_BLUR_PARAM)

            # 計算目前影格與平均影像的差異值
            diff = cv2.absdiff(self.avg, blur)

            # 將圖片轉為灰階
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # 篩選出變動程度大於門檻值的區域
            #ret, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
            ret, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # 使用型態轉換函數去除雜訊
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

            # 產生等高線
            cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            xywhs = []
            for c in cnts:
                # 忽略太小的區域
                if cv2.contourArea(c) < self.minArea:
                    continue

                # 計算等高線的外框範圍
                (x, y, w, h) = cv2.boundingRect(c)
                xywhs.append({'x': x, 'y': y, 'w': w, 'h': h})

            cv2.accumulateWeighted(blur, self.avg_float, self.updateWeight)
            self.avg = cv2.convertScaleAbs(self.avg_float)
            return xywhs, cnts
