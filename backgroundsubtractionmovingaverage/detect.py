import cv2
import sys
import argparse
from time import time as timer

from auomotiondetection.backgroundsubtractionmovingaverage.core import BackgroundSubtractMovingAverage


def draw_frame_fun(frame, xywhs, cnts, str_text=None):
    cv2.putText(frame, str_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 1, cv2.LINE_AA)

    for xywh in xywhs:
        # 畫出外框
        x = xywh['x']
        y = xywh['y']
        w = xywh['w']
        h = xywh['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 畫出等高線（除錯用）
    cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)
    return frame


def main(args):
    b_video_mode = False
    b_skip_draw = False
    # 開啟網路攝影機

    source = args.source
    minArea = args.minArea
    updateWeight = args.updateWeight

    if source[:5] != 'rtsp:':
        b_video_mode = True

    cap = cv2.VideoCapture(source)

    # 設定擷取影像的尺寸大小
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 計算畫面面積
    area = width * height
    bgmvave = BackgroundSubtractMovingAverage(minArea, updateWeight)

    # 初始化平均影像
    ret, frame = cap.read()

    xywhs, cnts = bgmvave.update(frame)
    fps = cap.get(cv2.CAP_PROP_FPS)

    framerate = timer()
    if args.output_dir:
        writer = cv2.VideoWriter(args.output_dir,
                                 cv2.VideoWriter_fourcc(*'MP4V'),  # fourcc(*'mp4') for mp4 format output
                                 fps,  # fps
                                 (width, height))  # resolution
    else:
        fps /= 1000
    elapsed = 0
    total_idx = 0

    while (cap.isOpened()):
        # 讀取一幅影格
        ret, frame = cap.read()
        total_idx += 1
        # if (total_idx % 8) != 0:
        #     continue

        # 若讀取至影片結尾，則跳出
        if ret == False:
            break
        start = timer()
        # 模糊處理
        xywhs, cnts = bgmvave.update(frame)

        if not b_skip_draw:
            # 畫出 frame_idx
            str_text = '%d' % (total_idx,)
            frame = draw_frame_fun(frame, xywhs, cnts, str_text)
        if args.output_dir:
            writer.write(frame)

        else:
            # 顯示偵測結果影像
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if b_video_mode:
                # diff = timer() - start
                # while diff < fps:
                #     diff = timer() - start

                elapsed += 1
                if elapsed % 5 == 0:
                    print('\r', flush=True)
                    # print('{0:3.3f} FPS'.format(elapsed / (timer() - framerate)), flush=True)
                    print('{0:3.3f} FPS'.format(elapsed / (timer() - start)), flush=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # input and output
    parser.add_argument('--source', type=str, help='video source', required=True)
    parser.add_argument('--output_dir', type=str, help='video output')
    parser.add_argument('--minArea', type=int, default=2500, help='minArea')
    parser.add_argument('--updateWeight', type=float, default=0.01, help='updateWeight')
    args = parser.parse_args()
    print(args)
    main(args)
