
import cv2
import time

USE_PYTORCH = False

if USE_PYTORCH:
    from fisheye_stitcher_torch import *
else:
    from fisheye_stitcher import *

MAX_FOVD = 195.0


def main():
    src_video = 'input/dual-fisheye-panorama-video.mp4'
    cap = cv2.VideoCapture(src_video)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
    filename = 'output/stitched_pano_video.avi'
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    enb_lc = True
    enb_ra = True
    mls_map_path = 'conf/grid_xd_yd_3840x1920.yml'
    start_time = time.time_ns()
    stitcher = Stitcher(width, height, MAX_FOVD, enb_lc, enb_ra, mls_map_path)
    print("Starting frame stitching..")

    single_stitch_time_list = []
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if frame is None:
                print('Get the last frame')
                break
            else:
                img_l = frame[:, :int(width / 2)]
                img_r = frame[:, int(width / 2):]
                stitch_start_time = time.time_ns()
                pano = stitcher.stitch(img_l, img_r)
                pano = cv2.resize(pano, (width, height))
                stitch_time = round(
                    (time.time_ns() - stitch_start_time) / 1000000)
                single_stitch_time_list.append(stitch_time)
                video.write(pano)
                counter += 1

    cap.release()
    video.release()
    whole_time = round((time.time_ns() - start_time) / 1000000)
    print('Whole processing time:', whole_time, 'ms')
    avg_time = sum(single_stitch_time_list) / len(single_stitch_time_list)
    max_time = max(single_stitch_time_list)
    min_time = min(single_stitch_time_list)
    print('Average single stitching time:', avg_time, 'ms')
    print('Maximum single stitching time:', max_time, 'ms')
    print('Minimum single stitching time:', min_time, 'ms')


if __name__ == '__main__':
    main()
