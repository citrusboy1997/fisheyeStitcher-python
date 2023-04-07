import cv2
import time

USE_PYTORCH = False

if USE_PYTORCH:
    try:
        import \
            torch  # Install PyTorch first: https://pytorch.org/get-started/locally/
        from fisheye_stitcher_pytorch import *

        device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    except ImportError as e:
        print(e)
else:
    from fisheye_stitcher import *

MAX_FOVD = 195.0


def main():
    src_img = cv2.imread("input/dual-fisheye-panorama.jpg")
    width = src_img.shape[1]
    height = src_img.shape[0]

    enb_lc = True
    enb_ra = True
    mls_map_path = 'conf/grid_xd_yd_3840x1920.yml'
    start_time = time.time_ns()
    stitcher = Stitcher(width, height, MAX_FOVD, enb_lc, enb_ra, mls_map_path)
    print("Starting frame stitching..")

    if src_img.size == 0:
        print("Empty image!")
    else:
        img_l = src_img[:, :int(width/2)]
        img_r = src_img[:, int(width/2):]
        stitch_start_time = time.time_ns()

        pano = stitcher.stitch(img_l, img_r)
        stitch_time = round((time.time_ns() - stitch_start_time) / 1000000)
        print('Single stitching time:', stitch_time, 'ms')

        whole_time = round((time.time_ns() - start_time) / 1000000)
        print('Whole processing time:', whole_time, 'ms')

        cv2.namedWindow("left", cv2.WINDOW_NORMAL)
        cv2.namedWindow("right", cv2.WINDOW_NORMAL)
        cv2.namedWindow("pano", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("left", (720, 720))
        cv2.resizeWindow("right", (720, 720))
        cv2.resizeWindow("pano", (1080, 540))

        cv2.imshow("left", img_l)
        cv2.imshow("right", img_r)
        cv2.imshow("pano", pano)

        cv2.waitKey(0)

        cv2.imwrite("output/stitched pano.jpg", pano)


if __name__ == '__main__':
    main()
