import cv2
import math
import numpy as np

DEBUG = False
MAX_FOVD = 195

P1 = -7.5625e-17
P2 = 1.9589e-13
P3 = -1.8547e-10
P4 = 6.1997e-08
P5 = -6.9432e-05
P6 = 0.9976


class Stitcher:
    def __init__(self, width, height, in_fovd, enb_lc_flag, enb_ra_flag, mls_map_path):

        self.enb_lc_flag = enb_lc_flag
        self.enb_ra_flag = enb_ra_flag
        self.mls_map_path = mls_map_path

        self.m_in_fovd = in_fovd
        self.m_inner_fovd = 183

        # source img
        self.m_ws = int(width / 2)  # 1920
        self.m_hs = height  # 1920
        self.m_ws2 = int(self.m_ws / 2)  # 960
        self.m_hs2 = int(self.m_hs / 2)  # 960

        # destination img
        self.m_wd = int(self.m_ws * 360.0 / MAX_FOVD)
        self.m_hd = int(math.floor(self.m_wd / 2))
        self.m_wd2 = int(math.floor(self.m_wd / 2))
        self.m_hd2 = int(math.floor(self.m_hd / 2))

        self.m_blend_post = []

        # init begin
        self.m_map_x, self.m_map_y = self.fish2map()
        self.m_cir_mask, self.m_inner_cir_mask = self.create_mask()
        self.m_binary_mask = self.create_blend_mask()
        self.m_scale_map = self.gen_scale_map()

        fs = cv2.FileStorage(self.mls_map_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            self.m_mls_map_x = np.float32(fs.getNode("Xd").mat())
            self.m_mls_map_y = np.float32(fs.getNode("Yd").mat())
        else:
            print("Cannot open map file: " + self.mls_map_path)

    def unwarp(self, img):
        img_unwarped = cv2.remap(img, self.m_map_x, self.m_map_y,
                                 interpolation=cv2.INTER_LINEAR)

        return img_unwarped

    def fish2eqt(self, x_dest, y_dest, w_rad):
        """
        Convert fisheye-vertical to equirectangular
        :param x_dest:
        :param y_dest:
        :param w_rad:
        :return:
        """
        phi = x_dest / w_rad
        theta = -y_dest / w_rad + math.pi / 2

        if theta < 0:
            theta = -theta
            phi += math.pi
        if theta > math.pi:
            theta = math.pi - (theta - math.pi)
            phi += math.pi

        s = math.sin(theta)
        v = np.zeros(2)
        v[0] = s * math.sin(phi)
        v[1] = math.cos(theta)
        r = np.sqrt(sum(v ** 2))
        theta = w_rad * math.atan2(r, s * math.cos(phi))

        x_src = theta * v[0] / r
        y_src = theta * v[1] / r

        return x_src, y_src

    def fish2map(self):
        """
        Map 2D fisheye image to 2D projected sphere
        Update member grid maps m_map_x, m_map_y
        :return:
        map_x map for x element.
        map_y map for y element.
        """
        w_rad = self.m_wd / (2 * math.pi)
        w2 = self.m_wd2 - 0.5
        h2 = self.m_hd2 - 0.5
        ws2 = self.m_ws2 - 0.5
        hs2 = self.m_hs2 - 0.5

        mapx = np.zeros((self.m_hd, self.m_wd))
        mapy = np.zeros((self.m_hd, self.m_wd))

        for i in range(self.m_hd):
            y_d = i - h2
            for j in range(self.m_wd):
                x_d = j - w2
                x_s, y_s = self.fish2eqt(x_d, y_d, w_rad)

                x_s += ws2
                y_s += hs2

                mapx[i][j] = x_s
                mapy[i][j] = y_s

        mapx = np.array(mapx, dtype=np.float32)
        mapy = np.array(mapy, dtype=np.float32)

        return mapx, mapy

    def create_mask(self):
        """
        Mask creation for cropping image data inside the FOVD circle
        Update member m_cir_mask (circular mask), inner_cir_mask (circular
        mask for the inner circle).
        :return:
        """
        cir_mask_ = np.zeros((self.m_hs, self.m_ws, 3))
        inner_cir_mask_ = np.zeros((self.m_hs, self.m_ws, 3))
        w_shift = int(math.floor(self.m_ws * (MAX_FOVD - self.m_inner_fovd) / MAX_FOVD) / 2.0)

        r1 = self.m_ws2
        r2 = self.m_ws2 - w_shift * 2
        cv2.circle(cir_mask_, (self.m_ws2, self.m_ws2), r1, (255, 255, 255), -1, 8, 0)
        cv2.circle(inner_cir_mask_, (self.m_ws2, self.m_ws2), r2, (255, 255, 255), -1,
                   8, 0)
        cir_mask_ = np.uint8(cir_mask_)
        inner_cir_mask_ = np.uint8(inner_cir_mask_)

        return cir_mask_, inner_cir_mask_

    def deform(self, img):
        """
        Rigid MLS deformation using the stored map matrix
        :param img: ndarray
                    source image
        :return: ndarray
                 deformed image
        """

        img_deformed = cv2.remap(img, self.m_mls_map_x, self.m_mls_map_y,
                                 interpolation=cv2.INTER_LINEAR)

        return img_deformed

    def gen_scale_map(self):

        w_ = self.m_ws2
        h_ = self.m_hs2
        x_coor = np.array([np.arange(0, w_, 1)], dtype=np.float32)
        r_pf = (P1 * np.power(x_coor, 5)
                                   + P2 * np.power(x_coor, 4)
                                   + P3 * np.power(x_coor, 3)
                                   + P4 * np.power(x_coor, 2)
                                   + P5 * x_coor + P6)
        r_pf = np.divide(np.ones((1, w_)), r_pf)
        scale_map_quad_4 = np.zeros((h_, w_))
        da = r_pf[0][w_ - 1]

        for x in range(w_):
            for y in range(h_):
                r = math.floor(math.sqrt(x ** 2 + y ** 2))
                if r >= w_ - 1:
                    scale_map_quad_4[y][x] = da
                else:
                    a = r_pf[0][r]
                    if x < w_ and y < h_:
                        b = r_pf[0][r+1]
                    else:
                        b = r_pf[0][r]
                    scale_map_quad_4[y][x] = (a + b) / 2

        scale_map_quad_1 = np.flip(scale_map_quad_4, 0)
        scale_map_quad_3 = np.flip(scale_map_quad_4, 1)
        scale_map_quad_2 = np.flip(scale_map_quad_1, 1)

        quad_21 = np.hstack((scale_map_quad_2, scale_map_quad_1))
        quad_34 = np.hstack((scale_map_quad_3, scale_map_quad_4))

        scale_map = np.vstack((quad_21, quad_34))

        if DEBUG:
            scale_map_ = np.expand_dims(scale_map, axis=-1)
            scale_map_ = np.uint8(((scale_map_ - 1) * 1000))
            scale_map_ = cv2.merge((scale_map_, scale_map_, scale_map_))
            # print(scale_map_)

            cv2.namedWindow("scale_map", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("scale_map", (720, 720))
            cv2.imshow("scale_map", scale_map_)
            cv2.waitKey(0)

        return scale_map

    def compensate_light_fall_off(self, img):
        bimg, gimg, rimg = cv2.split(img)

        bimg = np.float32(np.multiply(bimg, self.m_scale_map))
        gimg = np.float32(np.multiply(gimg, self.m_scale_map))
        rimg = np.float32(np.multiply(rimg, self.m_scale_map))

        bimg = np.clip(bimg, 0, 255)
        gimg = np.clip(gimg, 0, 255)
        rimg = np.clip(rimg, 0, 255)

        # out_img = cv2.merge((bimg, gimg, rimg))
        out_img = np.stack((bimg, gimg, rimg), axis=2)
        out_img = np.uint8(out_img)

        return out_img

    def create_blend_mask(self):
        ws2 = self.m_ws2
        hs2 = self.m_hs2
        wd2 = self.m_wd2

        inner_cir_mask_n = cv2.bitwise_not(self.m_inner_cir_mask)
        ring_mask = cv2.copyTo(self.m_cir_mask, inner_cir_mask_n)
        ring_mask_unwarped = cv2.remap(ring_mask, self.m_map_x, self.m_map_y, cv2.INTER_LINEAR)

        mask_ = ring_mask_unwarped[:self.m_hd, wd2-ws2:wd2-ws2 + self.m_ws]

        h_ = mask_.shape[0]
        w_ = mask_.shape[1]

        first_zero_col = 120
        first_zero_row = 45

        for i in range(h_):
            for j in range(w_):
                if j < first_zero_col or j > w_ - first_zero_col:
                    mask_[i][j] = np.array([255, 255, 255], dtype=np.uint8)

        for i in range(h_):
            for j in range(w_):
                if first_zero_col - 1 < j < w_ - first_zero_col + 1\
                        and i < h_ / 2:
                    mask_[i][j] = np.array([0, 0, 0], dtype=np.uint8)

        offset = 15
        for i in range(h_):
            if i > h_ - first_zero_row:
                self.m_blend_post.append(0)
                continue
            for j in range(first_zero_col - 10, int(w_/2+10)):
                if np.array_equal(mask_[i][j], np.array([0, 0, 0], dtype=np.uint8)):
                    self.m_blend_post.append(j - offset)
                    break

        binary_mask = np.uint8(mask_)

        return binary_mask

    def find_match_loc(self, ref, tmpl, img_window=''):
        img = ref
        templ = tmpl
        img_display = img

        match_method = cv2.TM_SQDIFF_NORMED

        result = cv2.matchTemplate(img, templ, match_method)
        result = cv2.normalize(result, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
            match_loc = min_loc
        elif match_method == cv2.TM_CCORR_NORMED:
            match_loc = max_loc

        if DEBUG:
            img_display = cv2.rectangle(img_display, match_loc,
                                        (match_loc[0] + templ.shape[1],
                                         match_loc[1] + templ.shape[0]),
                                        (0, 255, 0), 2)
            ref_temp_cat = np.hstack((img_display, tmpl))
            cv2.imshow(img_window, ref_temp_cat)
            cv2.waitKey(0)

        return match_loc

    def create_control_points(self, match_loc_left, match_loc_right, row_start,
                              row_end, p_wid, p_x1, p_x2, p_x2_ref):
        x1 = match_loc_left[0]
        y1 = match_loc_left[1]
        x2 = match_loc_right[0]
        y2 = match_loc_right[1]

        moving_points = []
        fixed_points = []

        moving_points.append([x1, y1 + row_start])
        moving_points.append([x1 + p_wid, y1 + row_start])
        moving_points.append([x1, y1 + row_end])
        moving_points.append([x1 + p_wid, y1 + row_end])

        moving_points.append([x2 + p_x2_ref, y2 + row_start])
        moving_points.append([x2 + p_x2_ref + p_wid, y2 + row_start])
        moving_points.append([x2 + p_x2_ref, y2 + row_end])
        moving_points.append([x2 + p_x2_ref + p_wid, y2 + row_end])

        fixed_points.append([p_x1, row_start])
        fixed_points.append([p_x1 + p_wid, row_start])
        fixed_points.append([p_x1, row_end])
        fixed_points.append([p_x1 + p_wid, row_end])

        fixed_points.append([p_x2, row_start])
        fixed_points.append([p_x2 + p_wid, row_start])
        fixed_points.append([p_x2, row_end])
        fixed_points.append([p_x2 + p_wid, row_end])

        moving_points = np.array(moving_points)
        fixed_points = np.array(fixed_points)

        return moving_points, fixed_points

    def blend_right(self, bg1, bg2):
        h = bg1.shape[0]
        w = bg1.shape[1]
        wdb = w
        bg1_ = np.array(bg1)
        bg2_ = np.array(bg2)

        bgr_bg1_b, bgr_bg1_g, bgr_bg1_r = cv2.split(bg1_)
        bgr_bg2_b, bgr_bg2_g, bgr_bg2_r = cv2.split(bg2_)

        bgr_bg_b = np.zeros(bgr_bg1_b.shape)
        bgr_bg_g = np.zeros(bgr_bg1_b.shape)
        bgr_bg_r = np.zeros(bgr_bg1_b.shape)

        for i in range(h):
            for j in range(w):
                alpha1 = j / wdb
                alpha2 = 1 - alpha1
                bgr_bg_b[i][j] = (alpha1 * bgr_bg1_b[i][j]
                                  + alpha2 * bgr_bg2_b[i][j])
                bgr_bg_g[i][j] = (alpha1 * bgr_bg1_g[i][j]
                                  + alpha2 * bgr_bg2_g[i][j])
                bgr_bg_r[i][j] = (alpha1 * bgr_bg1_r[i][j]
                                  + alpha2 * bgr_bg2_r[i][j])
        bg = cv2.merge((bgr_bg_b, bgr_bg_g, bgr_bg_r))
        bg = np.uint8(bg)

        return bg

    def blend_left(self, bg1, bg2):
        h = bg1.shape[0]
        w = bg1.shape[1]
        wdb = w
        bg1_ = np.array(bg1)
        bg2_ = np.array(bg2)

        bgr_bg1_b, bgr_bg1_g, bgr_bg1_r = cv2.split(bg1_)
        bgr_bg2_b, bgr_bg2_g, bgr_bg2_r = cv2.split(bg2_)

        bgr_bg_b = np.zeros(bgr_bg1_b.shape)
        bgr_bg_g = np.zeros(bgr_bg1_b.shape)
        bgr_bg_r = np.zeros(bgr_bg1_b.shape)

        for i in range(h):
            for j in range(w):
                alpha1 = (wdb - j + 1) / wdb
                alpha2 = 1 - alpha1
                bgr_bg_b[i][j] = (alpha1 * bgr_bg1_b[i][j]
                                   + alpha2 * bgr_bg2_b[i][j])
                bgr_bg_g[i][j] = (alpha1 * bgr_bg1_g[i][j]
                                   + alpha2 * bgr_bg2_g[i][j])
                bgr_bg_r[i][j] = (alpha1 * bgr_bg1_r[i][j]
                                   + alpha2 * bgr_bg2_r[i][j])
        bg = cv2.merge((bgr_bg_b, bgr_bg_g, bgr_bg_r))
        bg = np.uint8(bg)

        return bg

    def blend(self, left_img, right_img_aligned):
        mask = self.m_binary_mask
        h = mask.shape[0]
        w = mask.shape[1]

        worg = self.m_ws
        img_h = left_img.shape[0]
        img_w = left_img.shape[1]
        left_img_cr = left_img[:img_h, int(img_w / 2 + 1 - worg / 2):int(img_w / 2 + 1 + worg / 2)]

        side_w = 45

        for i in range(h):
            p = self.m_blend_post[i]
            if p == 0:
                continue

            lf_win_1 = left_img_cr[i:i + 1, p - side_w: p + side_w]
            rt_win_1 = right_img_aligned[i:i + 1, p - side_w: p + side_w]

            lf_win_2 = left_img_cr[i:i + 1, w - p - side_w: w - p + side_w]
            rt_win_2 = right_img_aligned[i:i + 1, w - p - side_w: w - p + side_w]

            bleft = self.blend_left(lf_win_1, rt_win_1)
            bright = self.blend_right(lf_win_2, rt_win_2)

            left_img_cr[i:i + 1, p - side_w: p + side_w] = bleft
            right_img_aligned[i:i + 1, p - side_w: p + side_w] = bleft
            left_img_cr[i:i + 1, w - p - side_w: w - p + side_w] = bright
            right_img_aligned[i:i + 1, w - p - side_w: w - p + side_w] = bright

        mask_ = mask[:mask.shape[0] - 2, :mask.shape[1]]
        mask_n = cv2.bitwise_not(mask_)
        left_img_cr = cv2.bitwise_and(left_img_cr, mask_)

        temp1 = left_img[:img_h, :int(img_w / 2 - worg / 2)]
        temp2 = left_img[:img_h, int(img_w / 2 + worg / 2): img_w]

        t = np.hstack((temp1, left_img_cr))
        left_img = np.hstack((t, temp2))

        right_img_aligned = cv2.bitwise_and(right_img_aligned, mask_n)

        pano = left_img
        temp = pano[:img_h, int(img_w / 2 - worg / 2):int(img_w / 2 + worg / 2)]
        t2 = cv2.bitwise_or(temp, right_img_aligned)
        pano[:img_h, int(img_w / 2 - worg / 2):int(img_w / 2 + worg / 2)] = t2

        return pano

    def stitch(self, in_img_l, in_img_r):
        w_in = self.m_ws
        in_img_l = cv2.bitwise_and(in_img_l, self.m_cir_mask)
        in_img_r = cv2.bitwise_and(in_img_r, self.m_cir_mask)
        if DEBUG:
            cv2.imwrite("output/in_img_l.jpg", in_img_l)
            cv2.imwrite("output/in_img_r.jpg", in_img_r)

        if not self.enb_lc_flag:
            left_img_compensated = in_img_l
            right_img_compensated = in_img_r
        else:
            left_img_compensated = self.compensate_light_fall_off(in_img_l)
            right_img_compensated = self.compensate_light_fall_off(in_img_r)

        left_unwarped = self.unwarp(left_img_compensated)
        right_unwarped = self.unwarp(right_img_compensated)

        if DEBUG:
            cv2.imwrite("output/left_unwarped.jpg", left_unwarped)
            cv2.imwrite("output/right_unwarped.jpg", right_unwarped)

        rightimg_crop = right_unwarped[:self.m_hd - 2, int(self.m_wd / 2 - w_in / 2):int(self.m_wd / 2 + w_in / 2)]
        rightimg_mls_deformed = self.deform(rightimg_crop)

        temp1 = left_unwarped[:self.m_hd - 2, :self.m_wd2]
        temp2 = left_unwarped[:self.m_hd - 2, self.m_wd2:self.m_wd2 * 2]

        left_unwarped_arr = np.hstack((temp2, temp1))
        leftimg_crop = left_unwarped_arr[:self.m_hd - 2, int(self.m_wd2 - (w_in / 2)):int(self.m_wd2 + (w_in / 2))]

        if DEBUG:
            cv2.imwrite("output/rightimg_crop.jpg", rightimg_crop)
            cv2.imwrite("output/rightimg_mls_deformed.jpg", rightimg_mls_deformed)
            cv2.imwrite("output/leftimg_crop.jpg", leftimg_crop)

        crop = 0.5 * self.m_ws * (MAX_FOVD - 180) / MAX_FOVD

        p_wid = 55
        p_x1 = 90 - 15
        p_x2 = 1780 - 5
        p_x1_ref = int(2 * crop)
        row_start = 590
        row_end = 1320
        p_x2_ref = int(self.m_ws - 2 * crop + 1)

        ref_1 = leftimg_crop[row_start:row_end, :p_x1_ref]
        ref_2 = leftimg_crop[row_start:row_end, p_x2_ref:self.m_ws]
        tmpl_1 = rightimg_mls_deformed[row_start:row_end, p_x1:p_x1 + p_wid]
        tmpl_2 = rightimg_mls_deformed[row_start:row_end, p_x2:p_x2 + p_wid]

        if not self.enb_ra_flag:
            warped_right_img = rightimg_mls_deformed
        else:
            lwname = 'Matching on left boundary'
            rwname = 'Matching on right boundary'

            match_loc_left = self.find_match_loc(ref_1, tmpl_1, lwname)
            match_loc_right = self.find_match_loc(ref_2, tmpl_2, rwname)

            moving_points, fixed_points = self.create_control_points(
                match_loc_left, match_loc_right, row_start, row_end, p_wid,
                p_x1, p_x2, p_x2_ref)
            tform_refine_mat, mask = cv2.findHomography(fixed_points,
                                                        moving_points, 0)
            warped_right_img = cv2.warpPerspective(rightimg_mls_deformed,
                                                   tform_refine_mat, (
                                                   rightimg_mls_deformed.shape[
                                                       1],
                                                   rightimg_mls_deformed.shape[
                                                       0]), cv2.INTER_LINEAR)

        if DEBUG:
            cv2.imwrite("output/left_unwarped_arr.jpg", left_unwarped_arr)
            cv2.imwrite("output/warped_right_img.jpg", warped_right_img)

        pano = self.blend(left_unwarped_arr, warped_right_img)

        return pano
