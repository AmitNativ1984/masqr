import cv2
import numpy as np
import random

def get_ax_ay(m_pixel, m_world):
    ax = (-m_world[0], -m_world[1], -1,
          0, 0, 0,
          m_pixel[0] * m_world[0], m_pixel[0] * m_world[1], m_pixel[0])

    ay = (0, 0, 0,
          -m_world[0], -m_world[1], -1,
          m_pixel[1] * m_world[0], m_pixel[1] * m_world[1], m_pixel[1])

    return ax, ay


def get_homography():
    # Pixel coordinates
    pixel_lst = [(633 ,  298),      #1 (u1', v1') u1'=u1/w1, v1'=v1/w1
                (466 ,  238),       #2 (u2', v2') u2'=u2/w2, v2'=v2/w2
                (331 ,  190),       #3 (u3', v3') u3'=u3/w3, v3'=v3/w3
                (277 ,  170),       #4 (u4', v4') u4'=u4/w4, v4'=v4/w4
                (512 ,  389),       #5 (u5', v5') u5'=u5/w5, v5'=v5/w5
                (334 ,  303),       #6 (u6', v6') u6'=u6/w6, v6'=v6/w6
                (199 ,  237),       #7 (u7', v7') u7'=u7/w7, v7'=v7/w7
                (145 ,  397),       #8 (u8', v8') u8'=u8/w8, v8'=v8/w8
                (21 ,  301)]        #9 (u9', v9') u9'=u9/w9, v9'=v9/w9

    world_lst = [(0, 0),            #1 (x1, y1, 1)
                (0, 111.1),        #2 (x2, y2, 1)
                (0, 231.4),        #3 (x3, y3, 1)
                (0, 291.8),        #4 (x4, y4, 1)
                (-120.2, 0),       #5 (x5, y5, 1)
                (-120.2, 111.1),   #6 (x6, y6, 1)
                (-120.2, 231.4),   #7 (x7, y7, 1)
                (-240.8, 111.1),   #8 (x8, y8, 1)
                (-240.8, 231.4)]   #9 (x9, y9, 1)

    # ax1 = (-x1, -y1, -1, 0, 0, 0, u1'x1, u1'y1, u1')
    # ay1 = (0, 0, 0, -x1, -y1, -1, v1'x1, v1'y1, v1')
    # A = []
    # for world, pixel in zip(world_lst, pixel_lst):
    #     ax, ay = get_ax_ay(world, pixel)
    #     A.append(ax)
    #     A.append(ay)
    #
    # A_mat = np.array(A)
    # A_transpose_A = np.matmul(A_mat.transpose(), A_mat)
    # w, v = np.linalg.eig(A_transpose_A)
    # # Minimum eigen value index
    # min_ev_index = np.argmin(w)
    # H = v[np.argmin(w)].reshape(3, 3)
    H, mask = cv2.findHomography(np.array(world_lst), np.array(pixel_lst), cv2.RANSAC, 5.0)
    return H

if __name__ == '__main__':
    homography = get_homography()
    print(homography)
