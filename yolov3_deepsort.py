import itertools
import math
import os
import cv2
import time
import argparse

# import imutils
import torch
import warnings
import numpy as np
import yaml

from detector import build_detector
from deep_sort import build_tracker
from utils.draw import draw_boxes
from utils.parser import get_config
from utils.log import get_logger
from utils.io import write_results
from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from homography import get_homography
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3

def get_centroids_and_groundpoints(bbox_xyxy):
    """
    For every bounding box, compute the centroid and the point located on the bottom center of the box
    @ array_boxes_detected : list containing all our bounding boxes
    """
    array_centroids, array_groundpoints = [], []  # Initialize empty centroid and ground point lists
  #  for index, box in enumerate(array_boxes_detected):
        # Draw the bounding box
        # c
        # Get the both important points
    for bb_xyxy in bbox_xyxy:
  #      bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))
        centroid, ground_point = get_points_from_box(bb_xyxy)
        array_centroids.append(centroid)
        array_groundpoints.append(centroid)
    return array_centroids, array_groundpoints


def get_points_from_box(bb_xyxy):
    """
    Get the center of the bounding and the point "on the ground"
    @ param = box : 2 points representing the bounding box
    @ return = centroid (x1,y1) and ground point (x2,y2)
    """
    # Center of the box x = (x1+x2)/2 et y = (y1+y2)/2
    center_x = int(((bb_xyxy[0] + bb_xyxy[2]) / 2))
    center_y = int(((bb_xyxy[1] + bb_xyxy[3]) / 2))
    # Coordiniate on the point at the bottom center of the box
    center_y_ground = center_y + ((bb_xyxy[3] - bb_xyxy[1]) / 2)
    return (center_x, center_y), (center_x, int(center_y_ground))

def change_color_on_topview(img,pair):
    """
    Draw red circles for the designated pair of points
    """
    cv2.circle(img, (pair[0][0], pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(img, (pair[0][0], pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
    cv2.circle(img, (pair[1][0], pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
    cv2.circle(img, (pair[1][0], pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)


def draw_rectangle(img,corner_points):
    # Draw rectangle box over the delimitation area
    cv2.line(img, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(img, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(img, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)
    cv2.line(img, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE,
             thickness=1)



class VideoTracker(object):
    def __init__(self, cfg, args, video_path):
        self.cfg = cfg
        self.args = args
        self.video_path = video_path
        self.logger = get_logger("root")

        use_cuda = args.use_cuda and torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Running in cpu mode which maybe very slow!", UserWarning)

        if args.display:
            cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("test", args.display_width, args.display_height)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            # self.vdo = cv2.VideoCapture(args.cam)
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()
        self.detector = build_detector(cfg, use_cuda=use_cuda)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.class_names = self.detector.class_names
        self.H = get_homography()
    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)

            # path of saved video and results
            self.save_video_path = os.path.join(self.args.save_path, "results.avi")
            self.save_results_path = os.path.join(self.args.save_path, "results.txt")

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.writer = cv2.VideoWriter(self.save_video_path, fourcc, 20, (self.im_width, self.im_height))

            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        results = []
        idx_frame = 0

        while self.vdo.grab():

            #########################################
            # Load the config for the top-down view #
            #########################################
            #  print(bcolors.WARNING + "[ Loading config file for the bird view transformation ] " + bcolors.ENDC)
            with open("configs/config_birdview.yml", "r") as ymlfile:
                cfg = yaml.safe_load(ymlfile)
            width_og, height_og = 0, 0
            corner_points = []
            for section in cfg:
                corner_points.append(cfg["image_parameters"]["p1"])
                corner_points.append(cfg["image_parameters"]["p2"])
                corner_points.append(cfg["image_parameters"]["p3"])
                corner_points.append(cfg["image_parameters"]["p4"])
                width_og = int(cfg["image_parameters"]["width_og"])
                height_og = int(cfg["image_parameters"]["height_og"])
                img_path = cfg["image_parameters"]["img_path"]
                size_frame = cfg["image_parameters"]["size_frame"]
            #   print(bcolors.OKGREEN + " Done : [ Config file loaded ] ..." + bcolors.ENDC)

            #########################################
            #     Compute transformation matrix		#
            #########################################
            # Compute  transformation matrix from the original frame
            matrix, imgOutput = compute_perspective_transform(corner_points, width_og, height_og, cv2.imread(img_path))
            height, width, _ = imgOutput.shape
            blank_image = np.zeros((height, width, 3), np.uint8)
            height = blank_image.shape[0]
            width = blank_image.shape[1]
            dim = (width, height)

            # Load the image of the ground and resize it to the correct size
            img = cv2.imread("img/chemin_1.png")
            bird_view_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            _, ori_im = self.vdo.retrieve()
            im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
            cv2.imwrite("img_calib.bmp", im)
            # # Draw the green rectangle to delimitate the detection zone
            # draw_rectangle(ori_im, corner_points)

            # do detection
            bbox_xywh, cls_conf, cls_ids = self.detector(im)

            # select person class
            mask = cls_ids == 0

            bbox_xywh = bbox_xywh[mask]
            # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
            bbox_xywh[:, 3:] *= 1.2
            cls_conf = cls_conf[mask]

            # do tracking
            outputs = self.deepsort.update(bbox_xywh, cls_conf, im)

            # draw boxes for visualization
            if len(outputs) > 0:
                bbox_tlwh = []
                bbox_xyxy = outputs[:, :4]
                track_identities = outputs[:, -2]
                track_aruco = outputs[:, -1]
                ori_im = draw_boxes(ori_im, bbox_xyxy, track_identities, track_aruco)

                array_centroids, array_groundpoints = get_centroids_and_groundpoints(bbox_xyxy)
                     # Use the transform matrix to get the transformed coordonates
                transformed_downoids = compute_point_perspective_transformation(matrix, array_groundpoints)

                    # Show every point on the top view image
                for point in transformed_downoids:
                    x, y = point
                    cv2.circle(bird_view_img, (x, y), BIG_CIRCLE, COLOR_GREEN, 2)
                    cv2.circle(bird_view_img, (x, y), SMALL_CIRCLE, COLOR_GREEN, -1)

                list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
                for i, pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
                    # Check if the distance between each combination of points is less than the minimum distance chosen
                    distance_minimum = 110
                    if math.sqrt((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2) < distance_minimum:
                        # Change the colors of the points that are too close from each other to red
                        if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height + 200 or pair[0][
                            1] < 0 or
                                pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height + 200 or pair[1][
                                    1] < 0):
                            change_color_on_topview(bird_view_img, pair)
                            # Get the equivalent indexes of these points in the original frame and change the color to red
                            index_pt1 = list_indexes[i][0]
                            index_pt2 = list_indexes[i][1]
                            cv2.rectangle(ori_im,
                                          (bbox_xyxy[index_pt1][0], bbox_xyxy[index_pt1][1]),
                                          (bbox_xyxy[index_pt1][2], bbox_xyxy[index_pt1][3]),
                                          COLOR_RED, 3)
                            cv2.rectangle(ori_im,
                                          (bbox_xyxy[index_pt2][0], bbox_xyxy[index_pt2][1]),
                                          (bbox_xyxy[index_pt2][2], bbox_xyxy[index_pt2][3]),
                                          COLOR_RED, 3)

                for bb_xyxy in bbox_xyxy:
                    bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                results.append((idx_frame - 1, bbox_tlwh, track_identities, track_aruco))

            end = time.time()

            cv2.imshow("Bird view", bird_view_img)

            if self.args.display:
                cv2.imshow("test", ori_im)
                cv2.waitKey(1)

            if self.args.save_path:
                self.writer.write(ori_im)

            # save results
            # write_results(self.save_results_path, results, 'mot')

            # logging
            self.logger.info("time: {:.03f}s, fps: {:.03f}, detection numbers: {}, tracking numbers: {}" \
                             .format(end - start, 1 / (end - start), bbox_xywh.shape[0], len(outputs)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--config_deepsort", type=str, default="./configs/deep_sort.yaml")
    # parser.add_argument("--ignore_display", dest="display", action="store_false", default=True)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="./output/")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.config_deepsort)

    with VideoTracker(cfg, args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()


