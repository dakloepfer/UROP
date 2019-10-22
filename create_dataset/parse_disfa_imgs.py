### Compared to the OpenFace landmarks, get rid of the OpenFace
### landmarks 60 and 64 (0-based index from 0 to 67)
### I have already filtered those out from the model they gave

### Code heavily inspired by yinguobing/head-pose-estimation

"""Estimate head pose according to the facial landmarks, uses DISFA videos"""
import argparse
import os
import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.io

ARGS = argparse.ArgumentParser()
ARGS.add_argument('--vid_dir', type=str, help='Directory for videos')
ARGS.add_argument('--lmk_dir', type=str, help='Directory for landmarks')
ARGS.add_argument('--model_file', type=str, help='File for 3D model')
ARGS.add_argument('--save_dir', type=str, help='directory to save images in')
ARGS.add_argument('--width_px', type=int, default=256,
                  help='number of pixels that final image shall be wide')
ARGS.add_argument('--height_px', type=int, default=256,
                  help='number of pixels that final image shall be high')
ARGS.add_argument('--rotation_cutoff', type=float,
                  help='Maximum value in rad that the modulus of the \
                        rotation vector is allowed to deviate from the mean for the video')
ARGS.add_argument('--align_faces', action='store_true', default=True,
                  help='Decides whether to align faces according to connecting line between eyes')
ARGS.add_argument('--width_factor', type=float, default=1.2,
                  help='Factor by which width is increased')
ARGS.add_argument('--height_factor', type=float, default=1.3,
                  help='Factor by which height is increased')
OPT = ARGS.parse_args()

T1 = time.time()

class PoseEstimator:
    """Estimate head pose according to the facial landmarks"""

    def __init__(self, img_size=(480, 640), model_file='model.txt'):
        self.size = img_size

        # 3d Model Points
        self.model_points = self._get_full_model_points(model_file)

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        #self.r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        #self.t_vec = np.array(
        #    [[-14.97821226], [-10.62040383], [-2053.03596872]])
        self.r_vec = None
        self.t_vec = None

    def _get_full_model_points(self, filename):
        """Get all 66 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1

        return model_points

    def solve_pose(self, image_points):
        """
        Solve pose from all the 66 image points
        Return (rotation_vector, translation_vector) as pose.
        """

        if self.r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                self.model_points, image_points, self.camera_matrix, self.dist_coeefs)
            self.r_vec = rotation_vector
            self.t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points,
            image_points,
            self.camera_matrix,
            self.dist_coeefs,
            rvec=self.r_vec,
            tvec=self.t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)

    def draw_annotation_box(self, image, rotation_vector, translation_vector,
                            color=(255, 255, 255), line_width=2):
        """Draw a 3D box as annotation of pose"""
        point_3d = []
        rear_size = 75
        rear_depth = 0
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = 100
        front_depth = 100
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d image points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          self.camera_matrix,
                                          self.dist_coeefs)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    def get_pose_marks(self, marks):
        """Get marks ready for pose estimation from 66 marks"""
        pose_marks = []
        pose_marks.append(marks[30])    # Nose tip
        pose_marks.append(marks[8])     # Chin
        pose_marks.append(marks[36])    # Left eye left corner
        pose_marks.append(marks[45])    # Right eye right corner
        pose_marks.append(marks[48])    # Mouth left corner
        pose_marks.append(marks[54])    # Mouth right corner
        return pose_marks

def filter_frames(img_size=(768, 1024)):
    '''
        Takes in directory with all landmarks, and a file containing a 3d model with
        world-coordinates for landmarks, and adds the corresponding frame number to a
        list if |rot_vector|-avg(|rot_vector|) < rotation_cutoff
        filtered_frame_ids has format:
        [('SN001', array[frame numbers]), ('SN002, array[frame numbers]), ...]
    '''
    landmark_dir = OPT.lmk_dir
    model_file = OPT.model_file
    rotation_cutoff = OPT.rotation_cutoff

    pose_estimator = PoseEstimator(img_size=img_size, model_file=model_file)
    filtered_frame_ids = []

    total_number_of_frames = 0

    for vid_id in os.listdir(landmark_dir):
        print('Now filtering', vid_id)

        all_frame_nos = np.empty(len(os.listdir(os.path.join(landmark_dir, vid_id))), dtype=int)
        all_rot_vector_lengths = 1000 * np.ones(len(os.listdir(os.path.join(landmark_dir, vid_id))))
        all_landmarks = np.empty((len(os.listdir(os.path.join(landmark_dir, vid_id))), 66, 2))


        for i, landmark_file in \
            tqdm(enumerate(os.listdir(os.path.join(landmark_dir, vid_id))), 'Filtering frames'):

            if landmark_file[0] == 'S':
                frame_number = int(landmark_file[6:10])
            else:
                frame_number = int(landmark_file[2:6]) -1

            all_frame_nos[i] = frame_number

            # Filter out every frame that is in the error log
            if vid_id == 'SN001':
                if 398 <= frame_number <= 420 or 3190 <= frame_number <= 3243:
                    continue
            elif vid_id == 'SN002':
                if 800 <= frame_number <= 826:
                    continue
            elif vid_id == 'SN004':
                if 4541 <= frame_number <= 4555:
                    continue
            elif vid_id == 'SN006':
                if 1349 <= frame_number <= 1405:
                    continue
            elif vid_id == 'SN009':
                if 1736 <= frame_number <= 1808 or 1851 <= frame_number <= 1885:
                    continue
            elif vid_id == 'SN011':
                if 4529 <= frame_number <= 4533 or 4830 <= frame_number <= 4845:
                    continue
            elif vid_id == 'SN021':
                if 574 <= frame_number <= 616 or 985 <= frame_number <= 1164 or \
                    1190 <= frame_number <= 1205 or 1305 <= frame_number <= 1338 or \
                        1665 <= frame_number <= 1710 or 1862 <= frame_number <= 2477 or \
                            2554 <= frame_number <= 4657 or 4710 <= frame_number <= 4722:
                    continue
            elif vid_id == 'SN023':
                if 1021 <= frame_number <= 1049 or 3378 <= frame_number <= 3557 or \
                    3584 <= frame_number <= 3668 or 4547 <= frame_number <= 4621 or \
                        4741 <= frame_number <= 4772 or 4825 <= frame_number <= 4845:
                    continue
            elif vid_id == 'SN025':
                if 4596 <= frame_number <= 4662 or 4816 <= frame_number <= 4835:
                    continue
            elif vid_id == 'SN027':
                if 3461 <= frame_number <= 3494 or 4738 <= frame_number <= 4785:
                    continue
            elif vid_id == 'SN028':
                if 1875 <= frame_number <= 1885 or 4571 <= frame_number <= 4690:
                    continue
            elif vid_id == 'SN029':
                if 4090 <= frame_number <= 4543:
                    continue
            elif vid_id == 'SN030':
                if 939 <= frame_number <= 962 or 1406 <= frame_number <= 1422 or \
                    2100 <= frame_number <= 2132 or 2893 <= frame_number <= 2955:
                    continue

            landmark_path = os.path.join(landmark_dir, vid_id, landmark_file)

            # landmarks are [[x1, y1], [x2, y2], ...] starting from top left corner
            landmarks = scipy.io.loadmat(landmark_path)['pts']

            rotation_vector, _ = pose_estimator.solve_pose(landmarks)

            all_rot_vector_lengths[i] = np.linalg.norm(rotation_vector)

            all_landmarks[i, :, :] = landmarks

        mean_rotvector = np.mean(all_rot_vector_lengths[np.where(all_rot_vector_lengths < 999)])

        filtered_ids = all_frame_nos[
            np.where(np.absolute(all_rot_vector_lengths - mean_rotvector) < rotation_cutoff)]

        filtered_frame_ids.append((vid_id, filtered_ids, np.array(all_landmarks)))

        print('Video %s keeps %d frames\n'%(vid_id, filtered_ids.size))
        total_number_of_frames += filtered_ids.size

    print('Total number of frames that will be saved:', total_number_of_frames)

    return filtered_frame_ids

def return_all_frames():
    '''
        Returns the same format as filter_frames(), but does not filter out anything
    '''
    landmark_dir = OPT.lmk_dir
    filtered_frame_ids = []
    total_number_of_frames = 0

    for vid_id in os.listdir(landmark_dir):

        all_frame_nos = np.empty(len(os.listdir(os.path.join(landmark_dir, vid_id))), dtype=int)
        all_landmarks = np.empty((len(os.listdir(os.path.join(landmark_dir, vid_id))), 66, 2))

        for i, landmark_file in enumerate(os.listdir(os.path.join(landmark_dir, vid_id))):

            if landmark_file[0] == 'S':
                frame_number = int(landmark_file[6:10])
            else:
                frame_number = int(landmark_file[2:6]) -1

            all_frame_nos[i] = frame_number

            landmark_path = os.path.join(landmark_dir, vid_id, landmark_file)

            # landmarks are [[x1, y1], [x2, y2], ...] starting from top left corner
            landmarks = scipy.io.loadmat(landmark_path)['pts']

            all_landmarks[i, :, :] = landmarks

        filtered_frame_ids.append((vid_id, all_frame_nos, np.array(all_landmarks)))

        print('Video %s saves %d frames\n'%(vid_id, all_frame_nos.size))
        total_number_of_frames += all_frame_nos.size

    print('Total number of frames that will be saved:', total_number_of_frames)

    return filtered_frame_ids

def rotate_img_and_bbox(img, landmarks):
    '''
        Takes in image and corresponding landmarks and rotates image so that eyes are horizontal, 
        returns rotated image as well as rotated landmarks
    '''
    # calculate angle by which face is off the horizontal
    left_eye_points = landmarks[36:42, :]
    right_eye_points = landmarks[42:48, :]

    left_eye_center = np.mean(left_eye_points, axis=0, dtype=int)
    right_eye_center = np.mean(right_eye_points, axis=0, dtype=int)

    delta_y = right_eye_center[1] - left_eye_center[1]
    delta_x = right_eye_center[0] - left_eye_center[0]

    angle = np.degrees(np.arctan2(delta_y, delta_x))

    eyes_midpoint = ((left_eye_center[0] + right_eye_center[0]) // 2,
                     (left_eye_center[1] + right_eye_center[1]) // 2)

    # get rotation matrix
    cv2_rot_matrix = cv2.getRotationMatrix2D(eyes_midpoint, angle, 1)
    new_image = cv2.warpAffine(img, cv2_rot_matrix, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_CUBIC)

    new_landmarks = np.matmul(cv2_rot_matrix[:, :2], landmarks.T)
    new_landmarks = np.add(new_landmarks.T, + cv2_rot_matrix[:, 2])

    return new_image, new_landmarks

def crop_image(img, landmarks):
    '''
        detects face and crops to that face; image afterwards has size width_px * height_pxs
    '''

    y_coord = np.amin(landmarks[:, 1])
    right = np.amax(landmarks[:, 0])
    bottom = np.amax(landmarks[:, 1])
    x_coord = np.amin(landmarks[:, 0])

    width = int((right - x_coord) * OPT.width_factor)
    height = int((bottom - y_coord) * OPT.height_factor)
    x_coord = int(x_coord - 0.5 * (right - x_coord) * (OPT.width_factor - 1))
    y_coord = int(y_coord - 0.5 * (bottom - y_coord) * (OPT.height_factor - 1))

    if width > height:
        diff = width - height
        height = width
        y_coord -= int(0.5 * diff)
    elif height > width:
        diff = height - width
        width = height
        x_coord -= int(0.5 * diff)

    # edge case
    if x_coord < 0:
        x_coord = 0
    if y_coord < 0:
        y_coord = 0

    img_height, img_width, _ = img.shape
    if x_coord + width > img_width:
        x_coord = img_width - width
    if y_coord + height > img_height:
        y_coord = img_height - height

    face = img[y_coord:y_coord + height, x_coord:x_coord + width, :]
    face = cv2.resize(face, dsize=(OPT.width_px, OPT.height_px))

    return face

def parse_videos(filtered_frame_ids):
    '''
        Wrapper method to save the correct frames, cropped to img_size, to save_dir
    '''

    for (video_id, frame_numbers, all_landmarks) in tqdm(filtered_frame_ids, desc='Videos saved:'):
        print('Starting to crop and save video', video_id)
        video_file_path = os.path.join(OPT.vid_dir, 'LeftVideo%s_comp.avi' %video_id)

        vidcap = cv2.VideoCapture(video_file_path)
        success, image = vidcap.read()

        frame_ind = 0
        while success:

            if frame_ind in frame_numbers:
                if OPT.align_faces:
                    image, landmarks = rotate_img_and_bbox(image, all_landmarks[frame_ind, :, :])
                else:
                    landmarks = all_landmarks[frame_ind, :, :]

                image = crop_image(image, landmarks)
                filename = video_id + '_' + str(frame_ind) + '.jpg'
                # save image as JPEG file
                cv2.imwrite(OPT.save_dir + '/' + filename, image)

            success, image = vidcap.read()
            frame_ind += 1

            if frame_ind % 1000 == 0:
                print('finished %d images in video, now %f has passed' %(frame_ind, time.time()-T1))

        print('finished video', video_id)

def plot_landmarks(video_file='disfa_dataset/raw_data/Videos/LeftVideoSN001_comp.avi', \
                   landmark_file='disfa_dataset/raw_data/Landmarks/SN001/SN001_0000_lm.mat', \
                   frame_number=0):
    '''
        Takes in image file name and numpy landmarks array
        and plots landmarks on top of image
    '''
    vidcap = cv2.VideoCapture(video_file)
    for _ in range(frame_number+1):
        _, image = vidcap.read()
    image = np.array(image)
    landmarks = scipy.io.loadmat(landmark_file)['pts']

    image, landmarks = rotate_img_and_bbox(image, landmarks)

    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1])
    plt.show()


if __name__ == '__main__':

    #plot_landmarks()

    if not os.path.exists(OPT.save_dir):
        os.mkdir(OPT.save_dir)

    if OPT.rotation_cutoff:
        FILTERED_FRAME_IDS = filter_frames()
    else:
        FILTERED_FRAME_IDS = return_all_frames()

    parse_videos(FILTERED_FRAME_IDS)
