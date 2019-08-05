import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace
import math

thresh = 0.8
scales = [1024, 1980]
video_file = ''
pad_ratio = 0.1

count = 1
frame_count = 0

gpuid = 0
detector = RetinaFace('./models/R50', 0, gpuid, 'net3')

cam = cv2.VideoCapture(video_file)

while True:
  ret, img = cam.read()
  frame_count += 1
  print(frame_count)
  # img = cv2.imread('sample-images/q2.png')
  print(img.shape)
  im_shape = img.shape
  target_size = scales[0]
  max_size = scales[1]
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  #im_scale = 1.0
  #if im_size_min>target_size or im_size_max>max_size:
  im_scale = float(target_size) / float(im_size_min)
  # prevent bigger axis from being more than max_size:
  if np.round(im_scale * im_size_max) > max_size:
      im_scale = float(max_size) / float(im_size_max)

  # print('im_scale', im_scale)

  scales = [im_scale]
  flip = False

  for c in range(count):
    faces, landmarks = detector.detect(img, thresh, scales=scales, do_flip=flip)
    # print(c, faces.shape, landmarks.shape)

  if faces is not None:
    # print('find', faces.shape[0], 'faces')
    for i in range(faces.shape[0]):
      #print('score', faces[i][4])
      box = faces[i].astype(np.int)
      #color = (255,0,0)
      # color = (0,0,255)
      # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
      # print("Box {}_{},{},{},{}".format(i, box[0], box[1], box[2], box[3]))
      width_face_init = box[2] - box[0]
      height_face = box[3] - box[1]
      padding = int(pad_ratio * height_face)
      box_0 = box[0] - padding
      box_1 = box[1] - padding
      width_face = width_face_init + 2*padding
      height_face = height_face + 2*padding

      if landmarks is not None:
        landmark5 = landmarks[i].astype(np.int)
        #print(landmark.shape)
        eye_left = landmark5[0]
        eye_right = landmark5[1]
        distance = (math.sqrt(((eye_left[0]-eye_right[0])**2)+((eye_left[1]-eye_right[1])**2)))/width_face_init
        # print("Eye distance",distance)
        # for l in range(landmark5.shape[0]):
        #   color = (0,0,255)
        #   if l==0 or l==1:
        #     color = (0,255,0)
        #   cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

      if distance > 0.3 and width_face >= 120:
        img_crop = img[box_1:box_1+height_face, box_0:box_0+width_face]
        cv2.imwrite('face_crop/cat_1/{}_{}.jpg'.format(frame_count,i), img_crop)

      elif distance <= 0.3 and width_face > 120:
        img_crop = img[box_1:box_1+height_face, box_0:box_0+width_face]
        cv2.imwrite('face_crop/cat_2/{}_{}.jpg'.format(frame_count,i), img_crop)

      else:
        img_crop = img[box_1:box_1+height_face, box_0:box_0+width_face]
        cv2.imwrite('face_crop/cat_3/{}_{}.jpg'.format(frame_count,i), img_crop)

  if cv2.waitKey(1) == 27:
      break  # esc to quit
cv2.destroyAllWindows()

    # filename = './detector_test.jpg'
    # print('writing', filename)
    # cv2.imwrite(filename, img)

