import face_model
import argparse
import cv2
import sys
import numpy as np
import datetime
import os
import csv

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--image', default='Tom_Hanks_54745.png', help='')
parser.add_argument('--model', default='gender-age/model/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
args = parser.parse_args()

gender_csv = open('face_crop/cat_1/gender_age_list.csv', "wb")
writer = csv.writer(gender_csv, delimiter=",", quoting=csv.QUOTE_MINIMAL)

model = face_model.FaceModel(args)
#img = cv2.imread('Tom_Hanks_54745.png')
for im in os.listdir('./face_crop/cat_1')
  img = cv2.imread(im)
  # img = cv2.resize(img,(112,112))
  img = model.get_input(img)
  # f1 = model.get_feature(img)
  # print(f1[0:10])
  gender, age = model.get_ga(img)
  if gender == 0:
    gen = 'female'
  else:
    gen = 'male'
  row =[im, gender, gen, age]
  writer.writerow(row)
  # for _ in range(5):
  #   gender, age = model.get_ga(img)
  # time_now = datetime.datetime.now()
  # count = 200
  # for _ in range(count):
  #   gender, age = model.get_ga(img)
  # time_now2 = datetime.datetime.now()
  # diff = time_now2 - time_now
  # print('time cost', diff.total_seconds()/count)
  # print('gender is',gender)
  # print('age is', age)

gender_csv.close()