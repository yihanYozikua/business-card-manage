import os
from os.path import basename
import sys
import json
import requests
import cv2
import numpy as np
import logging
from io import BytesIO
import pytesseract
from PIL import Image
import config

# take imgs
def shot_cv2():
  # set image storage path
  img_path = config.IMG_PATH

  #capture image by cv2
  cam = cv2.VideoCapture(0)

  img_counter = 0
  while True:
    ret, frame = cam.read()
    if not ret:
      print("ERROR: Fail to grab frame.")
      break
    cv2.imshow("image capture", frame)

    key = cv2.waitKey(1)
    if key%256 == 27:
      # ESC
      print("ESC hit, closing...")
      break
    elif key%256 == 32:
      # SPACE
      img_name = "cv2_shot_{}.jpg".format(img_counter)
      cv2.imwrite( img_path + img_name, frame )
      print("{} written...".format( img_name ))
      img_counter += 1

# OCR
def ocr():
  image = Image.open('/Users/xiaoyihan/Downloads/IMG_8024.JPG')
  text = pytesseract.image_to_string(image, lang='chi_tra')
  print(text)

# # text sort out
# def sort_text():


# # database
# def connect_to_db():


if __name__ == '__main__':

  try:
    # shot_cv2()
    ocr()
  except:
    logging.exception("Message")