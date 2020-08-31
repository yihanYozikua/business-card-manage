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
import threading
import time
import config

############## Print iterations progress ##############
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

############## Countdown ##############
def countdown_time():
  items = list(range(0, 80))
  l = len(items)

  printProgressBar(0, l, prefix = 'Progress: ', suffix = 'Complete', length = 50)
  for i, item in enumerate(items):
    time.sleep(0.1)
    printProgressBar(i + 1, l, prefix = 'Progress: ', suffix = 'Complete', length = 50)

############## Take Imgs ##############
def shot_cv2():
  # set image storage path
  img_path = config.IMG_STORAGE_PATH

  #capture image by cv2
  cam = cv2.VideoCapture(0)

  img_counter = 0
  while True:
    frame = cam.read()
    if not frame:
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

############## OCR ##############
def ocr():
  image = Image.open('/Users/xiaoyihan/Downloads/IMG_8026.JPG')
  text = pytesseract.image_to_string(image, lang='chi_tra+eng')
  return text

############## Text Sort Out ##############
# def sort_text():


############## Database ##############
# def connect_to_db():


############## Sign Up / In ##############
# def authentication():


if __name__ == '__main__':

  try:
    # shot_cv2()
    ct_countdown = threading.Thread(target = countdown_time) # make count_time() as a child thread function run at background
    ct_countdown.start() # execute the child thread
    result = ocr() # execute the countdown thread

    ct_countdown.join() # wait until the child thread work end
    print(result)
    print("DONE.")
  except:
    logging.exception("Message")