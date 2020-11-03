# coding: utf-8

import io
import os, uuid
from os.path import basename
import sys
import json
import requests
import cv2
import numpy as np
import logging
from io import BytesIO
import pytesseract
from matplotlib import pyplot as plt
from PIL import Image
import threading
import time
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

import config


### Function to write json files
def write_JSON( words , fname ): # words: result to be written; fname: name of the .json file
    JSON_dir = config.JSON_DIR
    with open( JSON_dir + fname + str( uuid.uuid4()) + '.json', 'w', encoding='utf-8') as f:
        json.dump( words, f, ensure_ascii=False, indent=4 )

### Print iterations progress
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

### timing,d child thread
def countdown_time():
  items = list(range(0, 80))
  l = len(items)

  printProgressBar(0, l, prefix = 'Progress: ', suffix = 'Complete', length = 50)
  for i, item in enumerate(items):
    time.sleep(0.1)
    printProgressBar(i + 1, l, prefix = 'Progress: ', suffix = 'Complete', length = 50)

### adjust the image=
def adj_img( img_name ):
  img = cv2.imread( config.IMG_PATH + img_name, 0 )
  img_resize = cv2.resize( img, None, fx = 0.9, fy = 0.9, interpolation=cv2.INTER_CUBIC )
  img_bin = img_resize
  cv2.threshold( img_resize, 100, 255, cv2.THRESH_BINARY_INV, img_bin )

  ### Show
  cv2.imshow( "image", img_bin )
  cv2.imwrite( config.IMG_PATH + 'img_bin2.jpg', img_bin )
  cv2.waitKey(0)
  return

### Tesseract OCR Local
def tesseract_local( img_name ):
  image = Image.open( config.IMG_PATH + img_name )
  text = pytesseract.image_to_string(image, lang='chi_tra+eng')
  return text

### Tesseract OCR Remote
def tesseract_remote( img_url ):
  response = requests.get( img_url )
  img = Image.open( io.BytesIO( response.content ) )
  text = pytesseract.image_to_string( img, lang='chi_tra+eng' )
  print( "=== Tesseract Result ===" )
  log_file.write( "\t" + "=== Tesseract Result ===\n" )

  print( text )
  log_file.write( "\t" + "\t" + text + "\n" )
  return text

### Azure OCR Remote
def azure_ocr_remote( img_url ):

    headers = {'Ocp-Apim-Subscription-Key': subscription_key}
    params = {'language': 'zh-Hant', 'detectOrientation': 'true'}
    data = {'url': img_url}
    response = requests.post(ocr_url, headers=headers, params=params, json=data)
    response.raise_for_status()

    analysis = response.json()
    # print( analysis ) # print out all the json contents

    i = 0
    j = 0

    print( "=== Azure Reslut ===" )
    log_file.write( "\t" + "=== Azure Reslut ===\n" )

    if ( analysis["orientation"] != "NotDetected" ):
      while i < len( analysis["regions"][0]["lines"] ):
        while j < len( analysis["regions"][0]["lines"][i]["words"] ):
          print( analysis["regions"][0]["lines"][i]["words"][j]["text"] )
          log_file.write( "\t" + "\t" + str(analysis["regions"][0]["lines"][i]["words"][j]["text"]) + "\n" )
          j+=1
        i+=1
      print()

    return analysis

### list files locally
def tesseract_ocr():
  ### list all files in the directory and ocr
  img_dir = config.IMG_PATH
  allFileList = os.listdir( img_dir )
  for file in allFileList:
    print( "Processing: " + file )
    result = tesseract_local( file )  ##img_bin2.jpg
    print(result)

### list the blobs inside the container and pass to the function:"azure_ocr_remote()" to analyze, and generate .json files
def blob_list_analysis( container_name ):
  ### get connection string
  connect_str = config.AZURE_STORAGE_CONNECTION_STRING

  ### assign a container
  container_name = container_name
  # container_name = "img-driving-assis-anomaly-road-construction"
  # container_name = "test"
  blob_service_client = BlobServiceClient.from_connection_string( connect_str ) # create BlobServiceClient obj
  container_client = blob_service_client.get_container_client( container_name )

  ### list blobs
  blob_list = container_client.list_blobs()
  for blob in blob_list:
    print( "\t" + blob.name + " analyzing..........." )
    log_file.write( blob.name + " analyzing...........\n" )

    blob_client = BlobClient.from_connection_string( connect_str, container_name=container_name, blob_name=blob.name )
    # print( blob_client.url )
    print()
    az_ocr_result = azure_ocr_remote( blob_client.url ) # az ocr analyze
    tesseract_ocr_result = tesseract_remote( blob_client.url ) # tesseract ocr analyze
    write_JSON( az_ocr_result , "az-ocr-{}".format( blob.name ) )
    # print( "=========================================" )

### Blob Upload files
def blob_upload_img( container_name_input ):
  print("Azure Blob storage v" + __version__ + " - Python quickstart sample")

  ######## get connection string
  connect_str = config.AZURE_STORAGE_CONNECTION_STRING

  ######## assign a container
  blob_service_client = BlobServiceClient.from_connection_string( connect_str ) # create BlobServiceClient obj
  container_name = container_name_input
  # Container='container'
  container_client = blob_service_client.create_container( container_name )

  ######## upload blobs
  local_path = config.BLOB_UPLOAD_PATH
  allFileList = os.listdir( local_path )
  for file in allFileList:
    print( "Uploading to Azure Storage as blob: " + file )
    upload_file_path = os.path.join( local_path, file )
    blob_client = blob_service_client.get_blob_client( container=container_name, blob=file )
    with open( upload_file_path, "rb" ) as data:
      blob_client.upload_blob( data=data )


if __name__ == '__main__':

  try:
    log_file = open( config.LOG_DIR + "result.txt" + str( uuid.uuid4()), "w")

    ### Upload blobs to Containers
    # blob_name = "20201103-test" + str( uuid.uuid4() )
    blob_name = "20201103-test"
    # t = threading.Thread( target = blob_upload_img( blob_name ) )
    # t.start()
    # t.join()
    # print( "Finish Uploading." )
  
    ### set up the subscription key and the endpoint of the Azure Computer Vision API
    subscription_key = config.COMPUTER_VISION_SUBSCRIPTION_KEY
    endpoint = config.COMPUTER_VISION_ENDPOINT
    ocr_url = endpoint + "vision/v3.0/ocr"

    ### Analyze
    print( "blob name: " + blob_name )
    blob_list_analysis( "20201103-test" )

    print("\n======================= DONE. =======================")
    log_file.write( "\n======================= DONE. =======================" )
    log_file.close()
    cv2.destroyAllWindows()

  except:
    logging.exception("Message")