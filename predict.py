import numpy as np
import imutils
import cv2
import os
import yaml

from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.models import load_model

params = yaml.safe_load(open('params.yaml'))['test']

filenames = open('output/test_images.txt').read().strip().split("\n")
imagePaths = []

for f in filenames:
    p = os.path.sep.join([params['img_path'], f])
    imagePaths.append(p)

print("[=========== LOADING MODEL =============")
model = load_model(params['model'])

for imagePath in imagePaths:
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    preds = model.predict(image)[0]
    (startX, startY, endX, endY) = preds
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)