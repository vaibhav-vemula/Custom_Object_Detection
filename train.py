import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import yaml

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

params = yaml.safe_load(open('params.yaml'))

print("=========== INGESTING DATASET =============")
rows = open(params['train']['annot_path']).read().strip().split("\n")
data, labels, filenames = [], [], []

for row in rows:
    row = row.split(",")
    (filename, startX, startY, endX, endY) = row
    imagePath = os.path.sep.join([params['train']['img_path'], filename])
    image = cv2.imread(imagePath)
    (h, w) = image.shape[:2]
    startX = float(startX) / w
    startY = float(startY) / h
    endX = float(endX) / w
    endY = float(endY) / h
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    data.append(image)
    labels.append((startX, startY, endX, endY))
    filenames.append(filename)

data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels, dtype="float32")

(trainImages, testImages,
    trainLabels, testLabels,
    _, testFilenames) = train_test_split(data, labels, filenames, test_size=0.10, random_state=42)

print("=========== CREATING TEST SPLIT =============")
f = open(params['output']['test_split'], "w")
f.write("\n".join(testFilenames))
f.close()

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
vgg.trainable = False
vggout = vgg.output

vggout = Flatten()(vggout)
outDense = Dense(256, activation="relu")(vggout)
outDense = Dense(128, activation="relu")(outDense)
outDense = Dense(64, activation="relu")(outDense)
outDense = Dense(32, activation="relu")(outDense)
outDense = Dense(4, activation="sigmoid")(outDense)
model = Model(inputs=vgg.input, outputs=outDense)

model.compile(loss="mse", optimizer=Adam(lr=params['parameters']['initial_lr']))
print(model.summary())

print("=========== TRAININGGGGGG =============")
history = model.fit(trainImages, trainLabels, validation_data=(testImages, testLabels),
    batch_size=params['parameters']['batch_size'],
	epochs=params['parameters']['epochs'],
	verbose=1)

print("=========== SAVING TRAINED MODEL =============")
model.save(params['output']['modelsave_path'], save_format="h5")

ep = params['parameters']['epochs']
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, ep), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, ep), history.history["val_loss"], label="val_loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(params['output']['loss_graph'])