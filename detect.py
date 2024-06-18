import cv2
import numpy as np
from keras.models import load_model

model = load_model(r"models\model_file_30epochs.h5")

faceDetect = cv2.CascadeClassifier(r"models\haarcascade_frontalface_default.xml")