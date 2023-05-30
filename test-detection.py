import cv2
import numpy as np

def load_image(image_path):
    loaded_image = cv2.imread(image_path)
    return loaded_image
    

def preprocess_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

