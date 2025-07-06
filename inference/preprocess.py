import cv2
import numpy as np

def deskew_and_align(img):
    return img  # assume roughly horizontal

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def reduce_glare(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
    mask = cv2.dilate(th, np.ones((5,5),np.uint8), iterations=1)
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def preprocess_frame(frame):
    img = deskew_and_align(frame)
    img = enhance_contrast(img)
    img = reduce_glare(img)
    return img
