from random import randint
import numpy as np
import cv2

def centroid_traingle(A,B,C):
    cent_x = ( A[0] + B[0] + C[0]  ) / 3
    cent_y = ( A[1] + B[1] + C[1]  ) / 3
    return [ cent_x , cent_y ]

def get_points(w):
    min_point = 0
    max_point = w 
    A = [ randint( min_point, max_point  ) , randint( min_point, max_point  ) ]
    B = [ randint( min_point, max_point  ) , randint( min_point, max_point  ) ]
    C = [ randint( min_point, max_point  ) , randint( min_point, max_point  ) ]
    return A,B,C

def gen_image_label_pair(w):
    number_of_channels = 1
    x1, x2, x3 =   get_points(w)
    cent = centroid_traingle(x1, x2, x3)
    img = np.zeros((w,w, number_of_channels), dtype=np.int32)
    pts = np.array([x1, x2, x3], dtype=np.int32 )
    pts = pts.reshape(-1,1,2)
    img = cv2.fillPoly(img,[pts], 255)
    return img, cent

def create_data(dataset_size, image_size):
    image_container = []
    label_container = []
    for i in range(dataset_size):
        #print(i)
        img, label = gen_image_label_pair(image_size)
        image_container.append(img)
        label = np.array(label)
        label_container.append(label.flatten() )
    img_ds = np.array(image_container)
    labels_ds = np.array(label_container)
    return img_ds, labels_ds