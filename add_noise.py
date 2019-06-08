# import cv2
# import random
# import glob
# import math
# path = glob.glob("/home/utl/PycharmProjects/Calligraphic-Images-Denoising-by-GAN/TrainingSet1/train/*.png")
#
#
# def cord():
#     x = random.randint(0, img.shape[1])
#     y = random.randint(0, img.shape[0])
#     return x,y
#
# for fl in path:
#     n = fl.split("/")[-1]
#     img= cv2.imread(fl)
#     for i in range(50):
#         x, y = cord()
#         r = random.randint(1, 3)
#         a = random.randint(1, 6)
#         cv2.rectangle(img, (x, y), (x + a, y + r), (0, 0, 0), -1)
#         cv2.circle(img, (cord()), r, (0, 0, 0), -1)
#     cv2.imwrite("/home/utl/Downloads/h/"+n, img)
    # cv2.imshow("dhi", img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        # cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
        #


import numpy as np
import os
import cv2
import math

def cord():
    x = random.randint(0, img.shape[1])
    y = random.randint(0, img.shape[0])
    return x,y

def noisy(noise_typ,image):
   if noise_typ == "gauss":
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy
   elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.05
      amount = 0.05
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
   elif noise_typ =="speckle":
      row,col,ch = image.shape
      gauss = np.random.randn(row,col,ch)
      gauss = gauss.reshape(row,col,ch)
      noisy = image + image * gauss
      return noisy
import glob
import random
lstdir=os.listdir("/home/utl/PycharmProjects/Calligraphic-Images-Denoising-by-GAN/test/")

for file in lstdir:
    n = file
    print(n)
    img=cv2.imread("/home/utl/PycharmProjects/Calligraphic-Images-Denoising-by-GAN/test/" + n,0)

    #res=noisy("gauss",img)
    res1=noisy("s&p",img)
    for i in range(100):
        x, y = cord()
        r = random.randint(1, 3)
        a = random.randint(1, 6)
        cv2.rectangle(res1, (x, y), (x + a, y + r), (128,128,128), -1)
        cv2.circle(res1, (cord()), r, (128,128,128), -1)
        #res2=noisy("poisson",img)
    cv2.imwrite("/home/utl/PycharmProjects/Calligraphic-Images-Denoising-by-GAN/test/" + n, res1)

#cv2.imshow("dhi",img)

#cv2.waitKey(0)
#cv2.destroyAllWindows()
#
# def random_noise(img):
#     [r, c, w] = img.shape
#     img_AddNoise = img
#     R = random.randint(1,3)
#     P_noise_x = random.randint(1, 1, [R, c-R])
#     P_noise_y = random.randint(1, 1, [R, r-R])
#     for i in range(1,r):
#         for j in range(1,c):
#             if math.sqrt((P_noise_x - i)^2 + (P_noise_y - j)^2) < R:
#                 if 5 -random.randint(1, 5) >= 1:
#                     img_AddNoise[i, j = 0
#
#     R = random.randint(1, 6);
#     P_noise_x = random.randint(1, 1, [R, c-R]);P_noise_y = random.randint(1, 1, [R, r-R]);
#     for i in range(1, r):
#         for j in range(1, c):
#             if j > P_noise_x and j < P_noise_x + R and i > P_noise_y and i < P_noise_y + R:
#                 if 5 - random.randint(1, 5) >= 1:
#                     img_AddNoise(i, j) = 0
#     return img_AddNoise
