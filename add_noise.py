import cv2
import random
import glob
import math
path = glob.glob("/home/utl/Downloads/gh/*.png")


def cord():
    x = random.randint(0, img.shape[1])
    y = random.randint(0, img.shape[0])
    return x,y

for fl in path:
    n = fl.split("/")[-1]
    img= cv2.imread(fl)
    for i in range(50):
        x, y = cord()
        r = random.randint(1, 3)
        a = random.randint(1, 6)
        cv2.rectangle(img, (x, y), (x + a, y + r), (0, 0, 0), -1)
        cv2.circle(img, (cord()), r, (0, 0, 0), -1)
    cv2.imwrite("/home/utl/Downloads/h/"+n, img)
    # cv2.imshow("dhi", img)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

        # cv2.ellipse(img,(256,256),(100,50),0,0,180,255,-1)
        #





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
