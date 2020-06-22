import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('C:/Users/jjvan/Documents/mayavi_movies/movie007/anim*.png'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('movies/SpaceX_demo.avi',cv2.VideoWriter_fourcc('M','P','E','G'), 20, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()

# Use: http://www.convertfiles.com/convert/video/AVI-to-264.html