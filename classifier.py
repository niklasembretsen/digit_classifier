import nn_class as nn
import cv2
import numpy as np
from PIL import Image

drawing = False # true if mouse is pressed
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),22,(255,255,255),-1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),22,(255,255,255),-1)


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
	cv2.imshow('image',img)
	k = cv2.waitKey(1) & 0xFF

	if k == 13:
		height, width, channels = img.shape
		pi = Image.fromarray(img)
		pi = pi.convert('L')
		pi = pi.convert('1')
		pi = pi.resize((32,32), Image.ANTIALIAS)
		image_array = np.array(pi).reshape((1,1024))
		X = image_array.astype(int)
		print(X[0].T.shape)
		P = nn.forward_pass(X[0].T)
		dig = np.argmax(P)
		#print(P)
		print("It's a", dig, "motherfucker")

	if k == 8:
		img = np.zeros((512,512,3), np.uint8)

	if k == 27:
		break

cv2.destroyAllWindows()