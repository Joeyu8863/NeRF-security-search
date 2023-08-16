import cv2
import numpy as np
if __name__ == '__main__':
   img = cv2.imread('data/nerf_synthetic/lego/train/r_46.png', cv2.IMREAD_UNCHANGED)
   cv2.imwrite('ori.png', img)
   h,w = img.shape[:2]
   print('Image Dimensions: height={}, width={}'.format(h, w))
# Define a black  mask
   mask = np.zeros_like(img)
#Flip the mask color to white
   mask[:, :] = [255, 255, 255,255] 

# Draw a black filled rectangle on top of the mask to 
# hide a specific area
   start_point = ((w//2-30),(h//4-100))
   end_point = ((w//2 + 30),(h//4 + 100))
   color = (0,0,0)
   mask = cv2.rectangle(mask, start_point, end_point, color, -1)
#mask = 1 - mask
# Apply the mask to image
   result = cv2.bitwise_and(img,mask)

   cv2.imwrite('r_46.png', result)
   
