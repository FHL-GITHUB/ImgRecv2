import cv2,os,glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from random import randint

def extractBBox(image,num):

    #clear the images in the sub directory
    savedir = os.path.join(os.getcwd(), 'model_test')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    boxes = []
    bb=[]
    transform = transforms.ToTensor()

    new_image = image.copy()
    filelist = glob.glob(os.path.join(savedir, "*.jpg"))

    gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY) # convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5,5), 0) # apply gaussian blur to blur the background
    thresh0 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 0)
    contours, _ = cv2.findContours(thresh0, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    
    #with_contours = cv2.drawContours(new_image, contours, -1,(0,255,0),1)

    #print('Total number of contours detected: ' + str(len(contours)))

    count = 0
    images = []
    cropped = []
    for i,contour in enumerate(contours):
        
        area = cv2.contourArea(contour)

        if area < 300:
            continue
        elif area > 12000:
            continue
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect

        
        if w < 20 or h < 20:
            continue
        if w > 2*h:
            continue

        img = new_image[y:y+h,x:x+w]

        print('Bounding Box: ',count)

        predImg = img.copy()
        images.append(predImg)

        #img = cv2.resize(img,(32,32))
        img = cv2.resize(img,(224,224))
    
        img = Image.fromarray(img)
        img = transform(img)
        img.requires_grad=False
        bb.append(img)
        boxes.append(np.array(rect))
        count+=1
    
    for i, rect in enumerate(boxes):
        img_copy = new_image.copy()
        x, y, w, h = rect

        if rect is not None:
            crop = img_copy[y:y+h,x:x+w]
            #save boumding box image to the destination for prediction
            newname = str(num)+'_'
            cv2.imwrite(os.path.join(savedir, newname+str(i) + '.jpg'), crop)

            #cv2.rectangle(img_copy, (x,y),(x+w,y+h), (255, 0, 255), 2)
            print('BB '+str(i)+str(x)+' '+str(y)+' '+str(x+w)+' '+str(y+h))
            print(str(w)+' '+str(h))
            img_copy = cv2.rectangle(img_copy, (x,y),(x+w,y+h), (0, 0, 0), 1)
            
            #crop = img_copy[y:y+h,x:x+w]
            #cv2.imshow('Image with bounding boxes', crop)
            #cv2.waitKey(0) # Wait for keypress to continue
            #cv2.destroyAllWindows()

            cropped.append(crop)

    print(boxes)
    print(len(cropped))
    print(len(bb))
    return boxes,bb,cropped,new_image
    
if __name__ == '__main__':
    num = 0
    filelist = glob.glob(os.path.join(os.getcwd(), "*.jpg"))
	for f in filelist:
        img = cv2.imread(f)
        extractBBox(img,num)
        num += 1
        
