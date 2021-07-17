import torch,glob
from efficientnet_pytorch import EfficientNet
import cv2
from torch.utils.data import DataLoader
import os
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import ImageFolder
from extract import extractBBox

import torch.nn as nn
import numpy as np


from pytorch_image_folder_with_file_paths import ImageFolderWithPaths as ImageFolderPath

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y
    
def test():
    img = cv2.imread("./model_test/up/1.jpg")
    img1 = img
    boxes,bb,images,img_copy=extractBBox(img)  

    result_save_dir = os.path.join(os.getcwd(), 'model_test')

    classes = ['0', '6', '7', '8', '9', "Bull", 'Down', 'Left', 'Right', 'Stop', 'Up', 'V', 'W', 'X', 'Y', 'Z']
    dic_class = {'Up':1,'Down':2,'Right':3,'Left':4,'Stop':5,'6':6,'7':7,'8':8,'9':9,'0':10,'V':11,'W':12,'X':13,'Y':14,'Z':15,'Bull':0}
    net = torch.load("model_dic.pt")
    net.eval()
    #TRAIN_PATH = os.path.join(os.getcwd(), 'train')
    VALIDATE_PATH = os.path.join(os.getcwd(), 'model_test')

    # transform = Compose([ToTensor(), Resize((32, 32))])
    #transform = Compose([ToTensor(), Resize((224, 224)), Grayscale(3)])
    transform = Compose([ToTensor(), Resize((32, 32)), Grayscale(3)])
    # Load train data and test data
    #train_data = ImageFolder(root=TRAIN_PATH, transform=transform)

    validate_data = ImageFolderPath(root=VALIDATE_PATH, transform=transform)

    validate_ds = DataLoader(validate_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    labels = {}
    count = 0

    for img, lbl,path in validate_ds:
        print(path[0])
        file_name = path[0].split('\\')[-1]
        file_name = file_name.split(".")[0] 
        file_name = int(file_name)
        scores = net(img)

        x,y = torch.max(scores,1)

        label = scores.argmax(dim=1).numpy()
        print(label)
        #print(classes[label[0]])

        labels[x[0].item()] = [int(label[0]),file_name]
        count+=1

    print(labels)
    print(max(labels.keys()))
    score = max(labels.keys())
    max_file_name = labels[max(labels.keys())][1]
    print('herererer')
    print(max_file_name)
    pred_result = classes[labels[max(labels.keys())][0]]

    print(classes[labels[max(labels.keys())][0]])
    #print(classes[max(labels.keys())])

    # show images
    result = img_copy
    x,y,w,h = 0,0,0,0
    #draw bounding box for predicted image
    filelist = glob.glob(os.path.join(os.getcwd(), 'model_test/up', "*.jpg"))
    for f in filelist:
        if str(max_file_name) in f:
            index = max_file_name
            break

    for i, rect in enumerate(boxes):
        if i == index:
            x, y, w, h = rect
            result = cv2.rectangle(img_copy, (x,y),(x+w,y+h), (0,255,0), 2)
            print(str(x)+' '+str(y)+' '+str(w)+' '+str(h))
            break


    cv2.putText(result, pred_result , (x+int(w/2), y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('Result', result)
    cv2.waitKey(0) # Wait for keypress to continue
    cv2.destroyAllWindows()

    savedir = os.path.join(os.getcwd(), 'model_test/up')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
            
    #test = os.listdir(savedir)
    #for f in test:
    #    if f.endswith(".jpg"):
    #        os.(os.path.join(savedir, f))

    filelist = glob.glob(os.path.join(savedir, "*.jpg"))
    for f in filelist:
        os.remove(f)

    cv2.imwrite(os.path.join(savedir, pred_result + '.jpg'), result)
    #print(pred_result)
    result_ID = dic_class[pred_result]
    return pred_result,result_ID




if __name__ == '__main__':
    target,result_ID = test()
    print(target,result_ID)
