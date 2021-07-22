import torch,cv2,glob,os
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import ImageFolder
from extract import extractBBox
from pytorch_image_folder_with_file_paths import ImageFolderWithPaths as ImageFolderPath

import torch.nn as nn
import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x

    return y
    
def test():
    img = cv2.imread("./predict_result/result/target.jpg")
    img1 = img
    boxes,bb,images,img_copy=extractBBox(img)  

    result_save_dir = os.path.join(os.getcwd(), 'predict_result')

    classes = ['0', '6', '7', '8', '9', 'Down', 'Left', 'Right', 'Stop', 'Up', 'V', 'W', 'X', 'Y', 'Z']
    #dic_class = {'Up':1,'Down':2,'Right':3,'Left':4,'Stop':5,'6':6,'7':7,'8':8,'9':9,'0':10,'V':11,'W':12,'X':13,'Y':14,'Z':15,'Bull':0}
    dic_class = {'Up':1,'Down':2,'Right':3,'Left':4,'Stop':5,'6':6,'7':7,'8':8,'9':9,'0':10,'V':11,'W':12,'X':13,'Y':14,'Z':15}
    net = torch.load("model_with_color.pt")
    net.eval()
    #TRAIN_PATH = os.path.join(os.getcwd(), 'train')
    VALIDATE_PATH = os.path.join(os.getcwd(), 'predict_result')

    transform = Compose([ToTensor(), Resize((32, 32))])
    # Load image data

    image_data = ImageFolderPath(root=VALIDATE_PATH, transform=transform)

    image_ds = DataLoader(image_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    labels = {}
    count = 0

    for img, lbl,path in image_ds:
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
    #print(max(labels.keys()))
    score = max(labels.keys())
    max_file_name = labels[max(labels.keys())][1]
    print(max_file_name)
    pred_result = classes[labels[max(labels.keys())][0]]

    print(classes[labels[max(labels.keys())][0]])
    
    # show images
    result = img_copy
    x,y,w,h = 0,0,0,0
    #draw bounding box for predicted image
    filelist = glob.glob(os.path.join(os.getcwd(), 'predict_result/result', "*.jpg"))
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
    #cv2.imshow('Result', result)
    #cv2.waitKey(0
    #cv2.destroyAllWindows()

    savedir = os.path.join(os.getcwd(), 'predict_result/result')
    if not os.path.exists(savedir):
        os.makedirs(savedir)

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
