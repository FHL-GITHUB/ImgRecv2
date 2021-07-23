from sklearn.model_selection import train_test_split
import os

#rootdir = os.listdir(os.path.join(os.getcwd(), '7007'))
rootdir= '/20210710Final/Tools/7019'

import os

import numpy as np

import shutil

#rootdir= '/content/drive/My Drive/images_original' #path of the original folder

classes = ['0', '6', '7', '8', '9', 'Down', 'Left', 'Right', 'Stop', 'Up', 'V', 'W', 'X', 'Y', 'Z']

for i in classes:
    os.makedirs(rootdir +'/train/' + i)
    #if not os.path.exists(os.path.join(os.getcwd(), 'train/'+i)):
    #    os.makedirs(os.path.join(os.getcwd(), 'train/'+i))
    #os.makedirs(rootdir +'/train/' + i)
    #if not os.path.exists(os.path.join(os.getcwd(), 'test/'+i)):
    #    os.makedirs(os.path.join(os.getcwd(), 'test/'+i))

    os.makedirs(rootdir +'/test/' + i)

    source = rootdir + '/' + i
    #print(source)

    allFileNames = os.listdir(source)

    np.random.shuffle(allFileNames)

    test_ratio = 0.25

    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                        [int(len(allFileNames)* (1 - test_ratio))])

    train_FileNames = [source+'/'+ name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]

    for name in train_FileNames:
        shutil.copy(name, rootdir +'/train/' + i)

    for name in test_FileNames:
        shutil.copy(name, rootdir +'/test/' + i)


