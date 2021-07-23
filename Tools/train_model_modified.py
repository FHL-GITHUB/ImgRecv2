import numpy as np
from torch import optim
import torch, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from efficientnet_pytorch import EfficientNet
from random import randint
import torch.nn.functional as F

def calcAccuracy(scores, label):
    _, prediction = torch.max(scores.cpu(), dim=1)
    return torch.tensor(torch.sum(prediction == label.cpu()).cpu().item() / len(scores))

# Cross validate
def validate(validate_ds, model, softmax, device):
    validate_length = 0
    accuracy = 0
    for img, lbl in validate_ds:
        img = img.to(device)
        lbl = lbl.to(device)
        scores = model(img)
        loss = softmax(scores, lbl)
        accuracy += calcAccuracy(scores, lbl)
        validate_length += 1
    accuracy /= validate_length
    return loss, accuracy
    #return accuracy

# Run the training and cross validation
def fit(train_ds, validate_ds, no_epochs, optimizer, model, device):
    history = []
    valid_acr_compare = 0
    softmax = nn.CrossEntropyLoss()
    try:
        for index in range(no_epochs):
            #torch.cuda.empty_cache()
            # Train
            for img, lbl in train_ds:
                img = img.to(device)
                lbl = lbl.to(device)
                scores = model(img)
                loss = softmax(scores, lbl)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Validate
            valid_loss, valid_acr = validate(validate_ds, model, softmax, device)
            #valid_acr = validate(validate_ds, model, softmax, device)

            # Print epoch record
            print(f"Epoch [{index + 1}/{no_epochs}] => loss: {loss}, val_loss: {valid_loss}, val_acc: {valid_acr}")
            if valid_acr > valid_acr_compare:
                torch.save(model.state_dict(), "./efficient_net.pt")
                valid_acr_compare = valid_acr

            history.append({"loss": loss,
                            "valid_loss": valid_loss,
                            "valid_acr": valid_acr
                            })
            #history.append({"valid_acr": valid_acr})
        del train_ds,validate_ds
        return history
    except:
        del train_ds,validate_ds
        return history


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class GPUDataLoader():
    def __init__(self, ds, device):
        self.ds = ds
        self.device = device

    def __iter__(self):
        for batch in self.ds:
            yield to_device(batch, self.device)

def run():
    torch.cuda.empty_cache()
    
    torch.multiprocessing.freeze_support()

    TRAIN_PATH = os.path.join(os.getcwd(), '7019/train')
    VALIDATE_PATH = os.path.join(os.getcwd(), '7019/test')

    transform = Compose([ToTensor(), Resize((32, 32))])
    #transform = Compose([ToTensor(), Resize((224, 224)), Grayscale(3)])
    # Load train data and test data
    #transform = Compose([ToTensor(), Resize((32, 32)), Grayscale(3)])

    train_data = ImageFolder(root=TRAIN_PATH, transform=transform)
    validate_data = ImageFolder(root=VALIDATE_PATH, transform=transform)

    print("Train dataset has {0} images".format(len(train_data)))

    # View image size and class
    fst_img, fst_lbl = train_data[0]
    print("First image has size: {0} and class: {1}.".format(fst_img.shape, fst_lbl))

    sc_img, sc_lbl = train_data[randint(0,len(train_data))]
    print("Another random image has size: {0} and class: {1}.".format(sc_img.shape, sc_lbl))

    # View all classes
    classes = train_data.classes
    print("There are {0} classes in total: ".format(len(classes)))
    print(classes)

    train_ds = DataLoader(train_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=8)
    validate_ds = DataLoader(validate_data, batch_size=32, shuffle=True, pin_memory=True, num_workers=8)

    model = EfficientNet.from_name('efficientnet-b0')  #efficientnet-b3
    # model = SimpleNet()
    feature = model._fc.in_features
    print('hererererererer')
    print(feature)
    model._fc = nn.Sequential(nn.Linear(model._fc.in_features, 512), 
                                           nn.ReLU(),  
                                           nn.Dropout(0.25),
                                           nn.Linear(512, 128), 
                                           nn.ReLU(),  
                                           nn.Dropout(0.50), 
                                           nn.Linear(128,15))
    #model._fc = nn.Linear(1280, 15)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output_params = list(map(id, model._fc.parameters()))
    feature_params = filter(lambda p: id(p) not in output_params, model.parameters())
    lr = 0.01
    optimizer = optim.SGD([{'params': feature_params},
                           {'params': model._fc.parameters(), 'lr': lr * 10}],
                          lr=lr, weight_decay=0.001)
    # train_ds = GPUDataLoader(train_ds, device)
    # validate_ds = GPUDataLoader(validate_ds, device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    no_epochs = 55
    history = fit(train_ds, validate_ds, no_epochs, optimizer, model, device)

    train_loss = []
    valid_loss = []
    valid_acr = []
    for x in history:
        train_loss.append(x["loss"])
        valid_loss.append(x["valid_loss"])
        valid_acr.append(x["valid_acr"])

    train_loss = [x.item() for x in train_loss]
    valid_loss = [x.item() for x in valid_loss]
    valid_acr = [x.item() for x in valid_acr]
    epochs = np.arange(no_epochs)

    torch.save(model.state_dict(), "./trained_model.pt",_use_new_zipfile_serialization=False)

    



if __name__ == '__main__':
    run()
