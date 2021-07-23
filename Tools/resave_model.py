import torch
from efficientnet_pytorch import EfficientNet
from torchvision.transforms import Compose, ToTensor, Resize, Grayscale
import torch.nn as nn

def do_nothing():
  pass

if __name__ == '__main__':
    net = EfficientNet.from_pretrained('efficientnet-b0')
    net._fc = nn.Sequential(nn.Linear(net._fc.in_features, 512), 
                                           nn.ReLU(),  
                                           nn.Dropout(0.25),
                                           nn.Linear(512, 128), 
                                           nn.ReLU(),  
                                           nn.Dropout(0.50), 
                                           nn.Linear(128,16))
    net.load_state_dict(torch.load(r'trained_model.pt', map_location=torch.device('cpu')))
    net.eval()
    torch.save(net, "./model_with_color.pt")
