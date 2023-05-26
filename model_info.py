# this file is used to print the model information

from torchsummary import summary
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import base_cnn
import torch

model = efficientnet_v2_s(pretrained=True)
model = model.cuda()
summary(model, (3, 32, 32))