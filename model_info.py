# this file is used to print the model information

from torchsummary import summary
import base_cnn
import torch

model = base_cnn.Large_CNN(100)
model = model.cuda()
summary(model, (3, 128, 128))