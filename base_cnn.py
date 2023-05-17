import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Base_CNN(nn.Module):
    def __init__(self, class_num):
        super(Base_CNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1) # 30
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1) # 28
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2) # 14
        self.conv4 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1) #12
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1) # 10
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, class_num)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(-1, 24 * 10 * 10)
        x = self.fc1(x)
        return x