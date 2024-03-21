import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization




class Model(nn.Module):
    def __init__(self, quantize=False):
        super().__init__()
        self.quantize = quantize

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # FloatFunction()
        self.skip_add = nn.quantized.FloatFunctional()
        # for inputs
        self.quant = torch.quantization.QuantStub()
        # for outputs
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        if self.quantize:
          x = self.quant(x)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.quantize:
          x = self.dequant(x)

        return x

    def get_weights(self):
      return [param.data for param in self.parameters()]


    def set_weights(self, weights):
      for param, weight in zip(self.parameters(), weights):
            param.data = weight


class Model_MNIST(nn.Module):
    def __init__(self, quantize=False):
        super(Model_MNIST, self).__init__()
        self.quantize = quantize

        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(5 * 5 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # FloatFunction()
        self.skip_add = nn.quantized.FloatFunctional()
        # for inputs
        self.quant = torch.quantization.QuantStub()
        # for outputs
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, X):
       # Input are quantized
        if self.quantize:
            X = self.quant(X)
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = torch.flatten(X, 1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        # Outputs are dequantized
        if self.quantize:
            X = self.dequant(X)
        return X # softmax hier unnötig da in loss function schon inkludiert


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

        # First Conv layer
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)

        # Second Conv layer
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)

        # Third, fourth, fifth convolution layer
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.3)

        # Fully Connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout5 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(0.5)

        self.fc4 = nn.Linear(128, 10)

    def get_weights(self):
        return [param.data for param in self.parameters()]

    def set_weights(self, weights):
        for param, weight in zip(self.parameters(), weights):
            param.data = weight

    def forward(self, x):
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Flatten the output for fully connected layers
        x = x.view(-1, 256 * 4 * 4)

        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout4(x)
        x = F.relu(self.fc2(x))
        x = self.dropout5(x)
        x = F.relu(self.fc3(x))
        x = self.dropout6(x)

        # Output layer
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)



def get_weights(model):
    return [param.data for param in model.parameters()]

def set_weights(model, weights):
  for param, weight in zip(model.parameters(), weights):
        param.data = weight