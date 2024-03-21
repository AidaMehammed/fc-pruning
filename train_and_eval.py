import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

INPUT_DIR = '/mnt/input'
OUTPUT_DIR = '/mnt/output'




class Clients:
    def __init__(self, model):
        self.model = model
        # self.dataloader = None
        self.train_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None

    def read_dataset(self):
        # Read the dataset and instantiate the model, dataloader, optimizer, loss, etc
        #train_dataset = pd.read_csv(f'{INPUT_DIR}/cifar_traindata.csv')
        # test_dataset = pd.read_csv(f'{INPUT_DIR}/cifar_testdata.csv')
        train_dataset = torch.load(f'{INPUT_DIR}/train_dataset.pth')
        test_dataset = torch.load(f'{INPUT_DIR}/test_dataset.pth')


        # print('Fehler 1')
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

        self.optimizer = optim.Adam(self.model.parameters(), 0.001)
        self.criterion = torch.nn.CrossEntropyLoss()


    def train_model(self, weights, epochs):
        # Set the weights
        self.set_weights(weights)

        # Perform the training loop
        # Get the weights and return
        for epoch in range(epochs):
            for images, labels in self.train_loader:

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                print(f'round:{epoch+1}, loss:{loss.item()}')
                #
        return self.get_weights()

    def acc(self):
        total_correct = 0
        total_samples = 0

        self.model.eval()

        with torch.no_grad():  # No gradient so we don't update our weights and biases with test data
            for X_test, Y_test in self.test_loader:  # Iterate through the test data
                outputs = self.model(X_test)
                predicted = torch.max(outputs, 1)[1]  # Adding up correct predictions
                total_samples += Y_test.size(0)
                total_correct += (predicted == Y_test).sum().item()  # Use .item() to get a scalar value

            # Calculate accuracy
            accuracy = total_correct / total_samples
            print(f'Accuracy: {accuracy:}')
        return accuracy

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)



def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')








def train(model, train_loader, epochs=2, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')
