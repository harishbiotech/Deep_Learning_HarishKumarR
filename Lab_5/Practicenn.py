import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28 * 28,512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512,512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512,10)


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        logits = self.linear3(x)
        return logits


def load_data():
    train_dataset = datasets.FashionMNIST(root='/home/ibab/Deep_Learning_HarishKumarR/Lab_3/data/',transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]),download=False,train=True)
    test_dataset = datasets.FashionMNIST(root='/home/ibab/Deep_Learning_HarishKumarR/Lab_3/data',train=False,download=False,transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]))
    return train_dataset, test_dataset


def train_network(mydataloader,model,loss_fn,optimizer,device,epochs):
    size = len(mydataloader.dataset)
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(mydataloader):
            X, y = X.to(device), y.to(device)
            # Forward pass
            pred = model(X)
            # Calculate loss
            loss = loss_fn(pred, y)
            # Backpropagation
            loss.backward()
            # Update weights
            optimizer.step()
            # Clear gradients
            optimizer.zero_grad()
            if batch % 100 == 0:
                loss = loss.item()
                current = (batch + 1) * len(X)
                print(f"Epoch: {epoch + 1}/{epochs} loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(mydataloader,model,loss_fn,device):
    size = len(mydataloader.dataset)
    num_batches = len(mydataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in mydataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred,y).item()
            correct += ((pred.argmax(1) == y).type(torch.float).sum().item())
    test_loss /= num_batches
    correct /= size
    print(f"Test Accuracy: {100 * correct:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")


def main():

    # Load dataset
    train_dataset, test_dataset = load_data()
    # Batch size
    batch_size = 64
    # Training DataLoader
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    # Test DataLoader
    test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    # Select CPU/GPU
    device = (torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")
    print(f"Using {device} device")
    # Create model
    model = NeuralNetwork().to(device)
    print(model)
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)
    # Train
    train_network(train_dataloader,model,loss_fn,optimizer,device=device,epochs=10)
    # Test
    test(test_dataloader,model,loss_fn,device)
    print("End")


if __name__ == "__main__":
    main()
# import torch
# from torch import nn, device, optim
# from torch.optim import optimizer
# from torch.utils.data import Dataset, DataLoader
# from torchvision import datasets
# from torchvision.transforms import v2
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear(28*28, 512)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(512, 512)
#         self.relu2 = nn.ReLU()
#         self.linear3 = nn.Linear(512, 10)
#     def forward(self, x):
#         x = self.flatten(x)
#         x = self.linear1(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         x = self.relu2(x)
#         logits = self.linear3(x)
#         return logits
# def load_data():
#     train_dataset = datasets.FashionMNIST(root='./data',transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32,scale=True)]),download=True,train=True)
#     test_data = datasets.FashionMNIST(root='./data',train=False,download=True,transform=v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32,scale=True)]))
#     return train_dataset, test_data
# def train_network(mydataloader, model,  loss_fn, device,epochs):
#     size = len(mydataloader.dataset)
#     for epoch in range(epochs):
#         model.train()
#         for batch, (X, y) in enumerate(mydataloader):
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             loss = loss_fn(pred, y)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             if batch % 100 == 0:
#                 loss=loss.item()
#                 current = (batch+1) * len(X)
#                 print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
# def test(mydataloader, model, loss_fn, device):
#     size = len(mydataloader.dataset)
#     num_batches = len(mydataloader)
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for X,y in mydataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
# def main():
#     train_dataset, test_dataset = load_data()
#     batch_size = 64
#     train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
#     devicee = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
#     print(f"using {device} device")
#     model = NeuralNetwork().to(device)
#     loss_fn = nn.CrossEntropyLoss()
#     optimizer = torch.optim.SGD(model.parameters(), lr=0.049)
#     train_network(train_dataloader, model, loss_fn,device=devicee,epochs=10)
#     test(test_dataloader,model,loss_fn,device)
#     print("End")
# if __name__ == "__main__":
#     main()