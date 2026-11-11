# # https://realpython.com/generative-adversarial-networks/
#
# import torch
# from torch import nn
#
# import math
# import matplotlib.pyplot as plt
#
# import torchvision
# import torchvision.transforms as transforms
#
#
# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(2, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         output = self.model(x)
#         return output
#
# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(2, 16),
#             nn.ReLU(),
#             nn.Linear(16, 32),
#             nn.ReLU(),
#             nn.Linear(32, 2),
#         )
#
#     def forward(self, x):
#         output = self.model(x)
#         return output
#
# def load_data_sim():
#     torch.manual_seed(111)
#
#     ### prepare training data
#     train_data_length = 1024
#     train_data = torch.zeros((train_data_length, 2))
#     train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
#     train_data[:, 1] = torch.sin(train_data[:, 0])
#     train_labels = torch.zeros(train_data_length)
#     train_set = [
#         (train_data[i], train_labels[i]) for i in range(train_data_length)
#     ]
#     ### plot training data
#     plt.plot(train_data[:, 0], train_data[:, 1], ".")
#     plt.show()
#
#     return train_set
#
# class Discriminator_mnist(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(784, 1024),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, x):
#         x = x.view(x.size(0), 784)
#         output = self.model(x)
#         return output
#
# class Generator_mnist(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(100, 256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 784),
#             nn.Tanh(),
#         )
#
#     def forward(self, x):
#         output = self.model(x)
#         output = output.view(x.size(0), 1, 28, 28)
#         return output
#
# def load_data_mnist():
#     torch.manual_seed(111)
#
#     transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
#     )
#
#     train_set = torchvision.datasets.MNIST(
#         root=".", train=True, download=True, transform=transform
#     )
#
#     return train_set
#
#
# def train_sim():
#     train_set = load_data_sim()
#
#     ### create a data loader
#     batch_size = 32
#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=batch_size, shuffle=True
#     )
#
#     ### instantiate a discriminator and generator object
#     discriminator = Discriminator()
#     generator = Generator()
#
#     ### setup parameters for training
#     # # for simulated data
#     lr = 0.001
#     num_epochs = 300
#
#     loss_function = nn.BCELoss()
#
#     optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
#     optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
#
#     ### training
#     for epoch in range(num_epochs):
#         for n, (real_samples, _) in enumerate(train_loader):
#             # Data for training the discriminator
#             real_samples_labels = torch.ones((batch_size, 1))
#             latent_space_samples = torch.randn((batch_size, 2))
#             generated_samples = generator(latent_space_samples)
#             generated_samples_labels = torch.zeros((batch_size, 1))
#             all_samples = torch.cat((real_samples, generated_samples))
#             all_samples_labels = torch.cat(
#                 (real_samples_labels, generated_samples_labels)
#             )
#
#             # Training the discriminator
#             discriminator.zero_grad()
#             output_discriminator = discriminator(all_samples)
#             loss_discriminator = loss_function(
#                 output_discriminator, all_samples_labels)
#             loss_discriminator.backward()
#             optimizer_discriminator.step()
#
#             # Data for training the generator
#             latent_space_samples = torch.randn((batch_size, 2))
#
#             # Training the generator
#             generator.zero_grad()
#             generated_samples = generator(latent_space_samples)
#             output_discriminator_generated = discriminator(generated_samples)
#             loss_generator = loss_function(
#                 output_discriminator_generated, real_samples_labels
#             )
#             loss_generator.backward()
#             optimizer_generator.step()
#
#             # Show loss
#             if epoch % 10 == 0 and n == batch_size - 1:
#                 print(f"Epoch: {epoch} Loss Descriminator: {loss_discriminator}")
#                 print(f"Epoch: {epoch} Loss Generator: {loss_generator}")
#
#     ### generate data with the trained model, GAN
#     latent_space_samples = torch.randn(100, 2)
#     generated_samples = generator(latent_space_samples)
#     generated_samples = generated_samples.detach()
#     plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
#     plt.show()
#
# def train_mnist():
#
#     ### use gpu if available
#     device = ""
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     else:
#         device = torch.device("cpu")
#
#     train_set = load_data_mnist()
#
#
#     ### load into dataloader
#     batch_size = 32
#     train_loader = torch.utils.data.DataLoader(
#         train_set, batch_size=batch_size, shuffle=True
#     )
#
#     # ### plot a few samples
#     # real_samples, mnist_labels = next(iter(train_loader))
#     # for i in range(16):
#     #     ax = plt.subplot(4, 4, i + 1)
#     #     plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
#     #     plt.xticks([])
#     #     plt.yticks([])
#     #     plt.show()
#
#     ### to device
#     discriminator = Discriminator_mnist().to(device=device)
#     generator = Generator_mnist().to(device=device)
#
#     ### prep for training
#     lr = 0.0001
#     num_epochs = 50
#     loss_function = nn.BCELoss()
#
#     optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
#     optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
#
#     for epoch in range(num_epochs):
#         for n, (real_samples, mnist_labels) in enumerate(train_loader):
#             # Data for training the discriminator
#             real_samples = real_samples.to(device=device)
#             real_samples_labels = torch.ones((batch_size, 1)).to(
#                 device=device
#             )
#             latent_space_samples = torch.randn((batch_size, 100)).to(
#                 device=device
#             )
#             generated_samples = generator(latent_space_samples)
#             generated_samples_labels = torch.zeros((batch_size, 1)).to(
#                 device=device
#             )
#             all_samples = torch.cat((real_samples, generated_samples))
#             all_samples_labels = torch.cat(
#                 (real_samples_labels, generated_samples_labels)
#             )
#
#             # Training the discriminator
#             discriminator.zero_grad()
#             output_discriminator = discriminator(all_samples)
#             loss_discriminator = loss_function(
#                 output_discriminator, all_samples_labels
#             )
#             loss_discriminator.backward()
#             optimizer_discriminator.step()
#
#             # Data for training the generator
#             latent_space_samples = torch.randn((batch_size, 100)).to(
#                 device=device
#             )
#
#             # Training the generator
#             generator.zero_grad()
#             generated_samples = generator(latent_space_samples)
#             output_discriminator_generated = discriminator(generated_samples)
#             loss_generator = loss_function(
#                 output_discriminator_generated, real_samples_labels
#             )
#             loss_generator.backward()
#             optimizer_generator.step()
#
#             # Show loss
#             if n == batch_size - 1:
#                 print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
#                 print(f"Epoch: {epoch} Loss G.: {loss_generator}")
#
#     ### checking for samples generated by the generator after training
#     latent_space_samples = torch.randn(batch_size, 100).to(device=device)
#     generated_samples = generator(latent_space_samples)
#     generated_samples = generated_samples.cpu().detach()
#     for i in range(16):
#         ax = plt.subplot(4, 4, i + 1)
#         plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
#         plt.xticks([])
#         plt.yticks([])
#
#     plt.show()
#
#
#
# def main():
#
#     train_sim()
#     # train_mnist()
#
#     return None
#
# ### Program begins here
# if __name__ == "__main__":
#     main()
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

torch.manual_seed(111)

# Training data: points on a sine curve (x, sin(x))
n = 1024
x = 2 * math.pi * torch.rand(n, 1)
y = torch.sin(x)
train_data = torch.cat((x, y), dim=1)      # shape (1024, 2)

plt.plot(train_data[:, 0], train_data[:, 1], ".")
plt.title("Real data")
plt.show()

# Generator: noise (2) -> point (2)
generator = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 2)
)

# Discriminator: point (2) -> real/fake probability
discriminator = nn.Sequential(
    nn.Linear(2, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1),
    nn.Sigmoid()
)

lr = 0.001
num_epochs = 300
batch_size = 32

loss_fn = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

real_labels = torch.ones(batch_size, 1)
fake_labels = torch.zeros(batch_size, 1)

for epoch in range(num_epochs):
    idx = torch.randperm(n)                       # shuffle data each epoch

    for i in range(0, n, batch_size):
        real_samples = train_data[idx[i:i + batch_size]]

        # Train Discriminator
        noise = torch.randn(batch_size, 2)
        fake_samples = generator(noise)

        all_samples = torch.cat((real_samples, fake_samples.detach()))
        all_labels = torch.cat((real_labels, fake_labels))

        d_loss = loss_fn(discriminator(all_samples), all_labels)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        noise = torch.randn(batch_size, 2)
        fake_samples = generator(noise)

        g_loss = loss_fn(discriminator(fake_samples), real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    if epoch % 10 == 0:
        print("Epoch", epoch, "D Loss:", d_loss.item(), "G Loss:", g_loss.item())

# Generate and plot new samples
generated = generator(torch.randn(100, 2)).detach()
plt.plot(generated[:, 0], generated[:, 1], ".")
plt.title("Generated data")
plt.show()
