import random
import torch
from torch import nn
from matplotlib import pyplot as plt


class Encoder(nn.Module):
    def __init__(self, patch_size, encoded_space_dim):
        super().__init__()

        # Convolutional section (W-K+2P)/S + 1 => W/2 => W/8
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 4, stride=2, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(8, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(True)
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(patch_size**2 // 2, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):

    def __init__(self, encoded_space_dim, patch_size):
        super().__init__()

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, patch_size**2 // 2),
            nn.LeakyReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(32,
                                                               patch_size//8,
                                                               patch_size//8))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class FOE_AE(nn.Module):
    def __init__(self, path, inp_dim, out_dim, device):
        super().__init__()

        self.path = path
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.device = device
        self.encoder = Encoder(inp_dim, out_dim).to(device)
        self.decoder = Decoder(out_dim, inp_dim).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def plot_outputs(self, vset, num_images):
        _, axs = plt.subplots(2, num_images)

        for i in range(num_images):
            idx = random.randint(0, len(vset))
            img = vset[idx][0].unsqueeze(0).to(self.device)
            axs[0, i].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[0, i].get_xaxis().set_visible(False)
            axs[0, i].get_yaxis().set_visible(False)
            if i == num_images//2:
                axs[0, i].set_title('Original images')

            self.eval()
            with torch.no_grad():
                rec_img = self.decoder(self.encoder(img))
            axs[1, i].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
            axs[1, i].get_xaxis().set_visible(False)
            axs[1, i].get_yaxis().set_visible(False)
            if i == num_images//2:
                axs[1, i].set_title('Reconstructed images')
        plt.show()
