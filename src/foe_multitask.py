import random
import torch
from torch import nn
from matplotlib import pyplot as plt


class FOE_AUTOENCODER(nn.Module):
    def __init__(self, path, patch_size, encoded_dim, device):
        super().__init__()

        self.path = path
        self.patch_size = patch_size
        self.encoded_dim = encoded_dim
        # self.out_dim = out_dim
        self.device = device

        # Convolutional section (W-K+2P)/S + 1 => W/2 => W/8
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, 4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, stride=2, padding=1),
            # nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.Flatten(start_dim=1),
            nn.Linear(patch_size**2 * 4, encoded_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, patch_size**2 * 16),
            nn.Unflatten(dim=1, unflattened_size=(1024,
                         patch_size//8, patch_size//8)),
            nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            nn.Tanh()
        )

        self = self.to(device)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
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
