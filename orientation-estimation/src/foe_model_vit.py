import torch
from torch import nn


# ImageToPatches returns multiple flattened square patches from an
# input image tensor.
class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    # end def

    def forward(self, x):
        assert len(x.size()) == 4
        y = self.unfold(x)
        y = y.permute(0, 2, 1)
        return y
    # end def
# end class


# The PatchEmbedding layer takes multiple image patches in (B,T,Cin) format
# and returns the embedded patches in (B,T,Cout) format.
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_size = embed_size
        # A single Layer is used to map all input patches to the output
        # embedding dimension, i.e. each image patch will share the weights of
        # this embedding layer.
        self.embed_layer = nn.Linear(in_features=in_channels,
                                     out_features=embed_size)
    # end def

    def forward(self, x):
        assert len(x.size()) == 3
        B, T, C = x.size()
        x = self.embed_layer(x)
        return x
    # end def
# end class


class VisionTransformerInput(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_size):
        """in_channels is the number of input channels in the input that will
        be fed into this layer. For RGB images, this value would be 3.
        """
        super().__init__()
        self.i2p = ImageToPatches(patch_size)
        self.pe = PatchEmbedding(patch_size * patch_size * in_channels,
                                 embed_size)
        num_patches = ((image_size[0] // patch_size) *
                       (image_size[1] // patch_size))
        # position_embed below is the learned embedding for the position of
        # each patch in the input image. They correspond to the cosine
        # similarity of embeddings visualized in the paper "An Image is Worth
        # 16x16 Words" https://arxiv.org/pdf/2010.11929.pdf (Figure 7, Center).
        self.position_embed = nn.Parameter(torch.randn(num_patches,
                                                       embed_size))
    # end def

    def forward(self, x):
        x = self.i2p(x)
        # print(x.shape)
        x = self.pe(x)
        x = x + self.position_embed
        return x
    # end def
# end class


# The MultiLayerPerceptron is a unit of computation. It expands the input
# to 4x the number of channels, and then contracts it back into the number
# of input channels. There's a GeLU activation in between, and the layer
# is followed by a droput layer.
class MultiLayerPerceptron(nn.Module):
    def __init__(self, embed_size, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(p=dropout),
        )
    # end def

    def forward(self, x):
        return self.layers(x)
    # end def
# end class


# This is a single self-attention encoder block, which has a multi-head
# attention block within it. The MultiHeadAttention block performs
# communication, while the MultiLayerPerceptron performs computation.
class SelfAttentionEncoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.embed_size = embed_size
        self.ln1 = nn.LayerNorm(embed_size)
        # self.kqv = nn.Linear(embed_size, embed_size * 3)
        self.mha = nn.MultiheadAttention(embed_size, num_heads,
                                         dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = MultiLayerPerceptron(embed_size, dropout)
    # end def

    def forward(self, x):
        y = self.ln1(x)
        # y = self.kqv(x)
        # (q, k, v) = torch.split(y, self.embed_size, dim=2)
        x = x + self.mha(y, y, y, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x
    # end def
# end class


# Similar to the PatchEmbedding class, we need to un-embed the representation
# of each patch that has been produced by our transformer network. We project
# each patch (that has embed_size) dimensions into patch_size * patch_size *
# output_dims channels, and then fold all the pathces back to make it look like
# an image.
class OutputProjection(nn.Module):
    def __init__(self, image_size, patch_size, embed_size, output_dims):
        super().__init__()
        self.patch_size = patch_size
        self.output_dims = output_dims
        self.projection = nn.Linear(embed_size,
                                    patch_size * patch_size * output_dims)
        self.fold = nn.Fold(output_size=(image_size[0]//8, image_size[1]//8),
                            kernel_size=patch_size, stride=patch_size)
    # end def

    def forward(self, x):
        B, T, C = x.shape
        x = self.projection(x)
        # x will now have shape (B, T, PatchSize**2 * OutputDims). This can be
        # folded into the desired output shape.

        # To fold the patches back into an image-like form, we need to first
        # swap the T and C dimensions to make it a (B, C, T) tensor.
        x = x.permute(0, 2, 1)
        x = self.fold(x)
        return x
    # end def
# end class


class VisionTransformerForSegmentation(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, out_channels,
                 embed_size, num_blocks, num_heads, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_size = embed_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout

        heads = [SelfAttentionEncoderBlock(embed_size, num_heads, dropout)
                 for i in range(num_blocks)]
        self.layers = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            VisionTransformerInput(image_size, patch_size, in_channels,
                                   embed_size),
            nn.Sequential(*heads),
            OutputProjection(image_size, 1, embed_size, out_channels)
        )
    # end def

    def forward(self, x):
        x = self.layers(x)
        return x
    # end def
# end class
