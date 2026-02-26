import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

def get_gaussian_kernel(kernel_size=11, sigma=3.0):
    ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

class GaussianBlurConv(nn.Module):
    def __init__(self, kernel_size=11, sigma=3.0):
        super().__init__()
        kernel = get_gaussian_kernel(kernel_size, sigma)
        self.register_buffer('weight', kernel[None, None, :, :])
        self.ks = kernel_size
        self.pad = kernel_size // 2

    def forward(self, x):
        B, C, H, W = x.shape
        weight = self.weight.repeat(C, 1, 1, 1)
        x = F.conv2d(x, weight, padding=self.pad, groups=C)
        return x

class ConvSharpnessQKVAttention(nn.Module):
    def __init__(self, embed_dim=64, kernel_size=5, sigma=3.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.ks = kernel_size
        self.sigma = sigma

        # 高斯模糊
        self.gaussian = GaussianBlurConv(kernel_size, sigma)

        # Q/K/V
        self.to_q = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(1, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )

        # 局部卷积注意力
        self.local_conv = nn.Conv2d(embed_dim, embed_dim,
                                    kernel_size=kernel_size,
                                    padding=kernel_size//2,
                                    groups=embed_dim,
                                    bias=False)

        # MLP 投影回像素空间
        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, 1, 1)
        )
        # 残差缩放参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, I):
        B, C, H, W = I.shape

        # 高斯模糊
        K_blur = self.gaussian(I)  # (B, C, H, W)

        # Q/K/V
        Q = self.to_q(I)         # (B, embed_dim, H, W)
        K = self.to_k(K_blur)    # (B, embed_dim, H, W)
        V = self.to_v(I)         # (B, embed_dim, H, W)

        # 局部卷积注意力
        K_local = self.local_conv(K)       # (B, embed_dim, H, W)
        attn = Q * K_local                 # 点积
        attn = torch.softmax(attn,dim=1)

        # 软增强
        sharp_feat = V * (1 - self.beta * attn)

        # 投影
        enhanced = self.proj(sharp_feat)
        # 残差保护原图
        out = I + self.gamma * enhanced

        return out

if __name__ == "__main__":

    img1 = Image.open("1-1.jpg").convert("L")  # 灰度
    img2 = Image.open("1-2.jpg").convert("L")


    transform = T.Compose([
        T.ToTensor()
    ])
    x1 = transform(img1).unsqueeze(0)
    x2 = transform(img2).unsqueeze(0)


    model = ConvSharpnessQKVAttention(embed_dim=32, kernel_size=7, sigma=2.0)
    model.eval()  # 测试模式

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    def show_tensor_image(tensor, title=""):
        img = tensor.squeeze().cpu().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()


    show_tensor_image(x1, title="Input Image 1")
    show_tensor_image(out1, title="Attention Output 1")

    show_tensor_image(x2, title="Input Image 2")
    show_tensor_image(out2, title="Attention Output 2")