import re
import cv2
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from eraser import PatchwiseRandomErasing
import random

random.seed(114514)

if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print(device)
else:
    device = torch.device('cpu')
    print(device)
model_name = 'com_c1'
config = {'n_epochs': -1,
          'batch_size': 1,  # don't change
          'num_workers': 0,  # don't change
          'lr': 7.4183e-05,
          'lr_stable_epochs': 35,
          'lr_decay_epochs': 25,
          'dropout': 0.10092,
          'patch_size': 16,
          'embed_dim': 768,
          'depth': 3,
          'n_heads': 6,
          'img_size': 224,
          'expansion_factor': 2,
          'label_smoothing': 0.1,
          'mix_ratio': 0.5694,
          'erase_ratio': 0.5,  # don't change
          'n_sample': 10  # don't change
          }
config['n_epochs'] = config['lr_stable_epochs'] + config['lr_decay_epochs']

transform_real = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
transform_mask = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                transforms.ToTensor(),
                                PatchwiseRandomErasing(patch_size=config['patch_size'], erase_ratio=config['erase_ratio'], mode='random'),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
to_tensor = transforms.ToTensor()


def get_dataloader(root, transform_real, transform_mask, batch_size):
    food101_real = torchvision.datasets.Food101(root=root, split='test', download=False,
                                                transform=transform_real)
    food101_mask = torchvision.datasets.Food101(root=root, split='test', download=False,
                                              transform=transform_mask)
    real_indices = list(range(0, config['n_sample']))
    masked_indices = list(range(0, config['n_sample']))
    real_img = Subset(food101_real, real_indices)
    masked_img = Subset(food101_mask, masked_indices)
    real_val_loader = DataLoader(real_img, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    masked_val_loader = DataLoader(masked_img, batch_size=batch_size, shuffle=False,
                                 drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    return real_val_loader, masked_val_loader


class PatchEmbed(nn.Module):
    def __init__(self, img_size=config['img_size'], patch_size=config['patch_size'], in_channels=3, embed_dim=config['embed_dim']):
        # embed_dim就是特征图的大小，num_patches是特征图的数目
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim,
                      kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )

    def forward(self, x):  # x:[batch_size, 3, 32, 32]
        x = self.conv_layer(x)  # [batch_size, embed_dim=384, num_patches=64]
        x = x.transpose(1, 2)  # [batch_size, num_patches=64, embed_dim=384]
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, embed_dim, expansion_factor=config['expansion_factor'], dropout=config['dropout']):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * expansion_factor, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)


class ConvolutionModule(nn.Module):
    def __init__(self, embed_dim, kernel_size=31):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim)
        self.pointwise_conv1 = nn.Conv1d(embed_dim, 2 * embed_dim, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size,
                                        padding=kernel_size // 2, groups=embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)
        self.pointwise_conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.drop = nn.Dropout(config['dropout'])

    def forward(self, x):  # x: [B, T, C]
        x_res = x
        x = self.ln(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)
        x = nn.functional.glu(x, dim=1)  # [B, C, T]
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = self.drop(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return x + x_res


class ConformerBlock(nn.Module):
    def __init__(self, embed_dim=config['embed_dim'],
                 num_heads=config['n_heads'], dropout=config['dropout']):
        super().__init__()
        self.ffn1 = FeedForwardModule(embed_dim)

        self.attn_ln = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)

        self.conv = ConvolutionModule(embed_dim)
        self.ffn2 = FeedForwardModule(embed_dim)

        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + 0.5 * self.ffn1(x)

        attn_input = self.attn_ln(x)
        attn_out, attn_weights = self.attn(attn_input, attn_input, attn_input, need_weights=True, average_attn_weights=False)
        x = x + self.attn_drop(attn_out)

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        x = self.final_ln(x)
        return x, attn_weights  # <<< 返回 attn 权重


class Conformer(nn.Module):
    def __init__(self, embed_dim=config['embed_dim'], depth=config['depth'], num_classes=101):
        super().__init__()
        self.in_layer = PatchEmbed()
        self.num_patches = self.in_layer.num_patches  # 获取num_patches的大小

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 将class token注册为参数，这个就是以后的结果
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))  # 将pos注册为参数，让模型自己学习特征之间的位置关系
        self.drop = nn.Dropout(config['dropout'])

        self.encoder_blocks = nn.Sequential(*[ConformerBlock() for _ in range(depth)])

        self.ln = nn.LayerNorm(embed_dim)
        self.out_layer = nn.Linear(in_features=embed_dim, out_features=num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.out_layer.weight, std=0.02)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x, return_attn=False):
        B = x.shape[0]
        x = self.in_layer(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop(x)

        attn_weights_all = []

        for block in self.encoder_blocks:
            x, attn = block(x)
            if return_attn:
                attn_weights_all.append(attn)

        x = self.ln(x)
        cls_token = x[:, 0]
        out = self.out_layer(cls_token)

        if return_attn:
            return out, attn_weights_all
        else:
            return out


def visualize_multihead_attention(attn_tensor, patch_size=16, log_writer=None, tag_prefix="attention", step=0):
    """
    attn_tensor: [n_heads, N, N] or [1, n_heads, N, N]
    log_writer: TensorBoard writer
    """
    if attn_tensor.dim() == 4:
        attn_tensor = attn_tensor[0]  # 去掉 batch 维度: [n_heads, N, N]

    num_heads = attn_tensor.shape[0]
    tokens = attn_tensor.shape[-1]
    num_patches = tokens - 1  # 去掉 cls token
    h = w = int(num_patches ** 0.5)  # 假设是正方形 patch

    cls_to_patch = attn_tensor[:, 0, 1:]  # 每个 head: cls -> patch 权重，[n_heads, 196]

    fig, axes = plt.subplots(1, num_heads + 1, figsize=(4 * (num_heads + 1), 4))

    # 平均注意力图
    mean_attn = cls_to_patch.mean(0).reshape(h, w).cpu()
    mean_attn = mean_attn.detach().numpy()
    axes[0].imshow(mean_attn, cmap='viridis')
    axes[0].set_title("Mean Attention")

    for i in range(num_heads):
        head_attn = cls_to_patch[i].reshape(h, w).cpu()
        head_attn = head_attn.detach().numpy()
        axes[i + 1].imshow(head_attn, cmap='viridis')
        axes[i + 1].set_title(f"Head {i}")

    for ax in axes:
        ax.axis("off")

    if log_writer:
        log_writer.add_figure(tag_prefix, fig, global_step=step)
    else:
        plt.tight_layout()
        plt.show()


def overlay_attention_on_image(img_tensor, attn_map_1d, alpha=0.6):
    """
    img_tensor: shape [3, H, W], torch.Tensor, 已经 Normalize 之后
    attn_map_1d: shape [196]，cls token → patch attention 权重
    return: PIL Image，叠加后的图
    """
    img = img_tensor.clone().cpu()
    img = img * 0.5 + 0.5  # 反标准化 [0,1]
    img = to_pil_image(img)

    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # attention map reshape 成 14x14
    num_patches = (config['img_size'] // config['patch_size'])
    attn_map = attn_map_1d.reshape(num_patches, num_patches).cpu()
    attn_map = attn_map.detach().numpy()

    # resize 到原图大小
    attn_resized = cv2.resize(attn_map, (img_cv.shape[1], img_cv.shape[0]))
    attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
    attn_heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)

    overlayed = cv2.addWeighted(attn_heatmap, alpha, img_cv, 1 - alpha, 0)

    return cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB)  # 返回 RGB 图

if __name__ == '__main__':
    # ----------------------------------------------------
    # 设置新 log 目录
    log_root = './Log/CAM_log'

    # 获取已有 exp 子目录列表
    existing_logs = [d for d in os.listdir(log_root) if
                    os.path.isdir(os.path.join(log_root, d)) and re.fullmatch(r'exp\d+', d)]
    exp_nums = [int(re.findall(r'\d+', name)[0]) for name in existing_logs]
    next_exp_num = max(exp_nums, default=0) + 1
    new_log_dir = os.path.join(log_root, f'exp{next_exp_num}')
    os.makedirs(new_log_dir, exist_ok=True)

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=new_log_dir)
    # ----------------------------------------------------

    conformer = Conformer()
    conformer.load_state_dict(torch.load(f'./models/{model_name}.pth', map_location='cpu'))
    conformer.eval().to(device)

    real_loader, mask_loader = get_dataloader(root='./datasets',
                                transform_mask=transform_mask,transform_real=transform_real, batch_size=config['batch_size'])
    n_sample = 1
    conformer.eval()

    for img, _ in tqdm(real_loader, desc="Processing Batches", unit="batch"):
        # print(img.shape)
        img = img.to(device)
        _, attn_maps = conformer(img, return_attn=True)

        # print(attn_maps, len(attn_maps))
        last_attn = attn_maps[-1][0]  # shape: (heads, N, N)
        # print(last_attn, last_attn.shape)
        visualize_multihead_attention(last_attn, patch_size=16,
                                    log_writer=writer, tag_prefix="Per head's attention/real", step=n_sample)

        # 得到叠加图
        cls_to_patch = last_attn.mean(0)[0, 1:]  # [196]
        img = (img.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        result = overlay_attention_on_image(img, cls_to_patch)
        result_tensor = to_tensor(result)
        writer.add_image("Avg attention map/real", result_tensor, n_sample)
        writer.add_image("Imgs/real", img, n_sample)

        n_sample += 1

    n_sample = 1
    for img, _ in tqdm(mask_loader, desc="Processing Batches", unit="batch"):
        # print(img.shape)
        img = img.to(device)
        _, attn_maps = conformer(img, return_attn=True)

        # print(attn_maps, len(attn_maps))
        last_attn = attn_maps[-1][0]  # shape: (heads, N, N)
        # print(last_attn, last_attn.shape)
        visualize_multihead_attention(last_attn, patch_size=16,
                                    log_writer=writer, tag_prefix="Per head's attention/masked", step=n_sample)

        # 得到叠加图
        cls_to_patch = last_attn.mean(0)[0, 1:]  # [196]
        img = (img.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        result = overlay_attention_on_image(img, cls_to_patch)
        result_tensor = to_tensor(result)
        writer.add_image("Avg attention map/masked", result_tensor, n_sample)
        writer.add_image("Imgs/masked", img, n_sample)

        n_sample += 1

    writer.close()
    print('test done!')
# --samples_per_plugin=images=1000
