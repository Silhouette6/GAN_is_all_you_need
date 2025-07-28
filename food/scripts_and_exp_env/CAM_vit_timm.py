import re
import cv2
import timm
import torch
import torchvision
from timm.models.vision_transformer import Attention
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
    device = torch.device('cuda:0')
    print(device)
else:
    device = torch.device('cpu')
    print(device)
model_name='vit_mutual_learning_11_r18-vbasepre'
config = {'n_epochs': -1,
          'batch_size': 1,  # don't change
          'num_workers': 0,  # don't change
          'lr': 0.00012715,
          'lr_stable_epochs': 35,
          'lr_decay_epochs': 25,
          'dropout': 0.15182,
          'patch_size': 16,
          'embed_dim': 768,
          'depth': 12,
          'n_heads': 12,
          'img_size': 224,
          'mlp_ratio': 4,
          'label_smoothing': 0.1,
          'mix_ratio': 0.30718,
          'erase_ratio': 0.5,  # don't change
          'n_sample': 50  # don't change
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


class AttentionWithReturn(Attention):
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_probs = attn.softmax(dim=-1)
        self.latest_attn = attn_probs  #  保存注意力权重

        attn_probs = self.attn_drop(attn_probs)
        out = (attn_probs @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


if __name__ == '__main__':
    # ----------------------------------------------------
    # 设置新 log 目录
    log_root = './Log/CAM_vit_log'

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

    vit = timm.create_model(
        'vit_base_patch16_224.augreg_in1k',
        pretrained=False,
        num_classes=101
    )
    vit.load_state_dict(torch.load(f'./models/saved/{model_name}.pth', map_location='cpu'))
    old_attn = vit.blocks[-1].attn
    new_attn = AttentionWithReturn(
        dim=old_attn.qkv.in_features,
        num_heads=old_attn.num_heads,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0
    )
    new_attn.load_state_dict(old_attn.state_dict())
    vit.blocks[-1].attn = new_attn

    vit.to(device)
    vit.eval()

    real_loader, mask_loader = get_dataloader(root='./datasets',
                                transform_mask=transform_mask,transform_real=transform_real, batch_size=config['batch_size'])
    n_sample = 1
    for img, _ in tqdm(real_loader, desc="Processing Batches", unit="batch"):
        # print(img.shape)
        img = img.to(device)
        _ = vit(img)

        attn_probs = vit.blocks[-1].attn.latest_attn
        # print(attn_probs, attn_probs.shape)
        last_attn = attn_probs[-1]  # shape: (heads, N, N)
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
        _ = vit(img)

        attn_probs = vit.blocks[-1].attn.latest_attn
        # print(attn_probs, attn_probs.shape)
        last_attn = attn_probs[-1]  # shape: (heads, N, N)
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
