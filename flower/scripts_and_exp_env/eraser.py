import torch
import torch.nn as nn
import random


class PatchwiseRandomErasing(nn.Module):
    def __init__(self, patch_size=16, erase_ratio=0.3, mode='random', constant_value=0.0, color_value=None, random_ratio=False):
        """
        :param patch_size: 每个 patch 的大小
        :param erase_ratio: 被遮挡 patch 的比例（0~1）
        :param constant_value: mode='constant' 时使用的值
        :param mode: 选择被选中patch的遮挡方式 'random'、'black'、'constant'、'color'
        :param color_value: mode='color' 时使用的 RGB 值 (tuple/list of 3 floats)
        :param random_ratio: 是否在每次 forward 时随机生成 erase_ratio，范围是 0~erase_ratio

        Example:
        # 使用全黑擦除
        eraser = PatchwiseRandomErasing(mode='black')
        # 使用 RGB 颜色擦除（如偏棕色）
        eraser = PatchwiseRandomErasing(mode='color', color_value=[0.5, 0.3, 0.1])
        # 使用常数值擦除（灰度0.7）
        eraser = PatchwiseRandomErasing(mode='constant', constant_value=0.7)
        # 使用像素随机擦除（最接近 RandomErasing 原始逻辑）
        eraser = PatchwiseRandomErasing(mode='random')
        """
        super().__init__()
        self.patch_size = patch_size
        self.max_erase_ratio = erase_ratio  # 保存最大擦除比例
        self.random_ratio = random_ratio
        self.mode = mode
        self.constant_value = constant_value
        self.color_value = color_value

    def forward(self, img):
        C, H, W = img.shape
        ph, pw = self.patch_size, self.patch_size
        nh, nw = H // ph, W // pw
        total_patches = nh * nw
        
        # 如果random_ratio为True，则在每次forward时随机生成erase_ratio
        current_ratio = random.uniform(0, self.max_erase_ratio) if self.random_ratio else self.max_erase_ratio
        num_erase = int(total_patches * current_ratio)

        indices = random.sample(range(total_patches), num_erase)

        for idx in indices:
            i = idx // nw
            j = idx % nw
            top = i * ph
            left = j * pw

            if self.mode == 'random':
                erase_val = torch.rand((C, ph, pw), dtype=img.dtype, device=img.device)
            elif self.mode == 'black':
                erase_val = torch.zeros((C, ph, pw), dtype=img.dtype, device=img.device)
            elif self.mode == 'constant':
                erase_val = torch.full((C, ph, pw), self.constant_value, dtype=img.dtype, device=img.device)
            elif self.mode == 'color':
                if self.color_value is None or len(self.color_value) != C:
                    raise ValueError("color_value must be a tuple/list with length matching channel count.")
                erase_val = torch.tensor(self.color_value, dtype=img.dtype, device=img.device).view(C, 1, 1).expand(C, ph, pw)
            else:
                raise ValueError("Unsupported mode: choose from 'random', 'black', 'constant', 'color'.")

            img[:, top:top+ph, left:left+pw] = erase_val

        return img
