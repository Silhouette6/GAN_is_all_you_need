import os
import re
import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
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
model_name='resnet18_flower_baseline'
config = {'n_epochs': -1,
          'batch_size': 1,  # don't change
          'patch_size': 16,
          'num_workers': 0,  # don't change
          'lr': 3e-4,
          'lr_stable_epochs': 35,
          'lr_decay_epochs': 25,
          'dropout': 0.10092,
          'img_size': 224,
          'label_smoothing': 0.1,  # default 0.1
          'mix_ratio': 0.5964,
          'erase_ratio': 0.5,  # 0.1 -> 10%, don't change
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
    flower102_real = torchvision.datasets.Flowers102(root=root, split='test', download=False,
                                                transform=transform_real)
    flower102_mask = torchvision.datasets.Flowers102(root=root, split='test', download=False,
                                              transform=transform_mask)
    real_indices = list(range(0, config['n_sample']))
    masked_indices = list(range(0, config['n_sample']))
    real_img = Subset(flower102_real, real_indices)
    masked_img = Subset(flower102_mask, masked_indices)
    real_val_loader = DataLoader(real_img, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    masked_val_loader = DataLoader(masked_img, batch_size=batch_size, shuffle=False,
                                 drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    return real_val_loader, masked_val_loader


class ResBlockType1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        output = self.block(x)
        return torch.relu(output + x)


class ResBlockType2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)

    def forward(self, x):
        output = self.block(x)
        x = self.shortcut(x)
        return torch.relu(output + x)


class ResNet18(nn.Module):  # 构建一个类ResNet18架构的残差神经网络
    def __init__(self, dropout=config['dropout'], num_classes=102):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # 原汁臭味！
        )
        self.res_layer1 = nn.Sequential(ResBlockType1(64, 64),
                                        ResBlockType1(64, 64))
        self.res_layer2 = nn.Sequential(ResBlockType2(64, 128),
                                        ResBlockType1(128, 128))
        self.res_layer3 = nn.Sequential(ResBlockType2(128, 256),
                                        ResBlockType1(256, 256))
        self.res_layer4 = nn.Sequential(ResBlockType2(256, 512),
                                        ResBlockType1(512, 512))
        self.out_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),  # TODO:是否要加入这个Dropout层？
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.in_layer(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.res_layer3(x)
        x = self.res_layer4(x)
        x = self.out_layer(x)
        return x

if __name__ == '__main__':
    # ----------------------------------------------------
    # 设置新 log 目录
    log_root = './Log/CAM_res_log'

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

    resnet = ResNet18()
    resnet.load_state_dict(torch.load(f'./models/saved/{model_name}.pth',
                                    map_location='cpu'))
    resnet.to(device)
    cam_extractor = SmoothGradCAMpp(resnet)

    real_loader, mask_loader = get_dataloader(root='./datasets',
                                transform_mask=transform_mask,transform_real=transform_real, batch_size=config['batch_size'])
    n_sample = 1
    resnet.eval()

    for img, _ in tqdm(real_loader, desc="Processing Batches", unit="batch"):
        # print(img.shape)
        img = img.to(device)
        out = resnet(img)
        pred_class = out.argmax(dim=1).item()
        activation_map = cam_extractor(pred_class, out)
        img = (img.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0], mode='F'),
                                alpha=0.5)
        result_tensor = to_tensor(result)
        writer.add_image("SmoothGradCAM++ Result/real", result_tensor, n_sample)
        writer.add_image("Imgs/real", img, n_sample)
        n_sample += 1

    n_sample = 1
    for img, _ in tqdm(mask_loader, desc="Processing Batches", unit="batch"):
        # print(img.shape)
        img = img.to(device)
        out = resnet(img)
        pred_class = out.argmax(dim=1).item()
        activation_map = cam_extractor(pred_class, out)
        img = (img.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
        result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0], mode='F'),
                                alpha=0.5)
        result_tensor = to_tensor(result)
        writer.add_image("SmoothGradCAM++ Result/masked", result_tensor, n_sample)
        writer.add_image("Imgs/masked", img, n_sample)
        n_sample += 1

    writer.close()
    print('test done!')
    # --samples_per_plugin=images=1000
