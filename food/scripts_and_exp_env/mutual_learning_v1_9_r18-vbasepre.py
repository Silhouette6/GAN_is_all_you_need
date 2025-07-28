import os
import re
import timm
import torch.nn as nn
import numpy as np
import torch
import torchvision
from collections import deque
from torch import optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from eraser import PatchwiseRandomErasing
# The file has been changed into a English version.
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    print(device)
else:
    device = torch.device('cpu')
    print(device)

# TODO: 'label_smoothing'当未被使用(已添加)
vit_config = {'num_workers': 8,
              'lr': 7.4183e-5,
              'dropout': 0.10092,
              'patch_size': 16,
              'embed_dim': 768,
              'depth': 3,
              'n_heads': 6,
              'mlp_ratio': 2,
              'label_smoothing': 0.1,
              'mix_ratio': 0.5964,
              'weight_decay': 1e-3,  # 于Vit_optuna_v2中被添加为搜索的参数之一
              'model_path': 'models/saved/vit_base_pretrain_baseline.pth',
              }

res_config = {'patch_size': 16,
              'num_workers': 8,
              'lr': 3e-4,
              'dropout': 0.10092,
              'label_smoothing': 0.1,
              'mix_ratio': 0.5964,
              'model_path': 'models/saved/resnet_18_pretrain_baseline.pth',
              }

config = {'n_epochs': -1,
          'lr_stable_epochs': 30,
          'lr_decay_epochs': 35,
          'batch_size': 256,
          'num_workers': 8,
          'patch_size': 16,
          'img_size': 224,
          'erase_ratio': 0,
          'threshold': 0.05,
          }
config['n_epochs'] = config['lr_stable_epochs'] + config['lr_decay_epochs']

transform_train = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      PatchwiseRandomErasing(patch_size=config['patch_size'],
                                                             erase_ratio=config['erase_ratio'], mode='random'),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
transform_val = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])


# ResNet18-framework
# Reference: Res_food.py
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
    def __init__(self, dropout=res_config['dropout'], num_classes=101):
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


def config_writer():
    table_str = f"""
    +-------------------+-----------------------
    | Hyperparameter    | Value
    +-------------------+-----------------------
    | Epochs            | {config['n_epochs']}
    | num_workers       | {config['num_workers']}
    | lr_stable_epochs  | {config['lr_stable_epochs']}
    | lr_decay_epochs   | {config['lr_decay_epochs']}
    | Batch size        | {config['batch_size']}
    | Patch_size        | {config['patch_size']}
    | Mix_ratio(vit/res)| {vit_config['mix_ratio'], res_config['mix_ratio']}
    | LR(vit/res)       | {vit_config['lr'], res_config['lr']}
    | Dropout(vit/res)  | {vit_config['dropout'], res_config['dropout']}
    | Img_size          | {config['img_size']}
    | Erase_ratio       | {config['erase_ratio']}
    +-------------------+-----------------------
    """
    return table_str


def get_dataloader(root, transform_train, transform_val, batch_size):
    food101_train = torchvision.datasets.Food101(root=root, split='train', download=True,
                                                 transform=transform_train)
    food101_val = torchvision.datasets.Food101(root=root, split='test', download=True,
                                               transform=transform_val)

    train_loader = DataLoader(food101_train, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(food101_val, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    return train_loader, val_loader


def lr_lambda_res(epoch, stable_epochs, decay_epochs):
    # 在前 stable_epochs 轮次，学习率保持不变
    if epoch < stable_epochs:
        return 1.0
    # 在接下来的 decay_epochs 轮次，逐步递减学习率
    elif epoch < stable_epochs + decay_epochs:
        return 1.0 - (epoch - stable_epochs) / decay_epochs
    else:
        return 0.0  # 学习率衰减到 0


def lr_lambda_vit(epoch, stable_epochs, decay_epochs):
    # 在前 stable_epochs 轮次，学习率保持不变
    if epoch < stable_epochs:
        return 1.0
    # 在接下来的 decay_epochs 轮次，逐步递减学习率
    elif epoch < stable_epochs + decay_epochs:
        return 1.0 - (epoch - stable_epochs) / decay_epochs
    else:
        return 0.0  # 学习率衰减到 0


def get_current_lr(optimizer):
    # 假设只有一个参数组，获取第一个参数组的学习率
    return optimizer.param_groups[0]['lr']


def get_acc(output, target):
    _, pred = torch.max(output, dim=1)
    correct = (pred == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


def mixup_data(x, y, alpha=1.0):
    # 返回混合后的图像和标签
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def create_log(log_root_name):
    # ----------------------------------------------------
    # 设置新 log 目录
    # log_root = 'mutual_learning_log'

    # 获取已有 exp 子目录列表
    existing_logs = [d for d in os.listdir(log_root_name) if
                     os.path.isdir(os.path.join(log_root_name, d)) and re.fullmatch(r'exp\d+', d)]
    exp_nums = [int(re.findall(r'\d+', name)[0]) for name in existing_logs]
    next_exp_num = max(exp_nums, default=0) + 1
    new_log_dir = os.path.join(log_root_name, f'exp{next_exp_num}')
    os.makedirs(new_log_dir, exist_ok=True)

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=new_log_dir)
    # ----------------------------------------------------
    return writer


if __name__ == '__main__':
    window_size = 10
    res_acc_window = deque(maxlen=window_size)
    vit_acc_window = deque(maxlen=window_size)
    '''
    resnet = torchvision.models.resnet18()
    resnet.fc = nn.Linear(resnet.fc.in_features, 101)  # 替换分类头为Food-101的类别数
    resnet.load_state_dict(torch.load(res_config['model_path'], map_location='cpu'))
    resnet.to(device)
    '''
    resnet = ResNet18().to(device)

    vit = timm.create_model(
        'vit_base_patch16_224.augreg_in1k',
        pretrained=True,
        num_classes=101  # 替换分类头为Food-101的类别数
    )
    # vit.load_state_dict(torch.load(vit_config['model_path'], map_location='cpu'))
    vit.to(device)

    res_ce_loss = nn.CrossEntropyLoss(label_smoothing=res_config['label_smoothing']).to(device)
    res_kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)
    vit_ce_loss = nn.CrossEntropyLoss(label_smoothing=vit_config['label_smoothing']).to(device)
    vit_kl_loss = nn.KLDivLoss(reduction='batchmean').to(device)

    optimizer_res = optim.AdamW(resnet.parameters(), lr=res_config['lr'], weight_decay=1e-3)
    optimizer_vit = optim.AdamW(vit.parameters(), lr=vit_config['lr'], weight_decay=vit_config['weight_decay'])
    scheduler_res = LambdaLR(optimizer_res,
                             lr_lambda=lambda epoch: lr_lambda_res(epoch, config['lr_stable_epochs'],
                                                                   config['lr_decay_epochs']))
    scheduler_vit = LambdaLR(optimizer_vit,
                             lr_lambda=lambda epoch: lr_lambda_vit(epoch, config['lr_stable_epochs'],
                                                                   config['lr_decay_epochs']))
    scaler = GradScaler()
    print('loading data...')
    train_loader, val_loader = get_dataloader(root='./datasets',
                                              transform_train=transform_train, transform_val=transform_val,
                                              batch_size=config['batch_size'])
    print('loading data successfully!')

    writer = create_log('Log/mutual_learning_log')
    writer.add_text('config', config_writer())
    epoch = 1
    max_res_acc = -1
    max_vit_acc = -1
    print('start training')
    mode = 'train_res'
    while epoch < config['n_epochs']:
        res_total_train_loss = 0
        res_total_train_acc = 0
        res_total_val_loss = 0
        res_total_val_acc = 0
        res_batch = 0

        vit_total_train_loss = 0
        vit_total_train_acc = 0
        vit_total_val_loss = 0
        vit_total_val_acc = 0
        vit_batch = 0

        local_res_acc = 0
        local_vit_acc = 0

        resnet.train()
        vit.train()
        for img, target in train_loader:
            # img_bak = img
            img, target = img.to(device), target.to(device)
            if mode == 'train_res':
                if epoch % 5 == 0 or epoch == 1:
                    with torch.no_grad():
                        resnet.eval()
                        output = resnet(img).detach()
                        res_total_train_acc += get_acc(output, target)
                        #############
                        del output  #
                        #############
                    resnet.train()
                img, target_a, target_b, lam = mixup_data(img, target, alpha=res_config['mix_ratio'])
                res_output = resnet(img)
                vit_output = vit(img).detach()  # 更严格的detach逻辑

                soft_targets = nn.functional.softmax(vit_output, dim=1).clamp(min=1e-6)  # 从vit中获取软标签，并softmax
                log_probs = nn.functional.log_softmax(res_output, dim=1)  # 对res的结果进行log_softmax，适应计算KL散度的需要

                res_loss = ((lam * res_ce_loss(res_output, target_a) + (1 - lam) * res_ce_loss(res_output, target_b))
                            + res_kl_loss(log_probs, soft_targets)) / 2  # 总loss = (CE_loss + KL_loss) / 2
                optimizer_res.zero_grad()
                res_loss.backward()
                optimizer_res.step()

                res_total_train_loss += res_loss.item()
                local_res_acc = get_acc(res_output, target)
                local_vit_acc = get_acc(vit_output, target)
                # 更新滑动窗口
                res_acc_window.append(local_res_acc)
                vit_acc_window.append(local_vit_acc)
                # 计算滑动平均
                avg_res_acc = sum(res_acc_window) / len(res_acc_window)
                avg_vit_acc = sum(vit_acc_window) / len(vit_acc_window)
                # 用滑动平均判断是否切换
                if avg_res_acc > avg_vit_acc + config['threshold']:
                    mode = 'train_vit'
                res_batch += 1

            elif mode == 'train_vit':
                optimizer_vit.zero_grad()
                with autocast():
                    if epoch % 5 == 0 or epoch == 1:
                        with torch.no_grad():
                            vit.eval()
                            output = vit(img).detach()
                            vit_total_train_acc += get_acc(output, target)
                            #############
                            del output  #
                            #############
                        vit.train()
                    img, target_a, target_b, lam = mixup_data(img, target, alpha=vit_config['mix_ratio'])
                    vit_output = vit(img)
                    res_output = resnet(img).detach()  # 更严格的detach逻辑
                    # print(img.shape, target.shape)
                    # print(output.shape)
                    soft_targets = nn.functional.softmax(res_output, dim=1).clamp(min=1e-6)   # 从res中获取软标签，并softmax, clamp 防止出现0
                    log_probs = nn.functional.log_softmax(vit_output, dim=1)  # 对vit的结果进行log_softmax，适应计算KL散度的需要

                    vit_loss = (((lam * vit_ce_loss(vit_output, target_a) + (1 - lam) * vit_ce_loss(vit_output,
                                                                                                    target_b)))
                                + vit_kl_loss(log_probs, soft_targets)) / 2  # 总loss = (CE_loss + KL_loss) / 2
                scaler.scale(vit_loss).backward()
                scaler.step(optimizer_vit)
                scaler.update()

                vit_total_train_loss += vit_loss.item()
                local_vit_acc = get_acc(vit_output, target)
                local_res_acc = get_acc(res_output, target)

                res_acc_window.append(local_res_acc)
                vit_acc_window.append(local_vit_acc)
                # 计算滑动平均
                avg_res_acc = sum(res_acc_window) / len(res_acc_window)
                avg_vit_acc = sum(vit_acc_window) / len(vit_acc_window)
                # 用滑动平均判断是否切换
                if avg_vit_acc > avg_res_acc + config['threshold']:
                    mode = 'train_res'
                vit_batch += 1
                
        # ---训练用图片取样---
        # writer.add_images('masked_imgs', img_bak, epoch)
        # ---Res数据记录---
        try:
            print(f'In current epoch, avg train Res_loss:{res_total_train_loss / res_batch},'
                  f'In current epoch, avg train Res_acc:{(res_total_train_acc / res_batch) * 100:.2f}%')
            writer.add_scalar("Loss/Res_train", (res_total_train_loss / res_batch), epoch)
        except Exception as e:
            print(f'error: {e}')
            print(f'Current Res data is incomplete! Current epoch: {epoch}')
        # ---ViT数据记录---
        try:
            print(f'In current epoch, avg train ViT_loss:{vit_total_train_loss / vit_batch},'
                  f'In current epoch, avg train ViT_acc:{(vit_total_train_acc / vit_batch) * 100:.2f}%')
            writer.add_scalar("Loss/ViT_train", (vit_total_train_loss / vit_batch), epoch)
        except Exception as e:
            print(f'error: {e}')
            print(f'Current ViT data is incomplete! Current epoch: {epoch}')
        # ---记录两个模型的训练比
        print(f'In current epoch, res_batch={res_batch}, vit_batch={vit_batch}')
        writer.add_scalar("Train_ratio/Res", (res_batch / len(train_loader)), epoch)
        writer.add_scalar("Train_ratio/ViT", (vit_batch / len(train_loader)), epoch)
        # ---逢5记录两个模型的acc---
        if epoch % 5 == 0 or epoch == 1:
            try:
                writer.add_scalar("Acc/ViT_train", (vit_total_train_acc / vit_batch) * 100, epoch)
                writer.add_scalar("Acc/Res_train", (res_total_train_acc / res_batch) * 100, epoch)
            except Exception as e:
                print(f'error: {e}')
                print(f'Current Acc data is incomplete! Current epoch: {epoch}')
        # ---验证两个模型---
        resnet.eval()
        vit.eval()
        with torch.no_grad(), autocast():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                res_output = resnet(img)
                vit_output = vit(img)

                res_loss = res_ce_loss(res_output, target)
                vit_loss = vit_ce_loss(vit_output, target)

                res_total_val_acc += get_acc(res_output, target)
                res_total_val_loss += res_loss.item()
                vit_total_val_acc += get_acc(vit_output, target)
                vit_total_val_loss += vit_loss.item()
        # ---Res数据记录---
        print(f'In current epoch, avg val Res_loss:{res_total_val_loss / len(val_loader)},'
              f'In current epoch, avg val Res_acc:{(res_total_val_acc / len(val_loader)) * 100:.2f}%')
        writer.add_scalar("Loss/Res_val", (res_total_val_loss / len(val_loader)), epoch)
        writer.add_scalar("Acc/Res_val", (res_total_val_acc / len(val_loader)) * 100, epoch)
        # ---ViT数据记录---
        print(f'In current epoch, avg val ViT_loss:{vit_total_val_loss / len(val_loader)},'
              f'In current epoch, avg val ViT_acc:{(vit_total_val_acc / len(val_loader)) * 100:.2f}%')
        writer.add_scalar("Loss/ViT_val", (vit_total_val_loss / len(val_loader)), epoch)
        writer.add_scalar("Acc/ViT_val", (vit_total_val_acc / len(val_loader)) * 100, epoch)
        # ---保存Res模型---
        avg_res_acc = res_total_val_acc / len(val_loader)
        if avg_res_acc > max_res_acc:
            max_res_acc = avg_res_acc
            torch.save(resnet.state_dict(), 'models/resnet_mutual_learning.pth')
            print(f'Res model has been saved, val_acc={avg_res_acc * 100:.2f}%, now epoch: {epoch}')
        if epoch == config['n_epochs']:
            torch.save(resnet.state_dict(), 'models/res_mutual_learning_backup.pth')
            print('Last training result has been backed up!')
        # ---保存ViT模型---
        avg_vit_acc = vit_total_val_acc / len(val_loader)
        if avg_vit_acc > max_vit_acc:
            max_vit_acc = avg_vit_acc
            torch.save(vit.state_dict(), 'models/vit_mutual_learning.pth')
            print(f'ViT model has been saved, val_acc={avg_vit_acc * 100:.2f}%, now epoch: {epoch}')
        if epoch == config['n_epochs']:
            torch.save(vit.state_dict(), 'models/vit_mutual_learning_backup.pth')
            print('Last training result has been backed up!')
        # ---记录lr-
        print('-----------------------------------------------------------------------------------')
        writer.add_scalar("lr/Res", get_current_lr(optimizer_res), epoch)
        writer.add_scalar("lr/ViT", get_current_lr(optimizer_vit), epoch)

        scheduler_res.step()
        scheduler_vit.step()
        epoch += 1
    writer.close()
    print('training done!')
