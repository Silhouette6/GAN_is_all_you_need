import os
import re
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
              'model_path': 'models/v2_0/vit_mutual_learning_55.pth',
              }

res_config = {'patch_size': 16,
              'num_workers': 8,
              'lr': 3e-4,
              'dropout': 0.10092,
              'label_smoothing': 0.1,
              'mix_ratio': 0.5964,
              'model_path': 'models/v2_0/resnet_mutual_learning_55.pth',
              }

config = {'n_epochs': -1,
          'lr_stable_epochs': 0,
          'lr_decay_epochs': 30,
          'batch_size': 256,
          'num_workers': 8,
          'patch_size': 16,
          'img_size': 224,
          'erase_ratio': 0,
          'threshold': 0.05,
          'train_res': True,
          'train_vit': False,
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


# ViT-framework
# Reference: Vit_food.py
class PatchEmbed(nn.Module):
    def __init__(self, img_size=config['img_size'], patch_size=vit_config['patch_size'], in_channels=3,
                 embed_dim=vit_config['embed_dim']):
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

class MLPBlock(nn.Module):
    def __init__(self, in_dim, dropout=vit_config['dropout'], mlp_ratio=vit_config['mlp_ratio']):
        # 在这个block中，特征数x的变化：x --> 4*x --> x
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=in_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=in_dim * mlp_ratio, out_features=in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class EncoderBlock(nn.Module):  # TODO:需要输出注意力权重，以便可视化注意力图
    def __init__(self, embed_dim=vit_config['embed_dim'],
                 num_heads=vit_config['n_heads'], dropout=vit_config['dropout']):
        # 在这个block中，输入与输出的形状是一样的
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.drop_1 = nn.Dropout(dropout)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, dropout=dropout)
        self.drop_2 = nn.Dropout(dropout)

    def forward(self, x):
        x_res = x
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.drop_1(attn_out)
        x = attn_out + x_res

        x_res = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = self.drop_2(x)
        x = x + x_res
        return x

class ViT(nn.Module):
    def __init__(self, embed_dim=vit_config['embed_dim'], depth=vit_config['depth'], num_classes=101):
        super().__init__()
        self.in_layer = PatchEmbed()
        self.num_patches = self.in_layer.num_patches  # 获取num_patches的大小

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 将class token注册为参数，这个就是以后的结果
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))  # 将pos注册为参数，让模型自己学习特征之间的位置关系
        self.drop = nn.Dropout(vit_config['dropout'])

        self.encoder_blocks = nn.Sequential(*[EncoderBlock() for _ in range(depth)])

        self.ln = nn.LayerNorm(embed_dim)
        self.out_layer = nn.Linear(in_features=embed_dim, out_features=num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.out_layer.weight, std=0.02)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x):
        B = x.shape[0]  # 获取batch_size大小
        x = self.in_layer(x)  # [B, N, C]，先embed，把图片处理成一维向量（实际上是二维，还有个batch_size）
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, C]，生成class token
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, C]，将class token加入资料，让模型学习
        # print(x.shape)
        # print(self.pos_embed.shape)
        x = x + self.pos_embed  # [B, N+1, C]  # 加入位置编码
        x = self.drop(x)  # 进encoder之前dropout一下

        x = self.encoder_blocks(x)  # [B, N+1, C]，过(depth)个encoder块

        x = self.ln(x)  # 出来之后先进行LayerNorm
        cls_token_extract = x[:, 0]  # [B, 1, C]，提取class token
        out = self.out_layer(cls_token_extract)  # [B, 101]，过全连接层，收束得到结果
        return out

# Net definition end



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
    resnet = ResNet18()
    resnet.load_state_dict(torch.load(res_config['model_path'], map_location='cpu'))
    resnet.to(device)

    vit = ViT()
    vit.load_state_dict(torch.load(vit_config['model_path'], map_location='cpu'))
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
    while epoch <= config['n_epochs']:
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
            if config['train_res']:
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

            if config['train_vit']:
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
                writer.add_scalar("Acc/Res_train", (res_total_train_acc / res_batch) * 100, epoch)
                writer.add_scalar("Acc/ViT_train", (vit_total_train_acc / vit_batch) * 100, epoch)
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
        if config['train_res']:
            avg_res_acc = res_total_val_acc / len(val_loader)
            if avg_res_acc > max_res_acc:
                max_res_acc = avg_res_acc
                torch.save(resnet.state_dict(), 'models/finetune/resnet_mutual_learning.pth')
                print(f'Res model has been saved, val_acc={avg_res_acc * 100:.2f}%, now epoch: {epoch}')
            if epoch == config['n_epochs']:
                torch.save(resnet.state_dict(), 'models/finetune/res_mutual_learning_backup.pth')
                print('Last training result has been backed up!')
        # ---保存ViT模型---
        if config['train_vit']:
            avg_vit_acc = vit_total_val_acc / len(val_loader)
            if avg_vit_acc > max_vit_acc:
                max_vit_acc = avg_vit_acc
                torch.save(vit.state_dict(), 'models/finetune/vit_mutual_learning.pth')
                print(f'ViT model has been saved, val_acc={avg_vit_acc * 100:.2f}%, now epoch: {epoch}')
            if epoch == config['n_epochs']:
                torch.save(vit.state_dict(), 'models/finetune/vit_mutual_learning_backup.pth')
                print('Last training result has been backed up!')
        # ---记录lr-
        print('-----------------------------------------------------------------------------------')
        writer.add_scalar("lr/Res", get_current_lr(optimizer_res), epoch)
        writer.add_scalar("lr/ViT", get_current_lr(optimizer_vit), epoch)

        scheduler_res.step()
        scheduler_vit.step()
        if epoch % 5 == 0:
            if config['train_res']:
                torch.save(resnet.state_dict(), f'models/finetune/resnet_mutual_learning_{epoch}.pth')
            if config['train_vit']:
                torch.save(vit.state_dict(),f'models/finetune/vit_mutual_learning_{epoch}.pth')
        epoch += 1
    writer.close()
    print('training done!')
