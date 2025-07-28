import json
import numpy as np
import optuna
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.cuda.amp import autocast, GradScaler
from eraser import PatchwiseRandomErasing
import logging

# 配置optuna日志输出
optuna.logging.get_logger("optuna").addHandler(
    logging.FileHandler("Vit_optuna_log.txt")
)

# 设备
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


# 固定transform（可以后续扩展一起调参）
def get_transforms(img_size, erase_ratio):
    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        PatchwiseRandomErasing(patch_size=16, erase_ratio=erase_ratio, mode='random'),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    transform_val = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform_train, transform_val


# 目标函数
def objective(trial):
    # 超参数搜索空间
    # batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    batch_size = 256
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    patch_size = trial.suggest_categorical('patch_size', [8, 16])  # 移除了32
    embed_dim = trial.suggest_categorical('embed_dim', [192, 384, 576, 768])
    valid_heads = [h for h in [2, 3, 4, 6, 8, 12] if embed_dim % h == 0]  # 增加了自适应头数，确保能整除
    depth = trial.suggest_int('depth', 2, 6)
    n_heads = trial.suggest_categorical('n_heads', valid_heads)
    mlp_ratio = trial.suggest_categorical('mlp_ratio', [2, 3, 4])
    mix_ratio = trial.suggest_float('mix_ratio', 0.2, 0.6)
    # erase_ratio = trial.suggest_float('erase_ratio', 0.1, 0.5)
    erase_ratio = 0
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 5e-2, log=True)  # 增加了搜索weight_decay的环节，范围是ViT官方代码（如Touvron et al., DeiT）中的经典配置

    config = {
        'n_epochs': 10,  # 调参时只训练少量epoch，加速
        'batch_size': batch_size,
        'num_workers': 4,
        'lr': lr,
        'lr_stable_epochs': 7,
        'lr_decay_epochs': 3,
        'dropout': dropout,
        'patch_size': patch_size,
        'embed_dim': embed_dim,
        'depth': depth,
        'n_heads': n_heads,
        'img_size': 224,
        'mlp_ratio': mlp_ratio,
        'label_smoothing': 0.1,
        'mix_ratio': mix_ratio,
        'erase_ratio': erase_ratio,
        'weight_decay': weight_decay,
    }

    # 加载数据
    def get_dataloader(root, transform_train, transform_val, batch_size, num_workers):
        food101_train = torchvision.datasets.Food101(root=root, split='train', download=False,
                                                     transform=transform_train)
        food101_val = torchvision.datasets.Food101(root=root, split='test', download=False,
                                                   transform=transform_val)

        train_loader = DataLoader(food101_train, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(food101_val, batch_size=batch_size, shuffle=False,
                                drop_last=True, num_workers=num_workers, pin_memory=True)
        return train_loader, val_loader

    class PatchEmbed(nn.Module):
        def __init__(self, img_size=config['img_size'], patch_size=config['patch_size'], in_channels=3,
                     embed_dim=config['embed_dim']):
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
        def __init__(self, in_dim, dropout=config['dropout'], mlp_ratio=config['mlp_ratio']):
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
        def __init__(self, embed_dim=config['embed_dim'],
                     num_heads=config['n_heads'], dropout=config['dropout']):
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
        def __init__(self, embed_dim=config['embed_dim'], depth=config['depth'], num_classes=101):
            super().__init__()
            self.in_layer = PatchEmbed()
            self.num_patches = self.in_layer.num_patches  # 获取num_patches的大小

            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # 将class token注册为参数，这个就是以后的结果
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))  # 将pos注册为参数，让模型自己学习特征之间的位置关系
            self.drop = nn.Dropout(config['dropout'])

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
            out = self.out_layer(cls_token_extract)  # [B, 102]，过全连接层，收束得到结果
            return out

    def lr_lambda(epoch, stable_epochs, decay_epochs):
        # 在前 stable_epochs 轮次，学习率保持不变
        if epoch < stable_epochs:
            return 1.0
        # 在接下来的 decay_epochs 轮次，逐步递减学习率
        elif epoch < stable_epochs + decay_epochs:
            return 1.0 - (epoch - stable_epochs) / decay_epochs
        else:
            return 0.0  # 学习率衰减到 0

    def get_acc(output, target):
        _, pred = torch.max(output, dim=1)
        correct = (pred == target).sum().item()
        accuracy = correct / target.size(0)
        return accuracy

    def mixup_data(x, y, alpha=1.0):
        '''返回混合后的图像和标签'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    transform_train, transform_val = get_transforms(config['img_size'], config['erase_ratio'])
    train_loader, val_loader = get_dataloader(root='./datasets', transform_train=transform_train,
                                              transform_val=transform_val, batch_size=config['batch_size'],
                                              num_workers=config['num_workers'])

    vit = ViT(embed_dim=config['embed_dim'], depth=config['depth']).to(device)
    optimizer = torch.optim.AdamW(vit.parameters(), lr=config['lr'], weight_decay=weight_decay)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, config['lr_stable_epochs'],
                                                                      config['lr_decay_epochs']))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device)
    scaler = GradScaler()

    # 简单训练几轮
    vit.train()
    total_train_loss = 0
    for epoch in range(config['n_epochs']):
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast():
                img, target_a, target_b, lam = mixup_data(img, target, alpha=config['mix_ratio'])
                output = vit(img)
                loss = lam * loss_fn(output, target_a) + (1 - lam) * loss_fn(output, target_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if epoch == config['n_epochs'] - 1:
                total_train_loss += loss.item()
        scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)

    # 评估在val集上的准确率
    vit.eval()
    total_val_acc = 0
    total_val_loss = 0
    with torch.no_grad(), autocast():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = vit(img)
            loss = loss_fn(output, target)
            total_val_loss += loss.item()

            val_acc = get_acc(output, target)
            total_val_acc += val_acc

    avg_val_acc = total_val_acc / len(val_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    overfit = avg_train_loss - avg_val_loss
    return avg_val_acc, overfit


if __name__ == '__main__':
    n_trials = 50  # 搜索的组数

    study = optuna.create_study(directions=["maximize", "minimize"])
    study.optimize(objective, n_trials=n_trials)  # 搜索n_trials组超参数

    print("Number of Pareto-optimal trials:", len(study.best_trials))
    for trial in study.best_trials:
        print(f"Trial#{trial.number}, val_acc={trial.values[0]:.4f}, overfit={trial.values[1]:.4f}")

    # 保存到JSON
    pareto_configs = []
    for trial in study.best_trials:
        pareto_configs.append({
            "trial_number": trial.number,
            "value": trial.values,
            "params": trial.params
        })

    with open('best_config.json', 'w') as f:
        json.dump(pareto_configs, f, indent=4)
