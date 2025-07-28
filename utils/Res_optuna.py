import json
import numpy as np
import optuna
import torch
import torch.nn as nn
import torchvision
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from eraser import PatchwiseRandomErasing
import logging

# 配置optuna日志输出
optuna.logging.get_logger("optuna").addHandler(
    logging.FileHandler("Res_optuna_log.txt")
)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    mix_ratio = trial.suggest_float('mix_ratio', 0.2, 0.6)
    # erase_ratio = trial.suggest_float('erase_ratio', 0.1, 0.5)
    erase_ratio = 0

    config = {'n_epochs': 10,
              'batch_size': batch_size,
              'num_workers': 4,
              'lr': lr,
              'lr_stable_epochs': 7,
              'lr_decay_epochs': 3,
              'dropout': dropout,
              'img_size': 224,
              'label_smoothing': 0.1,  # 默认0.1，应该不用改
              'mix_ratio': mix_ratio,
              'erase_ratio': erase_ratio  # 0.1 -> 10%
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
        def __init__(self, dropout=config['dropout'], num_classes=101):
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

    res = ResNet18(dropout=config['dropout']).to(device)
    optimizer = torch.optim.AdamW(res.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: lr_lambda(epoch, config['lr_stable_epochs'],
                                                                      config['lr_decay_epochs']))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device)

    # 简单训练几轮
    res.train()
    for epoch in range(config['n_epochs']):
        for img, target in train_loader:
            img, target = img.to(device), target.to(device)

            optimizer.zero_grad()
            img, target_a, target_b, lam = mixup_data(img, target, alpha=config['mix_ratio'])
            output = res(img)
            loss = lam * loss_fn(output, target_a) + (1 - lam) * loss_fn(output, target_b)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # 评估在val集上的准确率
    res.eval()
    total_val_acc = 0
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = res(img)
            acc = get_acc(output, target)
            total_val_acc += acc

    avg_val_acc = total_val_acc / len(val_loader)
    return avg_val_acc



if __name__ == '__main__':
    n_trials = 50

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)  # 搜索n_trials组超参数

    print('Best hyperparameters:', study.best_params)

    # 保存到JSON
    with open('Res_best_config.json', 'w') as f:
        json.dump(study.best_params, f, indent=4)
