# import json
import os
import re
import numpy as np
import seaborn
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from eraser import PatchwiseRandomErasing

if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(device)
else:
    device = torch.device('cpu')
    print(device)

config = {'n_epochs': -1,
          'class_names': None,
          'batch_size': 256,
          'patch_size': 16,
          'num_workers': 8,
          'lr': 3e-4,
          'lr_stable_epochs': 10,  # normal:35 finetune:10
          'lr_decay_epochs': 15,  # normal:25 finetune:15
          'dropout': 0.10092,
          'img_size': 224,
          'label_smoothing': 0.1,  # default 0.1
          'mix_ratio': 0.5964,
          'erase_ratio': 0.5,  # 0.1 -> 10%
          'running_mod': 'eval',  # 'normal' , 'finetune' or 'eval'
          'plot_only': True,  # True: only plot confusion matrix
          'constant_save': True,
          'model_path': 'models/saved/resnet_50_finetuned.pth',
          }
config['n_epochs'] = config['lr_stable_epochs'] + config['lr_decay_epochs']
transform_train = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                      transforms.RandomHorizontalFlip(),
                                      # torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                      transforms.RandomRotation(15),
                                      transforms.ToTensor(),
                                      PatchwiseRandomErasing(patch_size=config['patch_size'], erase_ratio=config['erase_ratio'], mode='random', random_ratio=True),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ])
if config['running_mod'] in ['finetune', 'normal']:
    transform_val = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                        # transforms.RandomHorizontalFlip(),
                                        # torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                        # transforms.RandomRotation(15),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
else:
    transform_val = transforms.Compose([transforms.Resize((config['img_size'], config['img_size'])),
                                        # transforms.RandomHorizontalFlip(),
                                        # torchvision.transforms.AutoAugment(policy=torchvision.transforms.AutoAugmentPolicy.IMAGENET),
                                        # transforms.RandomRotation(15),
                                        transforms.ToTensor(),
                                        PatchwiseRandomErasing(patch_size=config['patch_size'], erase_ratio=config['erase_ratio'], mode='random', random_ratio=False),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                        ])
'''
# 保存配置到 JSON 文件进行微操
with open('config_res.json', 'w') as f:
    json.dump(config, f, indent=4)


def load_config(path='config_res.json'):
    with open(path) as f:
        return json.load(f)
'''

def config_writer(config):
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
    | Mix_ratio         | {config['mix_ratio']}
    | Learning Rate     | {config['lr']}
    | Dropout           | {config['dropout']}
    | Img_size          | {config['img_size']}
    | Erase_ratio       | {config['erase_ratio']}
    | Running_mod       | {config['running_mod']}
    +-------------------+-----------------------
    model type:Training from scratch
    """
    return table_str


def get_dataloader(root, transform_train, transform_val, batch_size):
    food101_train = torchvision.datasets.Food101(root=root, split='train', download=False,
                                                 transform=transform_train)
    food101_val = torchvision.datasets.Food101(root=root, split='test', download=False,
                                               transform=transform_val)
    config['class_names'] = food101_val.classes
    train_loader = DataLoader(food101_train, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(food101_val, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    return train_loader, val_loader


class ResBlockType1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        output = self.block(x)
        return torch.relu(output + x)


class ResBlockType2(nn.Module):  # 做下采样
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        output = self.block(x)
        x = self.shortcut(x)
        return torch.relu(output + x)


class ResNet50(nn.Module):  # 构建一个类ResNet50架构的残差神经网络
    def __init__(self, dropout=config['dropout'], num_classes=101):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.res_layer1 = nn.Sequential(ResBlockType2(64, 256),
                                        ResBlockType1(256, 256),
                                        ResBlockType1(256, 256))
        self.res_layer2 = nn.Sequential(ResBlockType2(256, 512),
                                        ResBlockType1(512, 512),
                                        ResBlockType1(512, 512),
                                        ResBlockType1(512, 512))
        self.res_layer3 = nn.Sequential(ResBlockType2(512, 1024),
                                        ResBlockType1(1024, 1024),
                                        ResBlockType1(1024, 1024),
                                        ResBlockType1(1024, 1024),
                                        ResBlockType1(1024, 1024),
                                        ResBlockType1(1024, 1024))
        self.res_layer4 = nn.Sequential(ResBlockType2(1024, 2048),
                                        ResBlockType1(2048, 2048),
                                        ResBlockType1(2048, 2048))
        self.out_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
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


def get_current_lr(optimizer):
    # 假设只有一个参数组，获取第一个参数组的学习率
    return optimizer.param_groups[0]['lr']


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

def topk_acc(output, target):
    correct_top1 = 0
    correct_top5 = 0
    total = target.size(0)

    # Top-1 Accuracy
    _, predicted_top1 = output.topk(1, dim=1)
    correct_top1 += (predicted_top1.squeeze() == target).sum().item()

    # Top-5 Accuracy
    _, predicted_top5 = output.topk(5, dim=1)
    target_reshaped = target.view(-1, 1)
    correct_top5 += (predicted_top5 == target_reshaped).sum().item()

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total

    return top1_acc, top5_acc


def plot_confusion_matrix(cm, class_names=None, save_path="Plots/confusion_matrix.png", normalize=True):
    """
    绘制并保存混淆矩阵热力图
    :param cm: 混淆矩阵（二维 numpy array）
    :param class_names: 类别名称列表（如 ['pizza', 'burger', ...]），用于标签显示
    :param save_path: 图片保存路径
    :param normalize: 是否按行归一化（显示比例而不是绝对值）
    :param figsize: 图像大小（适配类别数量）
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 每行归一化，更清晰
        cm = np.nan_to_num(cm)  # 防止出现 NaN

    num_classes = cm.shape[0]

    # 动态设定 figsize，保持正方格子，同时控制最大图像尺寸
    square_size = 0.25  # 每个方格的尺寸（单位英寸）
    figsize = (num_classes * square_size, num_classes * square_size)
    figsize = (min(figsize[0], 25), min(figsize[1], 25))  # 防止图像过大

    plt.figure(figsize=figsize)
    seaborn.heatmap(cm,
                    annot=False,  # 若要显示每个格子的数值，可设为True
                    cmap='Blues',
                    xticklabels=class_names if class_names is not None else "auto",
                    yticklabels=class_names if class_names is not None else "auto",
                    square=True,
                    cbar=True)

    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)



if __name__ == '__main__':

    print('loading data...')
    train_loader, val_loader = get_dataloader(root='./datasets',
                                              transform_train=transform_train, transform_val=transform_val,
                                              batch_size=config['batch_size'])
    print('loading data successfully!')

    # 初始化模型
    resnet = ResNet50().to(device)

    # 根据运行模式决定是否加载预训练模型
    if config['running_mod'] in ['finetune', 'eval']:
        pretrained_path = config['model_path']  # 预训练模型路径
        print(f'loading model: {pretrained_path}')
        try:
            resnet.load_state_dict(torch.load(pretrained_path))
            print('load successfully!')
        except Exception as e:
            if config['running_mod'] == 'eval':
                if 'pretrain' in config['model_path']:
                    del resnet
                    resnet = torchvision.models.resnet50()
                    resnet.fc = nn.Linear(resnet.fc.in_features, 101)
                    resnet.load_state_dict(torch.load(config['model_path'], map_location='cpu'))
                    resnet.eval().to(device)
                    print('load pretrain model successfully!')
            else:
                print(f'load fail: {e}')
                print('--------------------------------------------------------------------')
                print(f'{config["running_mod"]} cannot proceed because model loading failed')
                exit()

    # 如果是评估模式，只进行验证
    if config['running_mod'] == 'eval':
        print('Evaluation mode only...')
        loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device)


        resnet.eval()
        total_val_loss = 0
        total_top1_acc = 0
        total_top5_acc = 0

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                output = resnet(img)
                # 计算loss
                loss = loss_fn(output, target)
                total_val_loss += loss.item()
                # 计算top1_acc, top5_acc
                local_top1_acc, local_top5_acc = topk_acc(output, target)
                total_top1_acc += local_top1_acc
                total_top5_acc += local_top5_acc
                # Collect predictions for confusion matrix
                _, predicted_top1 = output.topk(1, dim=1)
                all_preds.extend(predicted_top1.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # 生成Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds)
        plot_confusion_matrix(cm,
                              class_names=config['class_names'],
                              save_path=f"Plots/confusion_matrix_food101_ResNet50_{config['erase_ratio'] * 100}.png",
                              normalize=True)

        avg_top1_acc = total_top1_acc / len(val_loader)
        avg_top5_acc = total_top5_acc / len(val_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Evaluation results - Val_loss: {avg_val_loss}, '
              f'Top1_acc: {avg_top1_acc * 100:.2f}%, Top5_acc: {avg_top5_acc * 100:.2f}%')
        
        if config['plot_only']:
            print('Plotting completed...')
            exit()

        # 计算训练集准确率
        print('Calculating training accuracy...')
        total_train_acc = 0
        train_samples = 0
        with torch.no_grad():
            for img, target in train_loader:
                if train_samples >= len(val_loader):  # 当处理的batch数量达到验证集大小时停止
                    break
                img, target = img.to(device), target.to(device)
                output = resnet(img)
                total_train_acc += get_acc(output, target)
                train_samples += 1
                
        avg_train_acc = total_train_acc / len(val_loader)  # 使用相同的除数

        overfitting_ratio = (avg_train_acc - avg_top1_acc) / avg_train_acc
        print(f'Training accuracy on {len(val_loader)} batches: {avg_train_acc * 100:.2f}%')
        print(f'Overfitting ratio: {overfitting_ratio * 100:.2f}%')
        exit()


    # 训练模式（normal 或 finetune）
    optimizer = torch.optim.AdamW(resnet.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda epoch: lr_lambda(epoch, config['lr_stable_epochs'],
                                                           config['lr_decay_epochs']))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device)

    # ----------------------------------------------------
    # 设置新 log 目录
    log_root = './Log/res_log'

    # 获取已有 exp 子目录列表
    existing_logs = [d for d in os.listdir(log_root) if
                     os.path.isdir(os.path.join(log_root, d)) and re.fullmatch(r'exp\d+', d)]
    exp_nums = [int(re.findall(r'\d+', name)[0]) for name in existing_logs]
    next_exp_num = max(exp_nums, default=0) + 1
    new_log_dir = os.path.join(log_root, f'exp{next_exp_num}')
    print('current exp: ', new_log_dir)
    os.makedirs(new_log_dir, exist_ok=True)

    # 初始化 SummaryWriter
    writer = SummaryWriter(log_dir=new_log_dir)
    # ----------------------------------------------------

    writer.add_text('config', config_writer(config))

    # 根据运行模式显示不同的训练信息
    if config['running_mod'] == 'finetune':
        print('Starting model fine-tuning...')
    else:
        print('Starting model training from scratch...')

    epoch = 1
    max_acc = -1
    while epoch <= config['n_epochs']:
        total_train_loss = 0
        total_train_acc = 0
        total_val_loss = 0
        total_val_acc = 0
        # config = load_config('./config_res.json')

        print(f'epoch{epoch} start，training......')
        resnet.train()
        for img, target in train_loader:
            img_bak = img
            img, target = img.to(device), target.to(device)
            if epoch % 5 == 0 or epoch == 1:
                with torch.no_grad():
                    resnet.eval()
                    output = resnet(img).detach()
                    total_train_acc += get_acc(output, target)
                    #############
                    del output  #
                    #############
                resnet.train()

            img, target_a, target_b, lam = mixup_data(img, target, alpha=config['mix_ratio'])
            output = resnet(img)
            # print(img.shape, target.shape)
            # print(output.shape)
            loss = lam * loss_fn(output, target_a) + (1 - lam) * loss_fn(output, target_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        # ---图片数据记录---
        writer.add_images('masked_imgs', img_bak, epoch)
        print(f'now epoch:{epoch} average train_loss:{total_train_loss / len(train_loader)},'
              f'now epoch:{epoch} average train_acc:{(total_train_acc / len(train_loader)) * 100:.2f}%')
        writer.add_scalar("Loss/train", (total_train_loss / len(train_loader)), epoch)
        if epoch % 5 == 0 or epoch == 1:
            writer.add_scalar("Acc/train", (total_train_acc / len(train_loader)) * 100, epoch)

        print('validating model......')
        resnet.eval()
        with torch.no_grad():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                output = resnet(img)

                loss = loss_fn(output, target)

                total_val_acc += get_acc(output, target)
                total_val_loss += loss.item()
        # ---数据记录---
        print(f'now epoch:{epoch} average val_loss:{total_val_loss / len(val_loader)},'
              f'now epoch:{epoch} average val_acc:{(total_val_acc / len(val_loader)) * 100:.2f}%')
        writer.add_scalar("Loss/val", (total_val_loss / len(val_loader)), epoch)
        writer.add_scalar("Acc/val", (total_val_acc / len(val_loader)) * 100, epoch)
        # 判断是否保存模型
        avg_acc = total_val_acc / len(val_loader)
        if avg_acc > max_acc:
            max_acc = avg_acc
            # 根据运行模式决定保存的模型名称
            if config['running_mod'] == 'finetune':
                save_path = 'models/resnet50_finetuned.pth'
                print(f'Fine-tuned model saved, Val_acc={avg_acc * 100:.2f}%, current epoch: {epoch}')
            else:
                save_path = 'models/resnet50.pth'
                print(f'Model saved, Val_acc={avg_acc * 100:.2f}%, current epoch: {epoch}')
            torch.save(resnet.state_dict(), save_path)
        elif epoch == config['n_epochs']:
            # 根据运行模式决定备份的模型名称
            if config['running_mod'] == 'finetune':
                backup_path = 'models/resnet50_finetuned_backup.pth'
            else:
                backup_path = 'models/resnet50_backup.pth'
            torch.save(resnet.state_dict(), backup_path)
            print('Last training result has been backed up!')
        # ---记录lr---
        writer.add_scalar("lr", get_current_lr(optimizer), epoch)

        scheduler.step()

        if config['constant_save'] and epoch % 5 == 0:
            torch.save(resnet.state_dict(), f'models/finetune/resnet50_trinity_{epoch}.pth')

        epoch += 1
    writer.close()
    print('Training completed!')
