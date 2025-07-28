# import json
import os
import re
import seaborn
import numpy as np
import torch
import torch.nn as nn
import torchvision
import timm
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from torch.cuda.amp import autocast, GradScaler
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
          'num_workers': 8,
          'lr': 3.2183e-04,
          'lr_stable_epochs': 10,  # normal:35 finetune:10
          'lr_decay_epochs': 15,  # normal:25 finetune:15
          'dropout': 0.15092,
          'patch_size': 16,
          'embed_dim': 768,
          'depth': 12,
          'n_heads': 12,
          'img_size': 224,
          'mlp_ratio': 4,
          'label_smoothing': 0.1,
          'mix_ratio': 0.5694,
          'erase_ratio': 0,
          'running_mod': 'eval',  # 'normal' , 'finetune' or 'eval'
          'plot_only': False,  # True: only plot confusion matrix
          'constant_save': True,
          'model_path': 'models/saved/vit_base_pretrain_baseline.pth'
          }
'''
'patch_size': 16,  # 采样大小
'embed_dim': 768,  # emded_dim必须能被n_heads整除
'depth': 5,  # 堆叠Encoder Block的数量
'n_heads': 6,  # 注意力头数
'img_size': 224,  # 默认正方形
'mlp_ratio': 4,  # MLP隐藏层的维度倍数，整数，默认4
'label_smoothing': 0.1,  # 默认0.1，应该不用改
'mix_ratio':0.30718,  # mixup混合率，0~1
'erase_ratio': 0  # 0.1 -> 10%
'''

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
with open('config_vit.json', 'w') as f:
    json.dump(config, f, indent=4)


def load_config(path='config_vit.json'):
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
    | Learning Rate     | {config['lr']}
    | Dropout           | {config['dropout']}
    | Patch_size        | {config['patch_size']}
    | Embed_dim         | {config['embed_dim']}
    | Depth             | {config['depth']}
    | n_heads           | {config['n_heads']}
    | Img_size          | {config['img_size']}
    | MLP_ratio         | {config['mlp_ratio']}
    | Label_smoothing   | {config['label_smoothing']}
    | Mix_ratio         | {config['mix_ratio']}
    | Erase_ratio       | {config['erase_ratio']}
    | Running_mod       | {config['running_mod']}
    +-------------------+-----------------------
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

'''
    def get_dataloader(root, transform_train, transform_val, batch_size):

    flowers102_train = torchvision.datasets.Flowers102(root=root, split='train', download=True, transform=transform_train)
    flowers102_val = torchvision.datasets.Flowers102(root=root, split='val', download=True, transform=transform_val)
    flowers102_test = torchvision.datasets.Flowers102(root=root, split='test', download=True, transform=transform_train)
    combined_train = torch.utils.data.ConcatDataset([flowers102_train, flowers102_test])

    train_loader = DataLoader(combined_train, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(flowers102_val, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    return train_loader, val_loader


    flowers102_train = torchvision.datasets.Flowers102(root=root, split='train', download=True, transform=transform_train)
    flowers102_val = torchvision.datasets.Flowers102(root=root, split='val', download=True, transform=transform_val)

    train_loader = DataLoader(flowers102_train, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader = DataLoader(flowers102_val, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=config['num_workers'], pin_memory=True)

    return train_loader, val_loader
    '''


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
        out = self.out_layer(cls_token_extract)  # [B, 101]，过全连接层，收束得到结果
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
                                            transform_train=transform_train,transform_val=transform_val, batch_size=config['batch_size'])
    print('loading data successfully!')

    # 初始化模型
    vit = ViT().to(device)

    # 根据运行模式决定是否加载预训练模型
    if config['running_mod'] in ['finetune', 'eval']:
        pretrained_path = config['model_path']  # 预训练模型路径
        print(f'loading model: {pretrained_path}')
        try:
            vit.load_state_dict(torch.load(pretrained_path))
            print('load successfully!')
        except Exception as e:  # 若加载失败，检查是否是预训练模型
            if config['running_mod'] == 'eval':
                if 'pretrain' in config['model_path']:
                    del vit
                    vit = timm.create_model(
                        'vit_base_patch16_224.augreg_in1k',
                        pretrained=False,  # 加载ImageNet-1k预训练权重
                        num_classes=101  # 替换分类头为Food-101的类别数
                    )
                    vit.load_state_dict(torch.load(config['model_path'], map_location='cpu'))
                    vit.eval().to(device)
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

        vit.eval()
        total_val_loss = 0
        total_top1_acc = 0
        total_top5_acc = 0

        all_preds = []
        all_targets = []


        with torch.no_grad(), autocast():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                output = vit(img)

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
                              save_path=f"Plots/confusion_matrix_food101_Vitbase_{config['erase_ratio'] * 100}.png",
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
                output = vit(img)
                total_train_acc += get_acc(output, target)
                train_samples += 1
                
        avg_train_acc = total_train_acc / len(val_loader)  # 使用相同的除数

        overfitting_ratio = (avg_train_acc - avg_top1_acc) / avg_train_acc
        print(f'Training accuracy on {len(val_loader)} batches: {avg_train_acc * 100:.2f}%')
        print(f'Overfitting ratio: {overfitting_ratio * 100:.2f}%')
        exit()

    # 训练模式（normal 或 finetune）
    optimizer = torch.optim.AdamW(vit.parameters(), lr=config['lr'], weight_decay=1e-3)
    scheduler = LambdaLR(optimizer,
                        lr_lambda=lambda epoch: lr_lambda(epoch, config['lr_stable_epochs'], config['lr_decay_epochs']))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']).to(device)
    scaler = GradScaler()

    # ----------------------------------------------------
    # 设置新 log 目录
    log_root = './Log/vit_log'

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
        # config = load_config('./config_vit.json')

        print(f'epoch{epoch} start, training model......')
        vit.train()
        for img, target in train_loader:
            img_bak = img
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            with autocast():
                if epoch % 5 == 0 or epoch == 1:
                    with torch.no_grad():
                        vit.eval()
                        output = vit(img).detach()
                        total_train_acc += get_acc(output, target)
                        #############
                        del output  #
                        #############
                    vit.train()

                img, target_a, target_b, lam = mixup_data(img, target, alpha=config['mix_ratio'])
                output = vit(img)
                # print(img.shape, target.shape)
                # print(output.shape)
                loss = lam * loss_fn(output, target_a) + (1 - lam) * loss_fn(output, target_b)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
        # ---图片数据记录---
        # writer.add_images('masked_imgs', img_bak, epoch)
        print(f'now epoch:{epoch} average train_loss:{total_train_loss / len(train_loader)},'
            f'now epoch:{epoch} average train_acc:{(total_train_acc / len(train_loader)) * 100:.2f}%')
        writer.add_scalar("Loss/train", (total_train_loss / len(train_loader)), epoch)
        if epoch % 5 ==0 or epoch == 1:
            writer.add_scalar("Acc/train", (total_train_acc / len(train_loader)) * 100, epoch)

        print('validating model......')
        vit.eval()
        with torch.no_grad(), autocast():
            for img, target in val_loader:
                img, target = img.to(device), target.to(device)
                output = vit(img)

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
                save_path = 'models/vitbase_finetuned.pth'
                print(f'Fine-tuned model saved, Val_acc={avg_acc * 100:.2f}%, current epoch: {epoch}')
            else:
                save_path = 'models/vitbase.pth'
                print(f'Model saved, Val_acc={avg_acc * 100:.2f}%, current epoch: {epoch}')
            torch.save(vit.state_dict(), save_path)
        elif epoch == config['n_epochs']:
            # 根据运行模式决定备份的模型名称
            if config['running_mod'] == 'finetune':
                backup_path = 'models/vitbase_finetuned_backup.pth'
            else:
                backup_path = 'models/vitbase_backup.pth'
            torch.save(vit.state_dict(), backup_path)
            print('Last training result has been backed up!')
        # ---记录lr---
        writer.add_scalar("lr", get_current_lr(optimizer), epoch)

        scheduler.step()

        if config['constant_save'] and epoch % 5 == 0:
            torch.save(vit.state_dict(), f'models/finetune/vitbase_trinity_{epoch}.pth')

        epoch += 1
    writer.close()
    print('training done!')
