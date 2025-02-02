import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
import os
from datetime import datetime

source_transform = transforms.Compose([
    # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
    transforms.Grayscale(),
    # cv2 do not support skimage.Image, so we transform it to np.array, 
    # and then adopt cv2.Canny algorithm.
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    # Transform np.array back to the skimage.Image.
    transforms.ToPILImage(),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])
target_transform = transforms.Compose([
    # Turn RGB to grayscale.
    transforms.Grayscale(),
    # Resize: size of source data is 32x32, thus we need to 
    #  enlarge the size of target data from 28x28 to 32x32。
    transforms.Resize((32, 32)),
    # 50% Horizontal Flip. (For Augmentation)
    transforms.RandomHorizontalFlip(),
    # Rotate +- 15 degrees. (For Augmentation), and filled with zero 
    # if there's empty pixel after rotation.
    transforms.RandomRotation(15, fill=(0,)),
    # Transform to tensor for model inputs.
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)

class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y
    
feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()
m=nn.Sigmoid()
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())
epochs=3000

# 创建模型保存目录
os.makedirs('checkpoints', exist_ok=True)

# 在对抗训练主循环前添加变量
best_acc = 0.0
timestamp = datetime.now().strftime("%m%d_%H%M")  # 月日_时分时间戳

def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: control the balance of domain adaptatoin and classification.
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num, dom_hit = 0.0, 0.0,0.0

    # 使用tqdm包装迭代器
    loop = tqdm(zip(source_dataloader, target_dataloader), 
                total=min(len(source_dataloader), len(target_dataloader)),
                desc=f'Epoch {epoch}', leave=False)
    
    for i, ((source_data, source_label), (target_data, _)) in enumerate(loop):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        # Mixed the source data and target data, or it'll mislead the running params
        #   of batch_norm. (runnning mean/var of soucre and target data are different.)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # set domain label of source data to be 1.
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : train domain classifier
        feature = feature_extractor(mixed_data)
        # We don't need to train feature extractor in step 1.
        # Thus we detach the feature neuron to avoid backpropgation.
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()
        dom_hit+=torch.sum(m(domain_logits).ge(0.5)==domain_label).item()

        # Step 2 : train feature extractor and label classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        
        # 更新进度条显示
        loop.set_postfix({
            'D_loss': running_D_loss/(i+1),
            'F_loss': running_F_loss/(i+1),
            'acc': total_hit/total_num,
            'dom_acc': dom_hit/(2*total_num)
        })
    
    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num, dom_hit/(2*total_num)

# train 200 epochs
for epoch in range(epochs):
    p=epoch/(epochs-1)
    lambda_p=2/(math.exp(-10*p)+1)-1
    # 使用tqdm包装epoch循环
    with tqdm(range(1), desc=f'Total Progress') as epoch_pbar:
        train_D_loss, train_F_loss, train_classacc, train_domacc= train_epoch(source_dataloader, target_dataloader, lambda_p)
        epoch_pbar.set_postfix({
            'D_loss': train_D_loss,
            'F_loss': train_F_loss,
            'acc': train_classacc,
            'dom_acc': train_domacc
        })
    
    # 使用tqdm.write替代print保持输出整洁
    tqdm.write(f'epoch {epoch:3d}: train D loss: {train_D_loss:.4f}, train F loss: {train_F_loss:.4f}, '
               f'classacc {train_classacc:.4f}, domacc {train_domacc:.4f}')

    # 动态保存检查点（每200个epoch）
    if epoch % 200 == 0 or epoch == epochs - 1:
        ckpt_name = f'checkpoints/{timestamp}_epoch{epoch}_acc{train_classacc:.4f}.pth'
        torch.save({
            'feature_extractor': feature_extractor.state_dict(),
            'label_predictor': label_predictor.state_dict(),
            'epoch': epoch,
            'acc': train_classacc
        }, ckpt_name)
        tqdm.write(f'Saved checkpoint: {ckpt_name}')

    # 保存最佳模型
    if train_classacc > best_acc:
        best_acc = train_classacc
        best_name = f'checkpoints/{timestamp}_best_acc{best_acc:.4f}.pth'
        torch.save({
            'feature_extractor': feature_extractor.state_dict(),
            'label_predictor': label_predictor.state_dict(),
            'epoch': epoch,
            'acc': best_acc
        }, best_name)
        tqdm.write(f'New best model: {best_name}')

torch.save(feature_extractor.state_dict(),'extractor_model.bin')
torch.save(label_predictor.state_dict(),'predictor_model.bin')

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
label_predictor.load_state_dict(torch.load('predictor_model.bin'))

class_criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(list(feature_extractor.parameters()) + list(label_predictor.parameters()), lr=1e-3)
t_feature_extractor = FeatureExtractor().cuda()
t_label_predictor = LabelPredictor().cuda()
t_feature_extractor.load_state_dict(torch.load('extractor_model.bin'))
t_label_predictor.load_state_dict(torch.load('predictor_model.bin'))
t_feature_extractor.eval()
t_label_predictor.eval()

def new_state(model1, model2, beta=0.9):
    sd1 = model1.state_dict()
    sd2 = model2.state_dict()
    for key in sd2:
        sd2[key] = sd1[key] * (1 - beta) + sd2[key]*beta
        
    model2.load_state_dict(sd2)
    model2.eval()
for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        mixed_data = torch.cat([source_data, target_data], dim=0)
        
        class_logits = label_predictor(feature_extractor(target_data))
        with torch.no_grad():
          t_class_logits = t_label_predictor(t_feature_extractor(target_data))
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss_s = class_criterion(class_logits[:source_data.shape[0]], source_label)
        logits,t_logits=class_logits,t_class_logits
        prob2, pseudo_label2 = logits.softmax(dim=1).max(dim=1)
        prob, pseudo_label = t_logits.softmax(dim=1).max(dim=1)
        print(prob2)
        print(prob)
        break
ce = nn.CrossEntropyLoss(reduction='none')
def c_loss(logits, t_logits):
    prob2, pseudo_label2 = logits.softmax(dim=1).max(dim=1)
    prob, pseudo_label = t_logits.softmax(dim=1).max(dim=1)
    flag = prob > 0.95
    return (flag * ce(logits, pseudo_label)).sum() / (flag.sum() + 1e-8), flag.sum(), torch.sum((pseudo_label==pseudo_label2) & flag).item()/(flag.sum().item() + 1e-8)

def train_epoch(source_dataloader, target_dataloader):
    running_loss = 0.0
    total_hit, total_num = 0.0, 0.0
    total_t_used, total_t = 0.0, 0.0
    pacc=0.0

    loop = tqdm(zip(source_dataloader, target_dataloader),
                total=min(len(source_dataloader), len(target_dataloader)),
                desc='Fine-tuning', leave=False)
    
    for i, ((source_data, source_label), (target_data, _)) in enumerate(loop):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        mixed_data = torch.cat([source_data, target_data], dim=0)
        
        class_logits = label_predictor(feature_extractor(mixed_data))
        with torch.no_grad():
            t_class_logits = t_label_predictor(t_feature_extractor(target_data))
        # loss = cross entropy of classification - lamb * domain binary cross entropy.
        #  The reason why using subtraction is similar to generator loss in disciminator of GAN
        loss_s = class_criterion(class_logits[:source_data.shape[0]], source_label)
        loss_t, num, pa= c_loss(class_logits[source_data.shape[0]:], t_class_logits)
        loss = loss_s + loss_t
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pacc+=pa
        total_t_used += num
        total_t += target_data.shape[0]
        total_hit += torch.sum(torch.argmax(class_logits[:source_data.shape[0]], dim=1) == source_label).item()
        total_num += source_data.shape[0]
        
        # 更新进度条
        loop.set_postfix({
            'loss': running_loss/(i+1),
            'acc': total_hit/total_num,
            'used': total_t_used/total_t,
            'p_acc': pacc/(i+1)
        })
    
    new_state(feature_extractor, t_feature_extractor)
    new_state(label_predictor, t_label_predictor)
        
    return running_loss / (i+1), total_hit / total_num, total_t_used/total_t, pacc/(i+1)

# 在自训练循环前添加变量
best_self_acc = 0.0
self_timestamp = datetime.now().strftime("%m%d_%H%M")

# 修改自训练的epoch循环
with tqdm(range(epochs), desc='Self Training') as main_pbar:
    for epoch in main_pbar:
        train_loss, train_acc, used_rate,pred_acc = train_epoch(source_dataloader, target_dataloader)
        if epoch % 100 == 0 or epoch == epochs - 1:
            ckpt_name = f'checkpoints/{self_timestamp}_self_epoch{epoch}_acc{train_acc:.4f}.pth'
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'label_predictor': label_predictor.state_dict(),
                'epoch': epoch,
                'acc': train_acc,
                'used_rate': used_rate
            }, ckpt_name)
            tqdm.write(f'Saved self-training checkpoint: {ckpt_name}')

        # 保存最佳自训练模型
        if train_acc > best_self_acc:
            best_self_acc = train_acc
            best_name = f'checkpoints/{self_timestamp}_self_best_acc{best_self_acc:.4f}.pth'
            torch.save({
                'feature_extractor': feature_extractor.state_dict(),
                'label_predictor': label_predictor.state_dict(),
                'epoch': epoch,
                'acc': best_self_acc
            }, best_name)
            tqdm.write(f'New self-training best model: {best_name}')

        # 更新主进度条信息
        main_pbar.set_postfix({
            'loss': train_loss,
            'acc': train_acc,
            'used': used_rate,
            'p_acc': pred_acc
        })
        tqdm.write(f'epoch {epoch:3d}: train loss: {train_loss:.4f}, acc: {train_acc:.4f}, '
                   f'used rate {used_rate:.4f},pred_acc {pred_acc:.4f}')

torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
torch.save(label_predictor.state_dict(), f'predictor_model.bin')