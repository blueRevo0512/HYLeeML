# Import some useful packages for this homework
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset # "ConcatDataset" and "Subset" are possibly useful
from torchvision.datasets import DatasetFolder, VisionDataset
from torchsummary import summary
from tqdm.auto import tqdm
import random

# !nvidia-smi # list your current GPU

torch.cuda.set_device(0)  # 确保使用正确的GPU
torch.backends.cudnn.benchmark = False  # 禁用cudnn benchmark模式
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 启用同步CUDA

cfg = {
    'dataset_root': './food11-hw13',
    'save_dir': './outputs',
    'exp_name': "student_kd",  # 改名以区分实验
    'batch_size': 128,
    'lr': 5e-4,
    'seed': 20220013,
    'loss_fn_type': 'KD',  # 启用知识蒸馏
    'weight_decay': 1e-4,
    'grad_norm_max': 10,
    'n_epochs': 50,
    'patience': 20,  # 降低早停阈值
    'alpha': 0.3,        # 知识蒸馏权重
    'temperature': 5.0,  # 初始温度
    'temp_decay': 0.95,  # 温度衰减率
    'label_smoothing': 0.1,  # 添加标签平滑
}

myseed = cfg['seed']  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
random.seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

save_path = os.path.join(cfg['save_dir'], cfg['exp_name']) # create saving directory
os.makedirs(save_path, exist_ok=True)

# define simple logging functionality
log_fw = open(f"{save_path}/log.txt", 'w') # open log file to save log outputs
def log(text):     # define a logging function to trace the training process
    print(text)
    log_fw.write(str(text)+'\n')
    log_fw.flush()

log(cfg)  # log your configs to the log file

for dirname, _, filenames in os.walk('./food11-hw13'):
    if len(filenames) > 0:
        print(f"{dirname}: {len(filenames)} files.") # Show the file amounts in each split.

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# define training/testing transforms
test_tfm = transforms.Compose([
    # It is not encouraged to modify this part if you are using the provided teacher model. This transform is stardard and good enough for testing.
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    normalize,
])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super().__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
        # 判断是否为评估集
        self.is_eval = "evaluation" in path

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            filename = os.path.basename(fname)
            if self.is_eval:
                # 评估集：直接从文件名获取ID（去掉.jpg）
                label = -1  # 评估集没有标签
            else:
                # 训练集和验证集：从label_xxx.jpg格式解析标签
                label = int(filename.split("_")[0])
                if label < 0 or label >= 11:
                    print(f"Warning: Invalid label {label} in file {fname}")
                    label = 0
        except Exception as e:
            print(f"Error parsing label from {fname}: {e}")
            label = -1
        return im, label

# Form train/valid dataloaders
train_set = FoodDataset(os.path.join(cfg['dataset_root'],"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

valid_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

# Example implementation of Depthwise and Pointwise Convolution
def dwpw_conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels), #depthwise convolution
        nn.Conv2d(in_channels, out_channels, 1), # pointwise convolution
    )

# Define your student network here. You have to copy-paste this code block to HW13 GradeScope before deadline.
# We will use your student network definition to evaluate your results(including the total parameter amount).

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            # 初始卷积层
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # 第一个深度可分离卷积块
            dwpw_conv(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # 第二个深度可分离卷积块（带残差连接）
            nn.Sequential(
                dwpw_conv(32, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                dwpw_conv(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            ),
            nn.MaxPool2d(2, 2),

            # 最后的深度可分离卷积
            dwpw_conv(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加dropout防止过拟合
            nn.Linear(64, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def get_student_model(): # This function should have no arguments so that we can get your student network by directly calling it.
    # you can modify or do anything here, just remember to return an nn.Module as your student network.
    return StudentNet()

# End of definition of your student model and the get_student_model API
# Please copy-paste the whole code block, including the get_student_model function.

# DO NOT modify this block and please make sure that this block can run sucessfully.
student_model = get_student_model()
summary(student_model, (3, 224, 224), device='cpu')
# You have to copy&paste the results of this block to HW13 GradeScope.

# Load provided teacher model (model architecture: resnet18, num_classes=11, test-acc ~= 89.9%)
teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
# load state dict
teacher_ckpt_path = os.path.join(cfg['dataset_root'], "resnet18_teacher.ckpt")
teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))
# Now you already know the teacher model's architecture. You can take advantage of it if you want to pass the strong or boss baseline.
# Source code of resnet in pytorch: (https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)
# You can also see the summary of teacher model. There are 11,182,155 parameters totally in the teacher model
# summary(teacher_model, (3, 224, 224), device='cpu')

# Implement the loss function with KL divergence loss for knowledge distillation.
# You also have to copy-paste this whole block to HW13 GradeScope.
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.3, temperature=5.0, label_smoothing=0.1):
    # 确保标签在有效范围内
    labels = torch.clamp(labels, 0, 10)

    # 软目标KL散度损失
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    kd_loss = F.kl_div(log_soft_student, soft_teacher, reduction="batchmean") * (temperature**2)

    # 带标签平滑的交叉熵损失
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=label_smoothing)

    return alpha * ce_loss + (1 - alpha) * kd_loss

# choose the loss function by the config
if cfg['loss_fn_type'] == 'CE':
    # For the classification task, we use cross-entropy as the default loss function.
    loss_fn = nn.CrossEntropyLoss() # loss function for simple baseline.

if cfg['loss_fn_type'] == 'KD': # KD stands for knowledge distillation
    loss_fn = loss_fn_kd # implement loss_fn_kd for the report question and the medium baseline.

# You can also adopt other types of knowledge distillation techniques for strong and boss baseline, but use function name other than `loss_fn_kd`
# For example:
# def loss_fn_custom_kd():
#     pass
# if cfg['loss_fn_type'] == 'custom_kd':
#     loss_fn = loss_fn_custom_kd

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"device: {device}")

# The number of training epochs and patience.
n_epochs = cfg['n_epochs']
patience = cfg['patience'] # If no improvement in 'patience' epochs, early stop
# Initialize a model, and put it on the device specified.
student_model.to(device)
teacher_model.to(device)
teacher_model.eval()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(student_model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

# Initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=5,
    verbose=True
)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0.0

# 在训练循环开始前，加载已有的最佳模型（如果存在）
ckpt_path = f"{save_path}/student_best.ckpt"
if os.path.exists(ckpt_path):
    print(f"Loading model from {ckpt_path}")
    student_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    # 可以通过验证集评估一下加载的模型
    student_model.eval()
    with torch.no_grad():
        valid_loss = []
        valid_accs = []
        valid_lens = []
        for batch in valid_loader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = student_model(imgs)
            teacher_logits = teacher_model(imgs)
            loss = loss_fn(logits, labels, teacher_logits)
            acc = (logits.argmax(dim=-1) == labels).float().sum()
            valid_loss.append(loss.item() * len(imgs))
            valid_accs.append(acc)
            valid_lens.append(len(imgs))
        valid_loss = sum(valid_loss) / sum(valid_lens)
        valid_acc = sum(valid_accs) / sum(valid_lens)
        best_acc = valid_acc  # 更新best_acc为加载模型的准确率
        print(f"Loaded model validation accuracy: {valid_acc:.5f}")

# 添加混合精度训练
scaler = torch.cuda.amp.GradScaler()

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    student_model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []
    train_lens = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 只对训练和验证集的标签进行检查
        if torch.any((labels < 0) | (labels >= 11)):
            if torch.all(labels == -1):  # 如果全是-1，说明是测试集
                continue
            print(f"Warning: Invalid labels found: {labels}")
            labels = torch.clamp(labels, 0, 10)

        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                teacher_logits = teacher_model(imgs)
            logits = student_model(imgs)

            # 使用动态温度
            current_temp = cfg['temperature'] * (cfg['temp_decay'] ** epoch)
            loss = loss_fn_kd(logits, labels, teacher_logits,
                            temperature=current_temp,
                            label_smoothing=cfg['label_smoothing'])

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=cfg['grad_norm_max'])
        scaler.step(optimizer)
        scaler.update()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().sum()

        # Record the loss and accuracy.
        train_batch_len = len(imgs)
        train_loss.append(loss.item() * train_batch_len)
        train_accs.append(acc)
        train_lens.append(train_batch_len)

    train_loss = sum(train_loss) / sum(train_lens)
    train_acc = sum(train_accs) / sum(train_lens)

    # Print the information.
    log(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    student_model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []
    valid_lens = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = student_model(imgs)
            teacher_logits = teacher_model(imgs)

        # We can still compute the loss (but not the gradient).
        loss = loss_fn(logits, labels, teacher_logits)

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().sum()

        # Record the loss and accuracy.
        batch_len = len(imgs)
        valid_loss.append(loss.item() * batch_len)
        valid_accs.append(acc)
        valid_lens.append(batch_len)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / sum(valid_lens)
    valid_acc = sum(valid_accs) / sum(valid_lens)

    # update logs

    if valid_acc > best_acc:
        log(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        log(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        log(f"Best model found at epoch {epoch}, saving model")
        torch.save(student_model.state_dict(), f"{save_path}/student_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            log(f"No improvment {patience} consecutive epochs, early stopping")
            break

    # 在验证后更新学习率
    scheduler.step(valid_acc)

log("Finish training")
log_fw.close()
# create dataloader for evaluation
eval_set = FoodDataset(os.path.join(cfg['dataset_root'], "evaluation"), tfm=test_tfm)
eval_loader = DataLoader(eval_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
# Load model from {exp_name}/student_best.ckpt
student_model_best = get_student_model() # get a new student model to avoid reference before assignment.
ckpt_path = f"{save_path}/student_best.ckpt" # the ckpt path of the best student model.
student_model_best.load_state_dict(torch.load(ckpt_path, map_location='cpu')) # load the state dict and set it to the student model
student_model_best.to(device) # set the student model to device

# Start evaluate
student_model_best.eval()
eval_preds = [] # storing predictions of the evaluation dataset

# Iterate the validation set by batches.
for batch in tqdm(eval_loader):
    # A batch consists of image data and corresponding labels.
    imgs, _ = batch
    # We don't need gradient in evaluation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = student_model_best(imgs.to(device))
        preds = list(logits.argmax(dim=-1).squeeze().cpu().numpy())
    # loss and acc can not be calculated because we do not have the true labels of the evaluation set.
    eval_preds += preds

def pad4(i):
    return f"{i:04d}"  # 使用格式化字符串，自动补零到4位

# 生成与评估集文件名匹配的ID列表
ids = []
for fname in sorted(os.listdir(os.path.join(cfg['dataset_root'], "evaluation"))):
    if fname.endswith(".jpg"):
        # 直接使用文件名（去掉.jpg）作为ID
        ids.append(fname[:-4])
categories = eval_preds

df = pd.DataFrame()
df['Id'] = ids
df['Category'] = categories
df.to_csv(f"{save_path}/submission.csv", index=False)