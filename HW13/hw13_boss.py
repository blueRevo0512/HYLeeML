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
import math

# !nvidia-smi # list your current GPU
cfg = {
    'dataset_root': './food11-hw13',
    'save_dir': './outputs',
    'exp_name': "strong_baseline",
    'batch_size': 64,
    'lr': 3e-4,
    'seed': 20220013,
    'loss_fn_type': 'KD', # simple baseline: CE, medium baseline: KD. See the Knowledge_Distillation part for more information.
    'weight_decay': 1e-5,
    'grad_norm_max': 10,
    'n_epochs': 300, # train more steps to pass the medium baseline.
    'patience': 60,
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
    # add some useful transform or augmentation here, according to your experience in HW3.
    transforms.Resize(256),  # You can change this
    transforms.RandomCrop(224), # You can change this, but be aware of that the given teacher model's input size is 224.
    # The training input size of the provided teacher model is (3, 224, 224).
    # Thus, Input size other then 224 might hurt the performance. please be careful.
    transforms.RandomHorizontalFlip(), # You can change this.
    transforms.ToTensor(),
    normalize,
])
path=os.path.join(cfg['dataset_root'],"training")
print(os.listdir(path))
l = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])

class FoodDataset(Dataset):
    def __init__(self, path, tfm=test_tfm, files = None):
        super().__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm

    def __len__(self):
        return len(self.files)

    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

# Form train/valid dataloaders
train_set = FoodDataset(os.path.join(cfg['dataset_root'],"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)

valid_set = FoodDataset(os.path.join(cfg['dataset_root'], "validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

def dwpw_conv(in_channels, out_channels, kernel_size, stride=1, padding=1,bias=False):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,bias=bias, groups=in_channels), #depthwise convolution
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1,  bias= bias,), # pointwise convolution
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class StudentNet(nn.Module):
    def __init__(self, inplanes = 64):
        super().__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = dwpw_conv(inplanes, inplanes, kernel_size=3)
        self.layer2 = dwpw_conv(inplanes, 128, kernel_size=3, stride=2)
        self.layer3 = dwpw_conv(128, 256, kernel_size=3, stride=2)
        self.layer4 = dwpw_conv(256, 141, kernel_size=3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(141, 11)

    def forward(self, x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x=self.fc(x)

        return x

# DO NOT modify this block and please make sure that this block can run sucessfully.
student_model = StudentNet()
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
summary(teacher_model, (3, 224, 224), device='cpu')

Slayer1out, Slayer2out, Slayer3out, Tlayer1out, Tlayer2out, Tlayer3out = [], [], [], [], [], []

def hookS1(module, input, output):
    Slayer1out.append(output)
    return None

def hookS2(module, input, output):
    Slayer2out.append(output)
    return None

def hookS3(module, input, output):
    Slayer3out.append(output)
    return None

def hookT1(module, input, output):
    Tlayer1out.append(output)
    return None

def hookT2(module, input, output):
    Tlayer2out.append(output)
    return None

def hookT3(module, input, output):
    Tlayer3out.append(output)
    return None

student_model = StudentNet()
student_model.layer1.register_forward_hook(hookS1)
student_model.layer2.register_forward_hook(hookS2)
student_model.layer3.register_forward_hook(hookS3)

teacher_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False, num_classes=11)
teacher_ckpt_path = os.path.join(cfg['dataset_root'], "resnet18_teacher.ckpt")
teacher_model.load_state_dict(torch.load(teacher_ckpt_path, map_location='cpu'))

teacher_model.layer1.register_forward_hook(hookT1)
teacher_model.layer2.register_forward_hook(hookT2)
teacher_model.layer3.register_forward_hook(hookT3)

# Implement the loss function with KL divergence loss for knowledge distillation.
# You also have to copy-paste this whole block to HW13 GradeScope.
def loss_fn_kd(student_logits, labels, teacher_logits, alpha=0.99, temperature=25.0):
    kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
    CE_loss = nn.CrossEntropyLoss()
    p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.log_softmax(teacher_logits / temperature, dim=1)
    loss=alpha * (temperature**2) * kl_loss(p, q) + (1-alpha) * CE_loss(student_logits,labels)

    return loss

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

def use_pretrain():
    student_model.conv1.weight = teacher_model.conv1.weight
    student_model.bn1.weight = teacher_model.bn1.weight
    student_model.bn1.bias = teacher_model.bn1.bias
    student_model.bn1.running_mean = teacher_model.bn1.running_mean
    student_model.bn1.running_var = teacher_model.bn1.running_var
    student_model.conv1.weight.requires_grad = False
    student_model.bn1.weight.requires_grad = False
    student_model.bn1.bias.requires_grad = False
use_pretrain()

# Initialize a model, and put it on the device specified.
student_model.to(device)
teacher_model.to(device) # MEDIUM BASELINE

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, student_model.parameters()), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=9, T_mult=2, eta_min=1e-5)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0.0

teacher_model.eval()  # MEDIUM BASELINE
for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    student_model.train()

    # These are used to record information in training.
    train_loss = []
    train_loss_hidden = []
    train_accs = []
    train_lens = []
    p=epoch/(n_epochs-1)
    lamb= p * p # 0-1
    for batch in tqdm(train_loader):
        Slayer1out, Slayer2out, Slayer3out, Tlayer1out, Tlayer2out, Tlayer3out = [], [], [], [], [], []
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        with torch.no_grad():  # MEDIUM BASELINE
            teacher_logits = teacher_model(imgs)  # MEDIUM BASELINE


        logits = student_model(imgs)
        slayer1out, slayer2out, slayer3out, tlayer1out, tlayer2out, tlayer3out = \
          Slayer1out[0], Slayer2out[0], Slayer3out[0], Tlayer1out[0], Tlayer2out[0], Tlayer3out[0]
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss_output = loss_fn(logits, labels, teacher_logits) # MEDIUM BASELINE
        loss_hidden = F.smooth_l1_loss(slayer1out, tlayer1out) + F.smooth_l1_loss(slayer2out, tlayer2out) + F.smooth_l1_loss(slayer3out, tlayer3out)

        loss =  loss_hidden + 10 * lamb * loss_output
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=cfg['grad_norm_max'])

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels).float().sum()

        # Record the loss and accuracy.
        train_batch_len = len(imgs)
        train_loss.append(loss.item() * train_batch_len)
        train_loss_hidden.append(loss_hidden.item() * train_batch_len)
        train_accs.append(acc)
        train_lens.append(train_batch_len)

    train_loss = sum(train_loss) / sum(train_lens)
    train_acc = sum(train_accs) / sum(train_lens)
    train_hidden_loss = sum(train_loss_hidden) / sum(train_lens)

    # Print the information.
    log(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, hidden_loss = {train_hidden_loss:.5f}, acc = {train_acc:.5f}")


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
            teacher_logits = teacher_model(imgs) # MEDIUM BASELINE

        # We can still compute the loss (but not the gradient).
        loss = loss_fn(logits, labels, teacher_logits) # MEDIUM BASELINE
        # loss = loss_fn(logits, labels) # SIMPLE BASELINE

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
log("Finish training")
log_fw.close()

# create dataloader for evaluation
eval_set = FoodDataset(os.path.join(cfg['dataset_root'], "evaluation"), tfm=test_tfm)
eval_loader = DataLoader(eval_set, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

student_model.eval()
eval_preds = [] # storing predictions of the evaluation dataset

# Iterate the validation set by batches.
for batch in tqdm(eval_loader):
    # A batch consists of image data and corresponding labels.
    imgs, _ = batch
    # We don't need gradient in evaluation.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = student_model(imgs.to(device))
        preds = list(logits.argmax(dim=-1).squeeze().cpu().numpy())
    # loss and acc can not be calculated because we do not have the true labels of the evaluation set.
    eval_preds += preds

def pad4(i):
    return "0"*(4-len(str(i))) + str(i)

# Save prediction results
ids = [pad4(i) for i in range(0,len(eval_set))]
categories = eval_preds

df = pd.DataFrame()
df['Id'] = ids
df['Category'] = categories
df.to_csv(f"{save_path}/submission_final.csv", index=False) # now you can download the submission.csv and upload it to the kaggle competition.