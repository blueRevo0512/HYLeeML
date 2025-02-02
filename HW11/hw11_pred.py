import torchvision
import torch
import numpy as np
import pandas as pd
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

# 定义与训练代码一致的模型结构
class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, 3, 1, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        return self.conv(x).squeeze()

class LabelPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )

    def forward(self, h):
        return self.layer(h)

# 定义与训练一致的数据预处理
target_transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15, fill=(0,)),
    torchvision.transforms.ToTensor(),
])

def predict(model_path=None, predictor_path=None):
    # 自动寻找最佳模型
    if model_path is None or predictor_path is None:
        ckpt_files = [f for f in os.listdir('checkpoints') if 'best_acc' in f]
        if not ckpt_files:
            raise FileNotFoundError("未找到最佳模型检查点")
        # 按准确率排序选择最佳模型
        best_ckpt = sorted(ckpt_files, key=lambda x: float(x.split('acc')[-1].replace('.pth','')))[-1]
        ckpt_path = os.path.join('checkpoints', best_ckpt)
        checkpoint = torch.load(ckpt_path)
        
        feature_extractor = FeatureExtractor().cuda()
        label_predictor = LabelPredictor().cuda()
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        label_predictor.load_state_dict(checkpoint['label_predictor'])
    else:
        # 原有加载方式
        feature_extractor = FeatureExtractor().cuda()
        label_predictor = LabelPredictor().cuda()
        feature_extractor.load_state_dict(torch.load(model_path))
        label_predictor.load_state_dict(torch.load(predictor_path))
    
    # 设置为评估模式
    feature_extractor.eval()
    label_predictor.eval()

    # 加载测试数据
    test_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # 进行预测
    results = []
    with torch.no_grad():
        for test_data, _ in test_dataloader:
            test_data = test_data.cuda()
            features = feature_extractor(test_data)
            logits = label_predictor(features)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            results.append(preds)

    # 生成提交文件
    results = np.concatenate(results)
    df = pd.DataFrame({'id': np.arange(len(results)), 'label': results})
    df.to_csv('submission.csv', index=False)
    print("预测结果已保存到 submission.csv")

if __name__ == "__main__":
    predict() 