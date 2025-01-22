import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import torch.nn.functional as F
import numpy as np
import random
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import math
import os
import csv

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(87)

class myDataset(Dataset):
	def __init__(self, data_dir, segment_len=128):
		self.data_dir = data_dir
		self.segment_len = segment_len
	
		# Load the mapping from speaker neme to their corresponding id. 
		mapping_path = Path(data_dir) / "mapping.json"
		mapping = json.load(mapping_path.open())
		self.speaker2id = mapping["speaker2id"]
	
		# Load metadata of training data.
		metadata_path = Path(data_dir) / "metadata.json"
		metadata = json.load(open(metadata_path))["speakers"]
	
		# Get the total number of speaker.
		self.speaker_num = len(metadata.keys())
		self.data = []
		for speaker in metadata.keys():
			for utterances in metadata[speaker]:
				self.data.append([utterances["feature_path"], self.speaker2id[speaker]])
 
	def __len__(self):
			return len(self.data)
 
	def __getitem__(self, index):
		feat_path, speaker = self.data[index]
		# Load preprocessed mel-spectrogram.
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		# Segmemt mel-spectrogram into "segment_len" frames.
		if len(mel) > self.segment_len:
			# Randomly get the starting point of the segment.
			start = random.randint(0, len(mel) - self.segment_len)
			# Get a segment with "segment_len" frames.
			mel = torch.FloatTensor(mel[start:start+self.segment_len])
		else:
			mel = torch.FloatTensor(mel)
		# Turn the speaker id into long for computing loss later.
		speaker = torch.FloatTensor([speaker]).long()
		return mel, speaker
 
	def get_speaker_number(self):
		return self.speaker_num
	
def collate_batch(batch):
	# Process features within a batch.
	"""Collate a batch of data."""
	mel, speaker = zip(*batch)
	# Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
	mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
	# mel: (batch size, length, 40)
	return mel, torch.FloatTensor(speaker).long()


def get_dataloader(data_dir, batch_size, n_workers):
	"""Generate dataloader"""
	dataset = myDataset(data_dir)
	speaker_num = dataset.get_speaker_number()
	# Split dataset into training dataset and validation dataset
	trainlen = int(0.9 * len(dataset))
	lengths = [trainlen, len(dataset) - trainlen]
	trainset, validset = random_split(dataset, lengths)

	train_loader = DataLoader(
		trainset,
		batch_size=batch_size,
		shuffle=True,
		drop_last=True,
		num_workers=n_workers,
		pin_memory=True,
		collate_fn=collate_batch,
	)
	valid_loader = DataLoader(
		validset,
		batch_size=batch_size,
		num_workers=n_workers,
		drop_last=True,
		pin_memory=True,
		collate_fn=collate_batch,
	)

	return train_loader, valid_loader, speaker_num

class AttentivePooling(nn.Module):
	def __init__(self, dim):
		super().__init__()
		# 注意力权重计算层
		self.attention = nn.Sequential(
			nn.Linear(dim, dim),
			nn.Tanh(),
			nn.Linear(dim, 1)
		)
		
	def forward(self, x):
		# x: (batch_size, time_steps, dim)
		# 计算注意力权重
		attn_weights = self.attention(x)  # (batch_size, time_steps, 1)
		attn_weights = F.softmax(attn_weights, dim=1)  # 在时间维度上做softmax
		
		# 加权平均
		weighted_sum = torch.bmm(x.transpose(1, 2), attn_weights)  # (batch_size, dim, 1)
		weighted_sum = weighted_sum.squeeze(-1)  # (batch_size, dim)
		
		return weighted_sum

class AdditiveMarginSoftmax(nn.Module):
	def __init__(self, margin=0.35, scale=30):
		super().__init__()
		self.margin = margin
		self.scale = scale
		self.ce = nn.CrossEntropyLoss()
		
	def forward(self, logits, labels):
		# 添加数值稳定性
		eps = 1e-12
		# 特征归一化
		logits_norm = torch.norm(logits, p=2, dim=1, keepdim=True).clamp(min=eps)
		logits = logits / logits_norm
		
		# 获取对应真实标签的logits
		one_hot = torch.zeros_like(logits, device=logits.device)
		one_hot.scatter_(1, labels.view(-1, 1), 1.0)
		
		# 添加margin
		logits = logits - one_hot * self.margin
		
		# 缩放
		logits = logits * self.scale
		
		# 检查数值稳定性
		if torch.isnan(logits).any() or torch.isinf(logits).any():
			print("Warning: NaN or Inf in logits")
			
		return self.ce(logits, labels)

class Classifier(nn.Module):
	def __init__(self, d_model=256, n_spks=600, dropout=0.2):
		super().__init__()
		self.prenet = nn.Linear(40, d_model)
		
		# 增加Conformer层数和参数
		self.conformer_layers = nn.ModuleList([
			ConformerBlock(
				dim=d_model,
				dim_head=32,        # 减小每个头的维度以增加头数
				heads=16,           # 增加注意力头数
				ff_mult=4,
				conv_expansion_factor=2,
				conv_kernel_size=31,
				attn_dropout=dropout,
				ff_dropout=dropout,
				conv_dropout=dropout
			) for _ in range(4)     # 使用4层Conformer
		])
		
		self.attentive_pool = AttentivePooling(d_model)
		
		self.pred_layer = nn.Sequential(
			nn.Linear(d_model, d_model * 2),  # 增加中间层维度
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model * 2, d_model),  # 添加一层
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_model, n_spks),
			nn.BatchNorm1d(n_spks)
		)

	def forward(self, mels):
		out = self.prenet(mels)
		
		# 依次通过多层Conformer
		for layer in self.conformer_layers:
			out = layer(out)
		
		stats = self.attentive_pool(out)
		out = self.pred_layer(stats)
		return out

class ConformerBlock(nn.Module):
	def __init__(self, dim, dim_head, heads, ff_mult, conv_expansion_factor, 
				 conv_kernel_size, attn_dropout, ff_dropout, conv_dropout):
		super().__init__()
		
		# 添加每个子模块前的LayerNorm
		self.norm1 = nn.LayerNorm(dim)
		self.norm2 = nn.LayerNorm(dim)
		self.norm3 = nn.LayerNorm(dim)
		self.norm4 = nn.LayerNorm(dim)
		
		self.ff1 = FeedForward(dim, ff_mult, ff_dropout)
		self.attn = MultiHeadAttention(dim, heads, dim_head, attn_dropout)
		self.conv = ConformerConvModule(dim, conv_expansion_factor, conv_kernel_size, conv_dropout)
		self.ff2 = FeedForward(dim, ff_mult, ff_dropout)
		
	def forward(self, x):
		# 每个子模块前进行LayerNorm
		x = x + 0.5 * self.ff1(self.norm1(x))
		x = x + self.attn(self.norm2(x))
		x = x + self.conv(self.norm3(x))
		x = x + 0.5 * self.ff2(self.norm4(x))
		return x

class MultiHeadAttention(nn.Module):
	def __init__(self, dim, heads, dim_head, dropout):
		super().__init__()
		inner_dim = dim_head * heads
		self.heads = heads
		self.scale = dim_head ** -0.5
		
		self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
		self.to_out = nn.Linear(inner_dim, dim)
		
		self.dropout = nn.Dropout(dropout)
		
	def forward(self, x):
		b, n, d = x.shape
		qkv = self.to_qkv(x).chunk(3, dim=-1)
		q, k, v = map(lambda t: t.reshape(b, n, self.heads, -1).transpose(1, 2), qkv)
		
		dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
		attn = F.softmax(dots, dim=-1)
		attn = self.dropout(attn)
		
		out = torch.matmul(attn, v)
		out = out.transpose(1, 2).reshape(b, n, -1)
		return self.to_out(out)

class ConformerConvModule(nn.Module):
	def __init__(self, dim, expansion_factor, kernel_size, dropout):
		super().__init__()
		
		inner_dim = dim * expansion_factor
		padding = (kernel_size - 1) // 2
		
		self.net = nn.Sequential(
			nn.LayerNorm(dim),
			Transpose((1, 2)),
			nn.Conv1d(dim, inner_dim * 2, 1),
			nn.GLU(dim=1),
			nn.Conv1d(inner_dim, inner_dim, kernel_size, padding=padding, groups=inner_dim),
			nn.BatchNorm1d(inner_dim),
			nn.SiLU(),
			nn.Conv1d(inner_dim, dim, 1),
			Transpose((1, 2)),
			nn.Dropout(dropout)
		)
		
	def forward(self, x):
		return self.net(x)

class Transpose(nn.Module):
	def __init__(self, dims):
		super().__init__()
		self.dims = dims
		
	def forward(self, x):
		return x.transpose(*self.dims)

class FeedForward(nn.Module):
	def __init__(self, dim, mult, dropout):
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, dim * mult),
			nn.SiLU(),
			nn.Dropout(dropout),
			nn.Linear(dim * mult, dim),
			nn.Dropout(dropout)
		)
		
	def forward(self, x):
		return self.net(x)

def get_cosine_schedule_with_warmup(
	optimizer: Optimizer,
	num_warmup_steps: int,
	num_training_steps: int,
	num_cycles: float = 0.5,
	last_epoch: int = -1,
):
	"""
	Create a schedule with a learning rate that decreases following the values of the cosine function between the
	initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
	initial lr set in the optimizer.

	Args:
		optimizer (:class:`~torch.optim.Optimizer`):
		The optimizer for which to schedule the learning rate.
		num_warmup_steps (:obj:`int`):
		The number of steps for the warmup phase.
		num_training_steps (:obj:`int`):
		The total number of training steps.
		num_cycles (:obj:`float`, `optional`, defaults to 0.5):
		The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
		following a half-cosine).
		last_epoch (:obj:`int`, `optional`, defaults to -1):
		The index of the last epoch when resuming training.

	Return:
		:obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
	"""
	def lr_lambda(current_step):
		# Warmup
		if current_step < num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps))
		# decadence
		progress = float(current_step - num_warmup_steps) / float(
			max(1, num_training_steps - num_warmup_steps)
		)
		return max(
			0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
		)

	return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch, model, criterion, device):
	"""Forward a batch through the model."""

	mels, labels = batch
	mels = mels.to(device)
	labels = labels.to(device)

	outs = model(mels)

	# 使用AM-Softmax计算损失
	loss = criterion(outs, labels)

	# 对于预测，我们仍然使用普通的argmax
	preds = outs.argmax(1)
	# Compute accuracy.
	accuracy = torch.mean((preds == labels).float())

	return loss, accuracy

def valid(dataloader, model, criterion, device): 
	"""Validate on validation set."""
	model.eval()
	running_loss = 0.0
	running_accuracy = 0.0
	total_batches = 0
	
	pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

	for i, batch in enumerate(dataloader):
		with torch.no_grad():
			loss, accuracy = model_fn(batch, model, criterion, device)
			running_loss += loss.item()
			running_accuracy += accuracy.item()
			total_batches += 1

		pbar.update(len(batch[0]))  # 更新实际处理的样本数
		pbar.set_postfix(
			loss=f"{running_loss / (i+1):.4f}",
			accuracy=f"{running_accuracy / (i+1):.4f}",
		)

	pbar.close()
	model.train()

	avg_loss = running_loss / total_batches
	avg_accuracy = running_accuracy / total_batches
	
	return avg_accuracy, avg_loss

class InferenceDataset(Dataset):
	def __init__(self, data_dir):
		testdata_path = Path(data_dir) / "testdata.json"
		metadata = json.load(testdata_path.open())
		self.data_dir = data_dir
		self.data = metadata["utterances"]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		utterance = self.data[index]
		feat_path = utterance["feature_path"]
		mel = torch.load(os.path.join(self.data_dir, feat_path))

		return feat_path, mel


def inference_collate_batch(batch):
	"""Collate a batch of data."""
	feat_paths, mels = zip(*batch)

	return feat_paths, torch.stack(mels)

def parse_args():
	"""arguments"""
	config = {
		"data_dir": "./Dataset",
		"model_path": "./model.ckpt",
		"output_path": "./output.csv",
	}

	return config


def main(
	data_dir,
	model_path,
	output_path,
):
	"""Main function."""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"[Info]: Use {device} now!")

	mapping_path = Path(data_dir) / "mapping.json"
	mapping = json.load(mapping_path.open())

	dataset = InferenceDataset(data_dir)
	dataloader = DataLoader(
		dataset,
		batch_size=1,
		shuffle=False,
		drop_last=False,
		num_workers=8,
		collate_fn=inference_collate_batch,
	)
	print(f"[Info]: Finish loading data!",flush = True)

	speaker_num = len(mapping["id2speaker"])
	model = Classifier(n_spks=speaker_num).to(device)
	model.load_state_dict(torch.load(model_path))
	model.eval()
	print(f"[Info]: Finish creating model!",flush = True)

	results = [["Id", "Category"]]
	for feat_paths, mels in tqdm(dataloader):
		with torch.no_grad():
			mels = mels.to(device)
			outs = model(mels)
			preds = outs.argmax(1).cpu().numpy()
			for feat_path, pred in zip(feat_paths, preds):
				results.append([feat_path, mapping["id2speaker"][str(pred)]])

	with open(output_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerows(results)


if __name__ == "__main__":
	main(**parse_args())