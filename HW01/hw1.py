import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import csv
train_data = pd.read_csv('./covid.train.csv').drop(columns=['id']).values
x_train = train_data[:, :-1]
y_train = train_data[:, -1]
class COVID19Dataset(Dataset):
	def __init__(self, x, y=None):
		if y is None:
			self.y = y
		else:
			self.y = torch.FloatTensor(y)
		self.x = torch.FloatTensor(x)
	def __getitem__(self, idx):
		if self.y is None:
			return self.x[idx]
		else:
			return self.x[idx], self.y[idx]
	def __len__(self):
		return len(self.x)
train_dataset = COVID19Dataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
class My_model(nn.Module):
	def __init__(self, input_dim):
		super(My_model, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, 32),
			nn.ReLU(),
			nn.Linear(32, 32),
			nn.ReLU(),
			nn.Linear(32, 32),
			nn.ReLU(),
			nn.Linear(32, 1)
		)
	def forward(self, x):
		x = self.layers(x)
		x = x.squeeze(1)
		return x
model = My_model(input_dim = x_train.shape[1]).to('cuda')
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
n_epochs = 3000
for epoch in range(n_epochs):
	model.train()
	for x,y in train_loader:
		x, y = x.to('cuda'), y.to('cuda')
		pred = model(x)
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		if epoch % 100 == 0:
			print(f'Epoch {epoch}, Loss: {loss.item()}')
# test_data = pd.read_csv('./covid.test.csv').drop(columns=['id']).values
# x_test = test_data[:, :-1]
# y_test = test_data[:, -1]
# test_dataset = COVID19Dataset(x_test, y_test)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
# model.eval()
# total_loss = 0
# for x, y in test_loader:
# 	x, y = x.to('cuda'), y.to('cuda')
# 	with torch.no_grad():
# 		pred = model(x)
# 		loss = criterion(pred, y)
# 	total_loss += loss.item() * len(x)
# 	avg_loss = total_loss / len(test_loader.dataset)
# 	print(f'Test Loss: {avg_loss}')
# print(f'Test Loss: {avg_loss}')
# torch.save(model.state_dict(), 'model.pth')
test_data = pd.read_csv('./covid.test.csv').drop(columns=['id']).values
test_dataset = COVID19Dataset(test_data)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)
model.eval()
preds = []
for x in test_loader:
	x = x.to('cuda')
	with torch.no_grad():
		pred = model(x)
		preds.append(pred.detach().cpu())
preds = torch.cat(preds, dim = 0).numpy()
def save_pred(preds, file):
	with open(file, 'w') as fp:
		writer = csv.writer(fp)
		writer.writerow(['id', 'tested_positive'])
		for i, p in enumerate(preds):
			writer.writerow([i, p])
save_pred(preds, 'pred.csv')
torch.save(model.state_dict(), 'model.pth')
