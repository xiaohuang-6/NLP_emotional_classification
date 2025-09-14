import os
import random
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import make_pipeline, Pipeline
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 可修改此变量，代表从每个txt读取的最大行数
N = 1000  # 例如读取前500行
with open("bert.txt", "a") as f:
    print("--------------分割线---------------", file=f)
    print("Data Aug Size:", N, file=f)
# 要读取的情绪类别文件名
emotion_files = [
    "love_1000_clean.txt",
    "joy_1000_clean.txt",
    "sad_1000_clean.txt",
    "fear_1000_clean.txt",
    "surprise_1000_clean.txt",
    "anger_1000_clean.txt"
]

# 存储所有读取的句子
all_lines = []

# 逐个文件读取内容
for filename in emotion_files:
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:N]  # 读取前N行

            # all_lines.extend([line.strip() for line in lines if line.strip()])  # 去除空行和换行符
            # all_lines.extend([
            #     line.strip().replace(',', '').replace('\t', '')
            #     for line in lines if line.strip()
            # ])
            processed_lines = []
            for line in lines:
                if line.strip():  # 排除空行
                    cleaned = line.strip().replace(',', '').replace('\t', '')
                    if cleaned.count(';') >= 2:
                        # 分割成多段，保留最后一段前的内容和最后一段
                        parts = cleaned.split(';')
                        cleaned = ';'.join([''.join(parts[:-1]), parts[-1]])
                    processed_lines.append(cleaned)

            all_lines.extend(processed_lines)
    else:

        print(f"文件 {filename} 未找到，跳过。")

# 打乱顺序
random.shuffle(all_lines)

# 划分比例：2:1:1 => 50%, 25%, 25%
total = len(all_lines)
train_end = int(total * 2 / 3)
val_end = train_end + int(total / 6)

train_lines = all_lines[:train_end]
val_lines = all_lines[train_end:val_end]
test_lines = all_lines[val_end:]

# 写入三个文件
with open("train.txt", 'w', encoding='utf-8') as f:
    f.writelines(line + '\n' for line in train_lines)

with open("val.txt", 'w', encoding='utf-8') as f:
    f.writelines(line + '\n' for line in val_lines)

with open("test.txt", 'w', encoding='utf-8') as f:
    f.writelines(line + '\n' for line in test_lines)
# Read datasets
df_train = pd.read_csv('train.txt', names=['Text', 'Emotion'], sep=';')
df_val = pd.read_csv('val.txt', names=['Text', 'Emotion'], sep=';')
df_test = pd.read_csv('test.txt', names=['Text', 'Emotion'], sep=';')


# Create a mapping from emotion names to numerical indices
emotion_labels = df_train['Emotion'].unique()
label_mapping = {emotion: idx for idx, emotion in enumerate(emotion_labels)}

# Convert string labels to numerical indices
df_train['Emotion'] = df_train['Emotion'].map(label_mapping)
df_val['Emotion'] = df_val['Emotion'].map(label_mapping)
df_test['Emotion'] = df_test['Emotion'].map(label_mapping)


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Preprocess the text data
train_texts = df_train['Text'].tolist()
val_texts = df_val['Text'].tolist()
test_texts = df_test['Text'].tolist()

train_labels = df_train['Emotion'].tolist()
val_labels = df_val['Emotion'].tolist()
test_labels = df_test['Emotion'].tolist()

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)

class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(train_labels)))

device = torch.device('cpu')
model.to(device)

# Define the training loop
def train(model, device, loader, optimizer, epoch):
    model.train()
    total_loss = 0

    for batch in loader:

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    with open("bert.txt", "a") as f:
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(loader)}', file = f)
def evaluate(model, device, loader):
    model.eval()
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / len(loader.dataset)
    return accuracy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
start_time = time.time()
for epoch in range(3):
    train(model, device, train_loader, optimizer, epoch)
    val_acc = evaluate(model, device, val_loader)
    with open("bert.txt", "a") as f:
        print(f'Epoch {epoch+1}, Val Acc: {val_acc:.4f}', file=f)
end_time = time.time()
with open("bert.txt", "a") as f:
    print(f"运行时间: {end_time - start_time:.2f}秒", file=f)

# Evaluate the model on the test set
test_acc = evaluate(model, device, test_loader)
with open("bert.txt", "a") as f:
    print(f'Test Acc: {test_acc:.4f}', file=f)
torch.save(model.state_dict(), 'model.pth')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(set(train_labels)))

model.load_state_dict(torch.load('model.pth'))

def evaluate_cf_classification(model, device, loader, label_mapping):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():

        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())  # Collect true labels
            all_predictions.extend(predicted.cpu().numpy())  # Collect predictions

    # Generate confusion matrix

    cm = confusion_matrix(all_labels, all_predictions)

    # generate classification_report
    report = classification_report(all_labels, all_predictions, target_names=list(label_mapping.keys()))

    return cm,report
cm,report = evaluate_cf_classification(model, device, test_loader,label_mapping)
with open("bert.txt", "a") as f:
    print("Classification Report:", file=f)
    print(report,file=f)

# Plot confusion matrix with string labels
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#         xticklabels=list(label_mapping.keys()),
#         yticklabels=list(label_mapping.keys()))
#
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.title('Confusion Matrix')
# plt.show()
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
import torch.nn.functional as F

def evaluate_and_compute_auc(model, device, loader, label_mapping):
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)

    # Softmax to get probabilities
    all_probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()

    # Binarize true labels
    all_labels_bin = label_binarize(all_labels, classes=list(range(len(label_mapping))))

    # Calculate AUC
    macro_auc = roc_auc_score(all_labels_bin, all_probs, average='macro')
    micro_auc = roc_auc_score(all_labels_bin, all_probs, average='micro')

    return macro_auc, micro_auc
macro_auc, micro_auc = evaluate_and_compute_auc(model, device, test_loader, label_mapping)

with open("bert.txt", "a") as f:
    print("\n--- AUC Scores ---", file=f)
    print(f"Test Macro AUC: {macro_auc:.4f}", file=f)
    print(f"Test Micro AUC: {micro_auc:.4f}", file=f)
