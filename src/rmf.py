import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time


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

# 可修改此变量，代表从每个txt读取的最大行数
N = 1000  # 例如读取前500行

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
def clean_text(text):
    text = normalize('NFD', text.lower())
    text = re.sub('[^A-Za-z ]+', '', text)
    return text
df = pd.read_csv('train.txt', header=None)
df.head()
df.shape
train_dataset = pd.read_csv('train.txt', sep=';', header=None, names=['Text', 'Sentiment'])

train_dataset.info()
test_dataset = pd.read_csv('test.txt', sep=';', header=None, names=['Text', 'Sentiment'])

test_dataset.info()
validation_dataset = pd.read_csv('val.txt', sep=';', header=None, names=['Text', 'Sentiment'])

validation_dataset.info()
datasets = ["train", "test", "val"]
data_frames = {}
for dataset in datasets:
    file_path = f"{dataset}.txt"
    data_frames[dataset] = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        names=["text", "Sentiment"]
    )

data_frames["val"]['text_length'] = data_frames["val"]['text'].apply(len)

max_length_index = data_frames["val"]['text_length'].idxmax()

max_length_emotion = data_frames["val"].iloc[max_length_index]['Sentiment']
max_length_text = data_frames["val"].iloc[max_length_index]['text']
import matplotlib.pyplot as plt
import seaborn as sns


def bar_chart(data_frames):
    datasets = ["train", "test", "val"]
    num_datasets = len(datasets)
    fig, ax = plt.subplots(1, num_datasets, figsize=(5 * num_datasets, 5), sharey=True)
    palette = sns.color_palette("viridis", 6)

    for i, dataset in enumerate(datasets):
        sns.countplot(data=data_frames[dataset], x="Sentiment", palette=palette, ax=ax[i])
        ax[i].set_title(f"Number of each Sentiment in {dataset}")

        for p in ax[i].patches:
            ax[i].annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 5),
                textcoords="offset points",
            )

    fig.suptitle("Number of each emotion in the data sets", fontsize=16)
    plt.tight_layout()
    plt.show()


# bar_chart(data_frames)
nltk.download('stopwords')
print(stopwords.words('english'))


class TextProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, lower=False, stem=False):
        self.lower = lower
        self.stem = stem

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        def text_processing(text):
            processed_text = re.sub('[^a-zA-Z]', ' ', text)
            if self.lower:
                processed_text = processed_text.lower()
            processed_text = processed_text.split()
            if self.stem:
                ps = PorterStemmer()
                processed_text = [ps.stem(word) for word in processed_text if
                                  word not in set(stopwords.words('english'))]
            processed_text = ' '.join(processed_text)
            return processed_text

        return [text_processing(text) for text in X]


logging.getLogger('joblib').setLevel(logging.ERROR)

text_processor = TextProcessor(lower=True, stem=False)
vectorizer = CountVectorizer(max_features=3000)
RF = RandomForestClassifier(
    n_estimators=50, random_state=42, n_jobs=-1, verbose=0,  # Set verbose to 0 to suppress messages
    max_depth=100, min_samples_split=100, min_samples_leaf=5, max_features='sqrt'
)

pipeline = Pipeline([
    ("text_processing", text_processor),
    ("vectorizer", vectorizer),
    ("classifier", RF)
])

pipeline.fit(train_dataset['Text'], train_dataset['Sentiment'])

train_accuracies = []
test_accuracies = []
val_accuracies = []
start_time = time.time()
for i in range(1, 51):
    RF.set_params(n_estimators=i)
    pipeline.fit(train_dataset['Text'], train_dataset['Sentiment'])

    train_pred = pipeline.predict(train_dataset['Text'])
    train_accuracy = accuracy_score(train_dataset['Sentiment'], train_pred)
    train_accuracies.append(train_accuracy)

    test_pred = pipeline.predict(test_dataset['Text'])
    test_accuracy = accuracy_score(test_dataset['Sentiment'], test_pred)
    test_accuracies.append(test_accuracy)

    val_pred = pipeline.predict(validation_dataset['Text'])
    val_accuracy = accuracy_score(validation_dataset['Sentiment'], val_pred)
    val_accuracies.append(val_accuracy)

# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 51), train_accuracies, label='Training Accuracy')
# plt.plot(range(1, 51), test_accuracies, label='Test Accuracy')
# plt.plot(range(1, 51), val_accuracies, label='Validation Accuracy')
# plt.xlabel('Number of Estimators')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs Number of Estimators for Random Forest')
# plt.legend()
# plt.grid(True)
# plt.show()
end_time = time.time()
with open("randomforest.txt", "a") as f:
    print(f"运行时间: {end_time - start_time:.2f}秒", file=f)

with open("randomforest.txt", "a") as f:
    print("Data Aug Size:", N, file=f)
    print("Train set accuracy:", train_accuracy, file=f)

    print("Test set accuracy:", test_accuracy, file=f)

    print("Validation set accuracy:", val_accuracy, file=f)

def plot_confusion_matrices_and_classification_report(train_true, train_pred, val_true, val_pred, test_true, test_pred, labels):
    train_conf_matrix = confusion_matrix(train_true, train_pred)
    val_conf_matrix = confusion_matrix(val_true, val_pred)
    test_conf_matrix = confusion_matrix(test_true, test_pred)
    with open("randomforest.txt", "a") as f:
        print("Train Classification Report:\n", classification_report(train_true, train_pred, target_names=labels),file=f)
        print("Validation Classification Report:\n", classification_report(val_true, val_pred, target_names=labels), file=f)
        print("Test Classification Report:\n", classification_report(test_true, test_pred, target_names=labels), file=f)

    fig, axes = plt.subplots(3, 1, figsize=(8, 18))

    sns.heatmap(train_conf_matrix, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Train Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(val_conf_matrix, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Validation Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    sns.heatmap(test_conf_matrix, annot=True, fmt='d', cmap='Purples',
                xticklabels=labels, yticklabels=labels, ax=axes[2])
    axes[2].set_title('Test Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')

    plt.tight_layout()
    plt.show()
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def compute_and_plot_auc(train_true, train_prob, val_true, val_prob, test_true, test_prob, labels):
    # Binarize the labels for multi-class AUC
    train_true_bin = label_binarize(train_true, classes=labels)
    val_true_bin = label_binarize(val_true, classes=labels)
    test_true_bin = label_binarize(test_true, classes=labels)

    # 计算macro/micro AUC
    train_auc_macro = roc_auc_score(train_true_bin, train_prob, average='macro')
    val_auc_macro = roc_auc_score(val_true_bin, val_prob, average='macro')
    test_auc_macro = roc_auc_score(test_true_bin, test_prob, average='macro')

    train_auc_micro = roc_auc_score(train_true_bin, train_prob, average='micro')
    val_auc_micro = roc_auc_score(val_true_bin, val_prob, average='micro')
    test_auc_micro = roc_auc_score(test_true_bin, test_prob, average='micro')

    with open("randomforest.txt", "a") as f:
        print("\n--- AUC Scores ---", file=f)
        print(f"Train Macro AUC: {train_auc_macro:.4f}", file=f)
        print(f"Validation Macro AUC: {val_auc_macro:.4f}", file=f)
        print(f"Test Macro AUC: {test_auc_macro:.4f}", file=f)

        print(f"Train Micro AUC: {train_auc_micro:.4f}", file=f)
        print(f"Validation Micro AUC: {val_auc_micro:.4f}", file=f)
        print(f"Test Micro AUC: {test_auc_micro:.4f}", file=f)


plot_confusion_matrices_and_classification_report(
    train_dataset['Sentiment'], train_pred,
    test_dataset['Sentiment'], test_pred,
    validation_dataset['Sentiment'], val_pred,
    pipeline.classes_
)
train_prob = pipeline.predict_proba(train_dataset['Text'])
val_prob = pipeline.predict_proba(validation_dataset['Text'])
test_prob = pipeline.predict_proba(test_dataset['Text'])

compute_and_plot_auc(
    train_dataset['Sentiment'], train_prob,
    validation_dataset['Sentiment'], val_prob,
    test_dataset['Sentiment'], test_prob,
    list(pipeline.classes_)
)

