import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import utils
import pickle

# Гиперпараметры
EMBEDDING_DIM = 128
HIDDEN_DIM = 128 # Нейронов в скрытом слое
NUM_EPOCHS = 30 #Примеров/шагов в обучении
BATCH_SIZE = 32 # Примеров за 1 обучение
LEARNING_RATE = 0.001 # На сколько сильно изменяются параметры при обучении
MODEL_PATH = "sentiment_model.pth"


# Подготовка данных
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = [self.encode_text(text, vocab) for text in texts]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def encode_text(self, text, vocab):
        return torch.tensor([vocab.get(char, 0) for char in text], dtype=torch.long)  # Токенизация по символам

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Определение модели
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        output = self.fc(hidden[-1])
        return output


# Функция для обучения модели
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {total_loss / len(train_loader)}')

    # Сохранение модели
    torch.save(model.state_dict(), MODEL_PATH)
    print("Модель сохранена!")

texts, labels = utils.get_texts()
print(texts, labels)

# словарь для обучения
words = [word for text in texts for word in text.split()]
vocab = {word: i + 1 for i, (word, _) in enumerate(Counter(words).items())}

# что на обучение что для теста
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)


train_dataset = ReviewDataset(train_texts, train_labels, vocab)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# CPU vs GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda/cpu:", "cuda" if torch.cuda.is_available() else "cpu")


model = SentimentModel(len(vocab) + 1, EMBEDDING_DIM, HIDDEN_DIM, 2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
train_model(model, train_loader, criterion, optimizer)

model.load_state_dict(torch.load(MODEL_PATH))
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
print("Записал словарь!")
