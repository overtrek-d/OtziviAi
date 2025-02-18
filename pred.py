import torch
import torch.nn as nn
from collections import Counter
import pickle

EMBEDDING_DIM = 128 # Параметры как при обучении!!!!!!!!
HIDDEN_DIM = 128
MODEL_PATH = "sentiment_model.pth"


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


def predict(model, text, vocab, device):
    model.eval()
    encoded_text = torch.tensor([vocab.get(word, 0) for word in text.split()], dtype=torch.long).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(encoded_text)
        prediction = torch.argmax(output, dim=1).item()
    return "Позитивный" if prediction == 1 else "Негативный"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using (CPU/GPU):", "GPU" if torch.cuda.is_available() else "CPU")

with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = SentimentModel(len(vocab) + 1, EMBEDDING_DIM, HIDDEN_DIM, 2).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("Модель загружена!")


text = input("Отзыв: ")
print(f'Отзыв: "{text}" - {predict(model, text, vocab, device)}')
