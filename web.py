from flask import Flask, render_template, request
import torch
import torch.nn as nn
import pickle

from torch.serialization import MAP_LOCATION

app = Flask(__name__)

EMBEDDING_DIM = 128
HIDDEN_DIM = 1024
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
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

model = SentimentModel(len(vocab) + 1, EMBEDDING_DIM, HIDDEN_DIM, 2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.eval()


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        result = predict(model, text, vocab, device)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
