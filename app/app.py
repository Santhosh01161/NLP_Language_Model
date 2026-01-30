import torch
import torch.nn as nn
import torchtext
import math
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# --- 1. MODEL ARCHITECTURE (Must match your notebook) ---
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# --- 2. LOAD MODEL AND DATA ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Vocab and Tokenizer
# Adjust path if running from the 'app' directory: '../model/vocab'
vocab = torch.load('./model/vocab', map_location=device)
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Hyperparameters (must match your training notebook)
vocab_size = len(vocab)
emb_dim = 512
hid_dim = 512
num_layers = 3
dropout_rate = 0.65

model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
model.load_state_dict(torch.load('./model/best-val-lstm_lm.pt', map_location=device))
model.eval()

# --- 3. GENERATION LOGIC ---
def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device):
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break
            indices.append(prediction)

    itos = vocab.get_itos()
    return ' '.join([itos[i] for i in indices])

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt', '')
    # You can allow users to customize these or hardcode them
    temperature = float(data.get('temperature', 0.8)) 
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    generated_text = generate(prompt, 30, temperature, model, tokenizer, vocab, device)
    return jsonify({'result': generated_text})

if __name__ == '__main__':
    app.run(debug=True)