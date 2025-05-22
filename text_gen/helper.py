import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from nltk import word_tokenize, download # Ensure nltk is imported

# It's good practice to call nltk.download() at a more global scope or ensure it's run once.
# These lines are already in your file, ensure 'punkt' is available for word_tokenize.
download("punkt_tab") 
download("stopwords") # Not used in this specific change but kept from original

df = pd.read_csv('../stage_4_data/text_generation/data.csv') # I changed it to data.csv

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) # Get output from the last time step
        return out

# Define special tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>" # Token for unknown words

# Get jokes from DataFrame
raw_jokes = df['Joke'].astype(str).tolist()

# Create word vocabulary
all_words_for_vocab = []
for joke_str in raw_jokes:
    tokens = word_tokenize(joke_str.lower()) # Tokenize and convert to lowercase
    all_words_for_vocab.extend(tokens)

unique_words_from_data = sorted(list(set(all_words_for_vocab)))

# Add special tokens to the word list
words = [SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + unique_words_from_data # Changed from chars
word_to_int = {word: i for i, word in enumerate(words)} # Changed from char_to_int
int_to_word = {i: word for i, word in enumerate(words)} # Changed from int_to_char

vocab_size = len(words) # Changed from len(chars)
print(f"Word Vocabulary Size (including SOS/EOS/UNK): {vocab_size}")

embedding_dim = 64 # tunable
hidden_size = 128  # tunable
output_size = vocab_size # Output size must match new vocab_size
learning_rate = 0.001
sequence_length = 100 # Length of input sequences (now in words)
epochs = 10 # Number of epochs

# Prepare training data
dataX = []
dataY = []

for joke_str in raw_jokes:
    # Tokenize joke into words, add SOS/EOS, and convert to lowercase
    tokenized_joke_words = [SOS_TOKEN] + word_tokenize(joke_str.lower()) + [EOS_TOKEN]
    
    # Convert words to integers, using UNK_TOKEN for out-of-vocabulary words
    # (though all words from raw_jokes should be in vocab here as it's built from them)
    int_joke = [word_to_int.get(word, word_to_int[UNK_TOKEN]) for word in tokenized_joke_words]
    
    # Create sequences from this single joke
    # A sequence needs at least sequence_length + 1 tokens (words) to form one input-output pair
    if len(int_joke) > sequence_length:
        for i in range(0, len(int_joke) - sequence_length, 1):
            seq_in = int_joke[i : i + sequence_length]
            seq_out = int_joke[i + sequence_length] # The target word (or EOS_TOKEN)
            dataX.append(seq_in)
            dataY.append(seq_out)

n_patterns = len(dataX)
print("Total Patterns: ", n_patterns) # This will likely change

split_idx = int(n_patterns * 0.8)

train_dataX = dataX[:split_idx]
train_dataY = dataY[:split_idx]
test_dataX = dataX[split_idx:]
test_dataY = dataY[split_idx:]

X_train = torch.tensor(train_dataX, dtype=torch.long)
Y_train = torch.tensor(train_dataY, dtype=torch.long)
X_test = torch.tensor(test_dataX, dtype=torch.long)
Y_test = torch.tensor(test_dataY, dtype=torch.long)

# Instantiate the model - this will now use the updated vocab_size
model = RNN(vocab_size, embedding_dim, hidden_size, output_size)
train_data = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

val_data = torch.utils.data.TensorDataset(X_test, Y_test)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# The training loop would typically be in your main script (e.g., main.ipynb)
# or called from there if this helper.py is imported.
# For completeness, if you intend to run training directly from this script,
# you would add the training loop here:
#
# for epoch in range(epochs):
#     # Training phase
#     model.train()
#     train_loss_epoch = 0
#     for batch_X, batch_y in train_loader:
#         outputs = model(batch_X)
#         loss = criterion(outputs, batch_y)
# 
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss_epoch += loss.item()
#     avg_train_loss = train_loss_epoch / len(train_loader)
#     print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {avg_train_loss:.4f}')
# 
#     # Validation phase
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for batch_X_val, batch_y_val in val_loader:
#             outputs_val = model(batch_X_val)
#             loss_val = criterion(outputs_val, batch_y_val)
#             val_loss += loss_val.item()
#     
#     avg_val_loss = val_loss / len(val_loader)
#     print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}')
# 
# print("Training complete.")