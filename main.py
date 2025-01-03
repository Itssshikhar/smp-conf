# %% [markdown]
# ### Data Preparation

# %%
!pip install pandas
!pip install torch

# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

import matplotlib.pyplot as plt

# %%
pd.read_csv('extensive_dataset.csv')

# %%
data = data = pd.read_csv('extensive_dataset.csv')
data

# %%
cleaned_data = data.iloc[:, 7:14]
cleaned_data

# %%
cleaned_data.columns = ['Strain Dynamic (%)', 'Stress Dynamic (MPa)', 'Time (sec)', 'freq (Hz)', 'Temp (C)', "E' (MPa)", "E'' (MPa)"]

# %%
cleaned_data['Strain Dynamic (%)'] = pd.to_numeric(cleaned_data['Strain Dynamic (%)'], errors='coerce')
cleaned_data['Stress Dynamic (MPa)'] = pd.to_numeric(cleaned_data['Stress Dynamic (MPa)'], errors='coerce')
cleaned_data["Time (sec)"] = pd.to_numeric(cleaned_data["Time (sec)"], errors='coerce')
cleaned_data["freq (Hz)"] = pd.to_numeric(cleaned_data["freq (Hz)"], errors='coerce')
cleaned_data['Temp (C)'] = pd.to_numeric(cleaned_data['Temp (C)'], errors='coerce')
cleaned_data["E' (MPa)"] = pd.to_numeric(cleaned_data["E' (MPa)"], errors='coerce')
cleaned_data["E'' (MPa)"] = pd.to_numeric(cleaned_data["E'' (MPa)"], errors='coerce')

# %%
cleaned_data.dropna(how='all', inplace=True)
cleaned_data.ffill()

# %%
cleaned_data

# %%
X = cleaned_data[['Strain Dynamic (%)', 'Stress Dynamic (MPa)', 'Time (sec)', 'freq (Hz)', 'Temp (C)']].values
y = cleaned_data[["E' (MPa)"]].values

# %%
# Calculating the correlation of column A against all others
corr_matrix = cleaned_data.corr()["E' (MPa)"]
corr_matrix

# %%
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=69)

# %%
y_train

# %%
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(2) # unsqueeze for CNN, do not need for FFNN
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(2) # unsqueeze for CNN, do not need for FFNN
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# %% [markdown]
# ### Model Architecture

# %%
"""
# Normal CNN just working with some extra functions

# Define the deeper CNN model with BatchNorm, Dropout, and Xavier Initialization
class DeeperCNNWithBNDropout(nn.Module):
    def __init__(self):
        super(DeeperCNNWithBNDropout, self).__init__()

        # First block of convolutions
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=1)
        self.dropout1 = nn.Dropout(0.3)

        # Second block of convolutions
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(kernel_size=1)
        self.dropout2 = nn.Dropout(0.3)

        # Fully connected layers (fc1 input size will be set dynamically)
        self.fc1 = None  # Placeholder, will be set dynamically in forward
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        # Apply Xavier initialization
        self._initialize_weights()

    def forward(self, x):
        # First block
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Second block
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, features]

        # Dynamically define fc1 based on the flattened size of x
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    # Initialize the weights of the model using Xavier initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

# Instantiate the model
model = DeeperCNNWithBNDropout()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

"""

"""
# Normal Feed Forward Neural Network that works with SmoothL1Loss instead of MSE

class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
       # Input layer (taking 5 inputs: Stress, Strain, Freq, Time, Temp)
        self.fc1 = nn.Linear(5, 64)  # 5 inputs, 64 neurons in the first layer
        self.fc2 = nn.Linear(64, 128)  # 64 neurons in the second layer, 128 in the next
        self.fc3 = nn.Linear(128, 1)  # 128 neurons to 1 output (E')

        # Activation function
        self.relu = nn.ReLU()
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Ensure x is flattened to shape [batch_size, input_size]
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))  # Apply first layer and ReLU activation
        x = self.dropout(x)  # Dropout
        x = self.relu(self.fc2(x))  # Apply second layer and ReLU activation
        x = self.dropout(x)  # Dropout
        x = self.fc3(x)  # Output layer (no activation for regression)
        return x.view(-1, 1, 1)

input_size = X_train.shape[1]
print(f"This is the shape of Input size: {input_size}")
model = FeedForwardNN(input_size)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
"""

"""
# Define the CNN + LSTM Hybrid Model
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout_cnn = nn.Dropout(0.3)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # CNN layers
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.dropout_cnn(x)

        # Prepare the input for LSTM by permuting to [batch_size, sequence_length, features]
        x = x.permute(0, 2, 1)

        # LSTM layer
        x, _ = self.lstm(x)

        # Take the last output of the LSTM
        x = x[:, -1, :]

        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x.view(-1, 1)

# Instantiate the model
model = CNNLSTM()

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
"""

"""
# Defining the FNN(Feed Forward Neural Network) + Transformer
class FNNTransformerHybrid(nn.Module):
    def __init__(self, input_size, transformer_hidden_size, num_heads, transformer_layers):
        super(FNNTransformerHybrid, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First feedforward layer
        self.fc2 = nn.Linear(64, 128)  # Second feedforward layer

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=num_heads, dim_feedforward=transformer_hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Output layer
        self.fc3 = nn.Linear(128, 1)

        # Activation functions and dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x shape should be (batch_size, input_size)
        
        # Reshape inputs if necessary, to ensure compatibility
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Ensure it’s (batch_size, input_size)
        
        # Feed Forward NN layers
        x = self.relu(self.fc1(x))  # Apply first FC layer
        x = self.dropout(x)         # Apply dropout
        x = self.relu(self.fc2(x))  # Apply second FC layer

        # Add dimension for transformer (seq_len, batch_size, d_model)
        x = x.unsqueeze(0)  # Add a sequence length dimension (seq_len=1)

        # Transformer encoder expects shape (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)

        # Remove sequence length dimension
        x = x.squeeze(0)

        # Final output layer
        x = self.fc3(x)
        
        return x.view(-1, 1, 1)  # Adjust to expected output shape


model = FNNTransformerHybrid(input_size=5, transformer_hidden_size=256, num_heads=4, transformer_layers=2)
# Define the loss function and optimizer
criterion = nn.SmoothL1Loss()  # You can also use nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
"""

"""
# Defines the Long Short Term Memory class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)  # Output is 1 (for E')

    def forward(self, x):
        # x should have shape (batch_size, num_features)
        # Reshape x to (batch_size, seq_length, input_size)
        x = x.view(x.size(0), 1, -1)  # 1 is the seq_length, -1 infers the input_size
        h0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # Hidden state
        c0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)  # Cell state
        
        # Pass through LSTM layer
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_len, hidden_size)
        
        # Take the output from the last time step
        out = out[:, -1, :]  # out: (batch_size, hidden_size)
        
        # Pass through the fully connected layer
        out = self.fc(out)  # out: (batch_size, 1)
        
        return out.view(-1, 1, 1)  # Adjust output shape
    
# Hyperparameters
input_size = X_train.shape[1]  # Number of input features (Stress, Strain, Freq, Time, Temp)
hidden_size = 64  # Number of LSTM hidden units
num_layers = 2  # Number of LSTM layers
dropout_rate = 0.2  # Dropout rate

# Instantiate the model
model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout_rate=dropout_rate)

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()  # You can use MSELoss if you prefer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
"""

# Define the hybrid architecture
class CNN_FFN_Hybrid(nn.Module):
    def __init__(self, input_size, cnn_output_size, hidden_size, num_classes):
        super(CNN_FFN_Hybrid, self).__init__()
        
        # CNN Part
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers (Feed Forward Part)
        self.fc1 = nn.Linear(cnn_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # CNN Part
        x = x.unsqueeze(1)  # Adding channel dimension for 1D CNN
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output from CNN
        x = x.view(x.size(0), -1)
        
        # Feed Forward NN Part
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Hyperparameters and settings
input_size = 5  # Adjust based on your input data
cnn_output_size = 64 * (input_size // 4)  # Adjust based on CNN's output shape
hidden_size = 128
num_classes = 1  # Assuming regression task; for classification, adjust this

model = CNN_FFN_Hybrid(input_size, cnn_output_size, hidden_size, num_classes)

# Loss function and optimizer
criterion = nn.MSELoss()  # Assuming regression task
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% [markdown]
# ### Training Loop

# %%
# Training function
def train_model(model, train_loader, criterion, optimizer, epochs=1000):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients

            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# %% [markdown]
# ### Evaluation

# %%

# Evaluation function
def evaluate(model, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation
        predictions = []
        actuals = []
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())
    
    return total_loss / len(test_loader), np.vstack(predictions), np.vstack(actuals)

# Evaluate the model on the test set
test_loss, y_pred, y_true = evaluate(model, test_loader, criterion)
print(f'Test Loss (MSE): {test_loss:.4f}')


"""
# Use this for CNN only
# Function to evaluate the model on the test set
def evaluate_model(model, test_loader):
    model.eval()
    actuals = []
    predictions = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            actuals.append(targets)
            predictions.append(outputs)
    
    actuals = torch.cat(actuals).numpy().squeeze()  # Flatten to 1D
    predictions = torch.cat(predictions).numpy().squeeze()  # Flatten to 1D
    return actuals, predictions

# Get the actual and predicted values
actuals, predictions = evaluate_model(model, test_loader)


# Plot actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(actuals, predictions, color='blue', label='Predicted vs Actual')
plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], color='red', linestyle='--', label='Ideal Prediction')  # Line of perfect prediction
plt.xlabel("Actual E' (MPa)")
plt.ylabel("Predicted E' (MPa)")
plt.title("Predicted vs Actual E' Values for Convolutional Neural Network with LSTM (Hybrid)")
plt.legend()
plt.grid(True)
plt.show()
"""


# %%

# Plot Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.7)
plt.xlabel("Actual E' (MPa)")
plt.ylabel("Predicted E' (MPa)")
plt.title("Actual vs Predicted E' for Convolutional NN and Feed Forward NN Hybrid")
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line
plt.show()



# %%



