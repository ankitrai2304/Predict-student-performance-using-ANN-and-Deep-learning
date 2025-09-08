# ----------------------------------------------------------------------------
# Step 0: Import Necessary Libraries
# ----------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# Step 1 & 2: Dataset Preparation and Preprocessing
# ----------------------------------------------------------------------------
print("--- Step 1 & 2: Loading and Preprocessing Data ---")

# Load the dataset, specifying the semicolon delimiter
try:
    df = pd.read_csv('/workspaces/Predict-student-performance-using-ANN-and-Deep-learning/student-mat.csv', sep=';')
    print("Dataset loaded successfully.")

    # Define the target variable 'passed'
    # A student "passes" if their final grade (G3) is 10 or higher.
    df['passed'] = (df['G3'] >= 10).astype(int)

    # Select features (X) and target (y)
    # We drop G1, G2, G3 to prevent the model from learning the direct answer.
    X = df.drop(columns=['G1', 'G2', 'G3', 'passed'])
    y = df['passed']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object']).columns
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns

    # Split data into training (70%), validation (15%), and test (15%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Apply the preprocessing
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create TensorDatasets and DataLoaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    print("Data preprocessing and DataLoader setup complete.")
    print(f"Input features shape: {X_train_tensor.shape[1]}")

    # ----------------------------------------------------------------------------
    # Step 3: Model Design (ANN)
    # ----------------------------------------------------------------------------
    print("\n--- Step 3: Designing the ANN Model ---")
    
    input_features = X_train_tensor.shape[1]

    class StudentClassifier(nn.Module):
        def __init__(self, input_size):
            super(StudentClassifier, self).__init__()
            self.layer1 = nn.Linear(input_size, 32) # Increased neuron count for complexity
            self.layer2 = nn.Linear(32, 16)
            self.output_layer = nn.Linear(16, 1)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(0.3) # Added dropout for regularization

        def forward(self, x):
            x = self.relu(self.layer1(x))
            x = self.dropout(x)
            x = self.relu(self.layer2(x))
            x = self.dropout(x)
            x = self.sigmoid(self.output_layer(x))
            return x

    model = StudentClassifier(input_features)
    print("Model architecture defined:")
    print(model)

    # ----------------------------------------------------------------------------
    # Step 4: Training the Model
    # ----------------------------------------------------------------------------
    print("\n--- Step 4: Training the Model ---")
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    epochs = 75 # Increased epochs for better convergence
    
    # Lists to store metrics for plotting
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        validation_loss = val_loss / len(val_loader)
        validation_acc = 100 * correct_val / total_val
        val_losses.append(validation_loss)
        val_accuracies.append(validation_acc)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {validation_loss:.4f}, Val Acc: {validation_acc:.2f}%")

    print("Training finished.")

    # ----------------------------------------------------------------------------
    # Step 5: Evaluation
    # ----------------------------------------------------------------------------
    print("\n--- Step 5: Evaluating the Model ---")
    
    # 1. Plot Loss and Accuracy Curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 2. Evaluate on the unseen test data
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n--- Test Set Performance ---")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # 3. Display Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fail (0)', 'Pass (1)'],
                yticklabels=['Fail (0)', 'Pass (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Test Set')
    plt.show()

except FileNotFoundError:
    print("Error: 'student-mat.csv' not found. Please ensure the file is uploaded correctly.")
except Exception as e:
    print(f"An error occurred: {e}")
