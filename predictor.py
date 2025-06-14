import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Loading the dataset from ( https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset )
df = pd.read_csv("cardio_train.csv", sep=";")

# Preprocess data (turning data into dataframe)
df = df.drop(columns=["id"])

df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
df["age_years"] = df["age"] / 365  # Convert age from days to years
df["bp_diff"] = df["ap_hi"] - df["ap_lo"]  # Pulse pressure
df["map"] = (df["ap_hi"] + 2 * df["ap_lo"]) / 3  # Mean arterial pressure
df["weight_height_ratio"] = df["weight"] / df["height"]

# features selection from the csv columns
feature_names = ["age_years", "height", "weight", "ap_hi", "ap_lo", "cholesterol", 
                "gluc", "smoke", "alco", "active", "bmi", "bp_diff", "map", 
                "weight_height_ratio"]
X = df[feature_names]
y = df["cardio"]

# setup of scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# class declaration for dataset (OOP)
class BPDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Regulation
class BPClassifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.dropout4 = nn.Dropout(0.1)
        
        self.out = nn.Linear(32, 2)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        return self.out(x)

# Function to train the model
def train_model(model, train_loader, val_loader, epochs=100):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience = 15
    trigger_times = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # For resource optimization and prevention of overfitting, early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger_times = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered.")
                break
    
    return model

if __name__ == "__main__":
    # K-Fold Cross Validation
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
        print(f"\nFold {fold + 1}/{n_splits}")
        
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_ds = BPDataset(X_train, y_train)
        val_ds = BPDataset(X_val, y_val)
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64)
        
        model = BPClassifier(X.shape[1])
        model = train_model(model, train_loader, val_loader)
        
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(dim=1)
                all_preds.extend(preds.tolist())
                all_labels.extend(yb.tolist())
        
        fold_acc = accuracy_score(all_labels, all_preds)
        cv_scores.append(fold_acc)
        print(f"Fold {fold + 1} Accuracy: {fold_acc:.4f}")

    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # Final Training for the whole dataset
    print("\nTraining final model on full dataset...")
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)
    train_ds = BPDataset(X_train, y_train)
    val_ds = BPDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    final_model = BPClassifier(X.shape[1])
    final_model = train_model(final_model, train_loader, val_loader)

    final_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = final_model(xb).argmax(dim=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(yb.tolist())

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\nFinal Model Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    

    data["bmi"] = data["weight"] / ((data["height"] / 100) ** 2)
    data["bp_diff"] = data["ap_hi"] - data["ap_lo"]
    data["map"] = (data["ap_hi"] + 2 * data["ap_lo"]) / 3
    data["weight_height_ratio"] = data["weight"] / data["height"]
    
    input_vector = scaler.transform([[*data.values()]])
    input_tensor = torch.tensor(input_vector, dtype=torch.float32)
    output = final_model(input_tensor)
    pred = torch.argmax(output).item()
    prob = F.softmax(output, dim=1)[0][pred].item()
    print(f"\nPrediction: {'At Risk of Hypertension' if pred == 1 else 'Normal'}")
    print(f"Confidence: {prob:.2%}")

