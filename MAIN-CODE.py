import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score

p = []
r = []
f = []

seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)

    class TimeSeriesClassifier(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(TimeSeriesClassifier, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            out = self.fc(lstm_out[:, -1, :])
            return torch.softmax(out, dim=1)

    def load_data(root_path, source_files, target_file):
        source_data = []
        source_labels = []

        for file in source_files:
            df = pd.read_csv(root_path + file, encoding='gbk')
            features = df.drop('标签', axis=1).values
            labels = df['标签'].values
            source_data.append(features)
            source_labels.append(labels)

        source_data = np.concatenate(source_data, axis=0)
        source_labels = np.concatenate(source_labels, axis=0)

        target_df = pd.read_csv(root_path + target_file, encoding='gbk')
        target_data = target_df.drop('标签', axis=1).values
        target_labels = target_df['标签'].values

        return source_data, source_labels, target_data, target_labels

    def preprocess_data(X_train, X_test):
        scaler = StandardScaler()
        n_samples, sequence_length, n_features = X_train.shape
        X_train_reshaped = X_train.reshape(-1, n_features)
        X_test_reshaped = X_test.reshape(-1, n_features)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(n_samples, sequence_length, n_features)
        X_test_scaled = X_test_scaled.reshape(X_test.shape[0], sequence_length, n_features)

        return X_train_scaled, X_test_scaled

    def mmd_loss(source_features, target_features, kernel="rbf", sigma=1.0):

        def rbf_kernel(X, Y, sigma):
            XX = torch.sum(X ** 2, dim=1, keepdim=True)
            YY = torch.sum(Y ** 2, dim=1, keepdim=True)
            XY = torch.mm(X, Y.t())
            return torch.exp(-(XX + YY - 2 * XY) / (2 * sigma ** 2))

        loss = 0.0
        if kernel == "rbf":
            XX = rbf_kernel(source_features, source_features, sigma)
            YY = rbf_kernel(target_features, target_features, sigma)
            XY = rbf_kernel(source_features, target_features, sigma)
            loss = torch.mean(XX + YY - 2 * XY)

        return loss

    def compute_class_weights(labels):
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        weights = [total_samples / (len(class_counts) * count) for count in class_counts]
        return weights

    def classification_loss(predictions, labels, weight):
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weight, dtype=torch.float32))
        labels = torch.from_numpy(labels).long()
        loss = criterion(predictions, labels)
        return loss

    def l1_loss(predictions, target):
        loss = torch.mean(torch.abs(predictions - target))
        return loss

    def train_model(model, source_train_dataset, target_train_dataset, source_labels, optimizer, iterations=100, batch_size=32):
        model.train()
        for iteration in range(iterations):
            running_loss = 0.0

            indices = torch.randperm(len(source_train_dataset))
            source_inputs = source_train_dataset[indices[:batch_size]]
            source_labels_batch = source_labels[indices[:batch_size]]
            indices_target = torch.randperm(len(target_train_dataset))
            target_inputs = target_train_dataset[indices_target[:batch_size]]

            optimizer.zero_grad()

            source_inputs = torch.tensor(source_inputs[0])
            target_inputs = torch.tensor(target_inputs[0])

            source_features = model(source_inputs)
            target_features = model(target_inputs)

            cls_loss = classification_loss(source_features, source_labels_batch, weight=[0.20, 0.1, 0.048])
            mmd_loss_value = mmd_loss(source_features, target_features)
            l1_loss_value = l1_loss(source_features, target_features)

            gamma = 2 / (1 + math.exp(-10 * (iteration) / iterations)) - 1
            gamma = gamma * 2
            l1_loss_value = 0
            loss = cls_loss + gamma * (mmd_loss_value + l1_loss_value)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            print(f"Iteration {iteration + 1}, Loss: {running_loss / len(source_train_dataset)}")
        return model

    root_path = "OFFICE31/"
    source_files = ["bd_zhmk_92.csv", "bd_dhys_92.csv", "tj_zgjy_92.csv"]
    target_file = "price_commitment_92.csv"

    X_source, source_labels, X_target, target_labels = load_data(root_path, source_files, target_file)
    print(f"X_source.shape: {X_source.shape}, X_target.shape: {X_target.shape}")

    source_labels = source_labels - 1

    X_source_train, X_source_test = X_source, X_source
    X_target_train, X_target_test = X_target, X_target

    X_source_train = torch.tensor(X_source_train, dtype=torch.float32).view(-1, 1, X_source_train.shape[1])
    X_source_test = torch.tensor(X_source_test, dtype=torch.float32).view(-1, 1, X_source_test.shape[1])
    X_target_train = torch.tensor(X_target_train, dtype=torch.float32).view(-1, 1, X_target_train.shape[1])
    X_target_test = torch.tensor(X_target_test, dtype=torch.float32).view(-1, 1, X_target_test.shape[1])

    X_source_train, X_source_test = preprocess_data(X_source_train, X_source_test)
    X_target_train, X_target_test = preprocess_data(X_target_train, X_target_test)

    X_source_train_tensor = torch.tensor(X_source_train, dtype=torch.float32)
    X_target_train_tensor = torch.tensor(X_target_train, dtype=torch.float32)

    source_train_dataset = TensorDataset(X_source_train_tensor)
    target_train_dataset = TensorDataset(X_target_train_tensor)

    input_size = X_source_train.shape[2]
    hidden_size = 16
    output_size = 3
    model = TimeSeriesClassifier(input_size, hidden_size, output_size)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print(f"Training with seed {seed}...")
    trained_model = train_model(model, source_train_dataset, target_train_dataset, source_labels, optimizer, iterations=100, batch_size=128)

    target_features = trained_model(torch.tensor(target_train_dataset[:][0]))
    print(f"Result shape: {target_features.shape}")

    outputs = target_features
    _, predicted = torch.max(outputs, 1)
    labels = target_labels

    predicted = predicted.flatten() + 1
    labels = labels.flatten()

    precision_macro = precision_score(labels, predicted, average='macro')
    recall_macro = recall_score(labels, predicted, average='macro')
    f1_macro = f1_score(labels, predicted, average='macro')

    precision_per_class = precision_score(labels, predicted, average=None)
    recall_per_class = recall_score(labels, predicted, average=None)
    f1_per_class = f1_score(labels, predicted, average=None)

    print(f"Seed {seed} - Macro Precision: {precision_macro:.4f}")
    print(f"Seed {seed} - Macro Recall: {recall_macro:.4f}")
    print(f"Seed {seed} - Macro F1 Score: {f1_macro:.4f}")

    print(f"Seed {seed} - Per-class Results:")
    for i in range(len(precision_per_class)):
        print(f"Class {i}: Precision - {precision_per_class[i]:.4f}, Recall - {recall_per_class[i]:.4f}, F1 Score - {f1_per_class[i]:.4f}")

    p.append(precision_macro)
    r.append(recall_macro)
    f.append(f1_macro)

    print(f"Completed seed {seed}\n")

df = pd.DataFrame({'p': p, 'r': r, 'f': f})
df.to_csv('metrics.csv', index=False)

print("All experiments completed. Results saved to metrics.csv")