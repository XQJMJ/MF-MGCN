import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
import random
import torch.nn.init as init
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import KFold
import torch.optim as optim



class EdgeGCN(nn.Module):
    def __init__(self):
        super(EdgeGCN, self).__init__()
        self.conv1 = GCNConv(32, 16)
        self.conv2 = GCNConv(16, 2)


        self.lin = nn.Linear(19 * 2 * 5, 128)
        self.lin1 = nn.Linear(128, 32)
        self.lin2 = nn.Linear(32, 2)

        nn.init.kaiming_normal_(self.lin.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin.bias, 0)
        nn.init.kaiming_normal_(self.lin1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin1.bias, 0)
        nn.init.kaiming_normal_(self.lin2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.lin2.bias, 0)


    def forward(self, data_list, use_dropout=True):
        x_combined = []
        for data in data_list:
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)

            x = self.conv2(x, edge_index, (edge_weight > 0.5).float())
            x = F.relu(x)

            x_combined.append(x.view(-1, 19 * 2))


        x = torch.cat(x_combined, dim=1)
        x = x.to(torch.float32)
        if use_dropout:
            x = F.dropout(x, p=0.2, training=self.training)

        x = self.lin(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return x

def save_initial_weights(model, path='model_initial_weights.pth'):
    torch.save(model.state_dict(), path)
    print(f"Initial model weights saved to {path}")

def load_initial_weights(model, path='model_initial_weights.pth'):
    model.load_state_dict(torch.load(path))
    print(f"Initial weights loaded from {path}")

class MyCustomDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]




def load_and_group_data():
    groups = {'AD': [], 'NC': []}
    freq_bands = ['Alpha', 'Beta', 'Delta', 'Gamma', 'Theta']

    for group_idx in range(1, 881, 10):
        group = []
        for idx in range(group_idx, group_idx + 10):
            if group_idx <= 360:
                category = 'AD'
            elif 361 <= group_idx <= 650:
                category = 'NC'
            else:
                continue

            freq_data_list = []
            for band in freq_bands:
                features_path = f'E:/datasets/xinDE{band}10s309szhenzheng0.9huachuang880/sub-{idx}.csv'
                adjacency_path = f'E:/datasets/xinDE{band}10s309szhenzheng0.9huachuang880CorrelationResult/sub-{idx}.csv'
                features = pd.read_csv(features_path, header=None).values
                adjacency = pd.read_csv(adjacency_path, header=None).values

                features_tensor = torch.tensor(features, dtype=torch.float32)
                adjacency_tensor = torch.tensor(adjacency, dtype=torch.long)
                edge_index = torch.tensor(np.transpose(np.nonzero(adjacency_tensor)), dtype=torch.long)
                edge_weight = torch.tensor(adjacency_tensor[edge_index[0], edge_index[1]], dtype=torch.float32).unsqueeze(-1)

                y = torch.tensor([0 if category == 'AD' else 1])
                freq_data = Data(x=features_tensor, edge_index=edge_index, edge_attr=edge_weight, y=y)
                freq_data_list.append(freq_data)

            group.append(freq_data_list)

        if group:
            groups[category].append(group)

    return groups


def train(model, train_loader, criterion, optimizer, use_dropout=True):
    model.train()
    total_loss, total_correct = 0, 0
    for data_list in train_loader:
        optimizer.zero_grad()
        out = model(data_list, use_dropout=use_dropout)

        loss = criterion(out, data_list[0].y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = out.argmax(dim=1)
        total_correct += (pred == data_list[0].y).sum().item()
    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / len(train_loader.dataset)
    print(f'Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
    return avg_loss, accuracy


def test(model, test_loader, criterion):
    model.eval()

    total_loss, total_correct = 0, 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for data_list in test_loader:
            out = model(data_list)
            loss = criterion(out, data_list[0].y)
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += (pred == data_list[0].y).sum().item()
            all_outputs.append(out)
            all_targets.extend(data_list[0].y.tolist())


        all_outputs = torch.cat(all_outputs, dim=0)
        all_probs = F.softmax(all_outputs, dim=1)[:, 1].cpu().numpy()
        all_preds = all_outputs.argmax(dim=1).cpu().numpy()

        avg_loss = total_loss / len(test_loader.dataset)
        accuracy = total_correct / len(test_loader.dataset)
        precision = precision_score(all_targets, all_preds)
        recall = recall_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        auc = roc_auc_score(all_targets, all_probs)

        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, AUC: {auc:.4f}')
        return avg_loss, accuracy, precision, recall, f1, auc



def main():
    groups = load_and_group_data()
    ad_groups = groups['AD']
    nc_groups = groups['NC']

    fixed_indices = [
        # Fold 1
        ([21, 2, 24, 11, 19, 30, 14, 35, 27, 0, 3, 10, 6, 12, 31, 4, 32, 25, 33, 13, 7, 28, 26, 9, 18, 8, 23, 1, 5], [15, 22, 17, 20, 34, 16, 29], [17, 25, 19, 0, 3, 16, 10, 6, 12, 2, 4, 22, 13, 7, 9, 18, 28, 8, 1, 5, 23, 14, 26], [11, 21, 27, 15, 20, 24]),
        # Fold 2
        ([18, 35, 32, 11, 2, 7, 5, 4, 22, 25, 13, 14, 6, 21, 26, 29, 9, 19, 1, 23, 12, 17, 20, 24, 15, 10, 31, 30, 0], [34, 16, 3, 28, 27, 33, 8], [8, 13, 5, 4, 21, 23, 16, 28, 14, 6, 22, 18, 9, 19, 1, 12, 17, 20, 24, 15, 10, 0, 7], [11, 3, 27, 26, 25, 2]),
        # Fold 3
        ([0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35], [2, 6, 10, 15, 21, 32, 34], [0, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28], [1, 10, 15, 16, 20, 27]),
        # Fold 4
        ([1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35], [0, 7, 13, 16, 17, 20, 33], [0, 1, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28], [2, 9, 11, 24, 25]),
        # Fold 5
        ([1, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35], [0, 3, 4, 6, 16, 22, 33], [0, 1, 3, 4, 5, 6, 8, 10, 11, 12, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 28], [2, 7, 9, 13, 20, 27])
    ]

    fold_results = []

    for fold, (train_ad_indices, val_ad_indices, train_nc_indices, val_nc_indices) in enumerate(fixed_indices):
        print(f'  FOLD {fold + 1}')
        print('  --------------------------------')

        train_data = []
        val_data = []


        for idx in train_ad_indices:
            train_data.extend(ad_groups[idx])
        for idx in val_ad_indices:
            val_data.extend(ad_groups[idx])
        for idx in train_nc_indices:
            train_data.extend(nc_groups[idx])
        for idx in val_nc_indices:
            val_data.extend(nc_groups[idx])

        train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=10, shuffle=False)

        model = EdgeGCN()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        best_precision = 0.0
        best_recall = 0.0
        best_f1 = 0.0
        best_accuracy = 0.0
        best_auc = 0.0

        for epoch in range(1, 201):
            use_dropout = epoch > 5
            train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, use_dropout=use_dropout)
            val_loss, val_accuracy, precision, recall, f1, auc = test(model, val_loader, criterion)

            if precision > best_precision:
                best_precision = precision
            if recall > best_recall:
                best_recall = recall
            if f1 > best_f1:
                best_f1 = f1
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
            if auc > best_auc:
                best_auc = auc

            if epoch > 190 and epoch % 10 == 0:
                scheduler.step()

        fold_results.append((best_precision, best_recall, best_f1, best_accuracy, best_auc))
        print(f'Best Precision: {best_precision:.4f}, Best Recall: {best_recall:.4f}, Best F1 Score: {best_f1:.4f}, Best Accuracy: {best_accuracy:.4f}, Best AUC: {best_auc:.4f}')

    print('Top 5 Folds Results:')
    for i, fold in enumerate(fold_results):
        print('--------------------------------')
        print(f'Fold {i + 1}')
        print(f'Best Precision: {fold[0]:.4f}')
        print(f'Best Recall: {fold[1]:.4f}')
        print(f'Best F1 Score: {fold[2]:.4f}')
        print(f'Best Accuracy: {fold[3]:.4f}')
        print(f'Best AUC: {fold[4]:.4f}')

if __name__ == '__main__':
    main()