#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


# ==================== 超参数配置 ====================
class Config:
    INPUT_DIR = r"E:\4.联合用药参考\code-yj\1.Baseline实验\data\result"
    OUTPUT_DIR = r"E:\4.联合用药参考\code-yj\1.Baseline实验\result"
    MODEL_SAVE_DIR = r"E:\4.联合用药参考\code-yj\1.Baseline实验\models"  # 新增模型保存路径
    DRUG_COMB_MATRIX_FILE = "drug_comb_matrix.csv"
    FEATURE_MATRIX_FILE = "feature_matrix_h_concat_46890×212_pca95.pkl"
    RANDOM_SEEDS = [42, 123, 456, 789, 1000]
    TEST_SIZE = 0.2
    VAL_SIZE = 0.5
    BATCH_SIZE = 16
    NUM_EPOCHS = 200
    LEARNING_RATE = 0.00012

    # 设备配置
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 模型定义 ====================
class HSA_matrix_Regressor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.4):
        super(HSA_matrix_Regressor, self).__init__()

        f1 = 512
        f2 = 1024
        f3 = 2048
        f4 = 1024
        f5 = 512
        self.fc1 = nn.Linear(input_size, f1)
        self.bn1 = nn.BatchNorm1d(f1)

        self.fc2 = nn.Linear(f1, f2)
        self.bn2 = nn.BatchNorm1d(f2)
        #
        self.fc3 = nn.Linear(f2, f3)
        self.bn3 = nn.BatchNorm1d(f3)

        self.fc4 = nn.Linear(f3, f4)
        self.bn4 = nn.BatchNorm1d(f4)

        self.fc5 = nn.Linear(f4, f5) # 占位
        self.bn5 = nn.BatchNorm1d(f5)

        self.fc7 = nn.Linear(f5, 1)  # 输出层

        self.leaky_relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        out = self.fc7(x)
        return out


# ==================== 训练与评估函数 ====================
def train_model(model, train_loader, val_loader, seed):
    """训练模型并返回最优模型和损失记录"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_weights = None

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = model.state_dict().copy()
            # 保存模型到文件
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
            model_save_path = os.path.join(
                Config.MODEL_SAVE_DIR,
                f"HSA_matrix_best_model_seed_{seed}.pth"
            )
            torch.save(best_model_weights, model_save_path)

    return best_model_weights, train_losses, val_losses


def evaluate_model(model, data_loader):
    """评估模型性能"""
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
            outputs = model(inputs).squeeze(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics = {
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'Pearson': pearsonr(y_true, y_pred)[0]
    }

    return metrics, y_true, y_pred


# ==================== 主函数 ====================
def main():
    # 打印超参数配置
    print("=" * 50)
    print("超参数配置:")
    print(f"- 数据路径: {Config.INPUT_DIR}")
    print(f"- 模型保存路径: {Config.MODEL_SAVE_DIR}")
    print(f"- 特征矩阵: {Config.FEATURE_MATRIX_FILE}")
    print(f"- 训练参数: LR={Config.LEARNING_RATE}, Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}")
    print(f"- 使用设备: {Config.DEVICE}")
    print("=" * 50 + "\n")

    # 加载数据
    print("加载数据...")
    df_comb = pd.read_csv(os.path.join(Config.INPUT_DIR, Config.DRUG_COMB_MATRIX_FILE))
    feature_matrix = pd.read_pickle(os.path.join(Config.INPUT_DIR, Config.FEATURE_MATRIX_FILE))
    X = feature_matrix
    y = df_comb['HSA_matrix']

    val_results = {
        'MSE': [], 'RMSE': [], 'R2': [], 'Pearson': []
    }
    test_results = {
        'MSE': [], 'RMSE': [], 'R2': [], 'Pearson': []
    }
    for seed in Config.RANDOM_SEEDS:
        print(f"\n=== 实验开始 (随机种子: {seed}) ===")
        torch.manual_seed(seed)
        np.random.seed(seed)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=Config.VAL_SIZE, random_state=seed)

        # 准备数据加载器
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train.values, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val.values, dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test.values, dtype=torch.float32)
        )

        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        model = HSA_matrix_Regressor(X_train.shape[1]).to(Config.DEVICE)

        print("开始训练...")
        best_weights, train_losses, val_losses = train_model(model, train_loader, val_loader, seed)

        model.load_state_dict(best_weights)

        test_metrics, y_test_true, y_test_pred = evaluate_model(model, test_loader)
        print("\n测试集性能:")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")
            test_results[name].append(value)

    # 计算平均结果
    print("\n=== 最终结果 ===")
    final_results = []
    for metric in ['MSE', 'RMSE', 'R2', 'Pearson']:
        val_values = val_results[metric]
        test_values = test_results[metric]

        val_mean = np.mean(val_values)
        val_std = np.std(val_values)
        test_mean = np.mean(test_values)
        test_std = np.std(test_values)

        print(f"{metric}:")
        print(f"  测试集: {test_mean:.4f} ± {test_std:.4f}")

        final_results.append({
            'Metric': metric,
            'Val_Mean': val_mean,
            'Val_Std': val_std,
            'Test_Mean': test_mean,
            'Test_Std': test_std,
            'All_Val_Values': [f"{v:.4f}" for v in val_values],
            'All_Test_Values': [f"{v:.4f}" for v in test_values]
        })

    # 保存结果
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    result_df = pd.DataFrame(final_results)
    output_path = os.path.join(Config.OUTPUT_DIR, "Ours_HSA_matrix_result.csv")
    result_df.to_csv(output_path, index=False)
    print(f"\n完整结果已保存至: {output_path}")


if __name__ == "__main__":
    main()