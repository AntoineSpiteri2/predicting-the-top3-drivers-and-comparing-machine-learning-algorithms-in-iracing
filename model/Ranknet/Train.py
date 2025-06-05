import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
from pandas.errors import SettingWithCopyWarning
from sklearn.experimental import enable_iterative_imputer  # Move this import to the top
from sklearn.impute import IterativeImputer, SimpleImputer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.stats import spearmanr

import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

import torch.nn as nn

class RankNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn2(self.fc2(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn3(self.fc3(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.bn4(self.fc4(x))
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        return x


def pairwise_loss(y_pred, y_true, bottom_weight=2.0):
    """
    Classic RankNet binary-cross-entropy on pairwise signs
    with optional extra weight for pairs involving bottom buckets.
    """
    # y_pred, y_true: (batch, 1)
    diff_true  = y_true - y_true.t()          # (B,B)
    sign       = torch.sign(diff_true)        # -1,0,+1
    mask       = sign != 0                    # ignore ties
    diff_pred  = y_pred - y_pred.t()          # (B,B)

    # up-weight any pair where either item is in bottom 20 %
    bottom = (y_true < 0.2) | (y_true.t() < 0.2)
    w      = torch.where(bottom, bottom_weight, 1.0)

    loss = (w*F.softplus(-sign*diff_pred))[mask].mean()
    return loss


def model_predict(net,X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return net(X_tensor).detach().numpy()

def trainRank(data):
    scaler = MinMaxScaler(feature_range=(0, 1))

    features = data.columns.tolist()
    featurestest = data.columns.tolist()

    race_ids = data['RaceID']


    featurestest.remove('Drv_RankScoreAdjusted')
    featurestest.remove('RankScoreAdjusted_norm')
    features.remove('RaceID')
    features.remove('CarIdx')
    features.remove('Drv_RankScoreAdjusted')
    features.remove('RankScoreAdjusted_norm')
    features.remove('Race_RaceDate')
    featurestest.remove('Race_RaceDate')
    
    leak_cols = [
    "LiveScoreAdjusted", "LapCompletionRate",
    "WeightedLiveScoreAdjusted"
    ]
    features = [col for col in features if col not in leak_cols]

    groups = data['RaceID'].values
    X = data[features]
    Xproof = data[featurestest]
    y = data['RankScoreAdjusted_norm']

# ───────────────────────── Initial Train/Test Split ─────────────────────────

    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups))

    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    
    race_ids_train = race_ids.iloc[train_idx]
    race_ids_test = race_ids.iloc[test_idx] 
    
    # Split proof set
    print(race_ids_train.unique())
    print(race_ids_test.unique())

    train_idx_proof, test_idx_proof = next(gss.split(Xproof, y, groups))
    X_test_proof = Xproof.iloc[test_idx_proof]


    # ─────────────── Imputation & Scaling on Global Train/Test ───────────────

    # imputer = IterativeImputer(random_state=42)
    imputer = SimpleImputer(strategy="mean") 

    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    np.set_printoptions(suppress=True)

    input_dim = X_train_imputed.shape[1]
    
    # ─────────────── Prepare tensors for the “outer” test───────────────────────

    print("PyTorch version:", torch.__version__)
    print("CUDA version in PyTorch:", torch.version.cuda)
    print("Is CUDA available?", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    print("Using device:", device)
    
    # ─────────────── Train one global RankNet to save as model₁ ───────────────

    model = RankNet(input_dim=input_dim)
    model = model.to(device)
    
    # adjust these parameters to control the scheduler for model training and validation
    # factor_Metric and patience_Metric are used to control the learning rate scheduler
    # patience_Metric is the number of epochs with no improvement after which learning rate will be reduced
    factor_Metric = 0.5
    patience_Metric = 500
    mode_metric = 'min'  # 'min' for loss, 'max' for accuracy
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode=mode_metric,
                                                       factor=factor_Metric,
                                                       patience=patience_Metric)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)
    


    X_train_tensor = torch.tensor(X_train_imputed, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test_imputed, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)


    # adjust the number of epochs to control the training time and validation training time
    num_epochs = 8000
    tensor_ds  = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    loader     = torch.utils.data.DataLoader(tensor_ds, batch_size=512, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = pairwise_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
               # ——— 2) COMPUTE VALIDATION LOSS ———
            model.eval()
            
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in loader:
                    out_val = model(x_val)
                    l = pairwise_loss(out_val, y_val)
                    val_losses.append(l.item() if torch.is_tensor(l) else l)
            val_loss = sum(val_losses) / len(val_losses)

            # ——— 3) STEP THE SCHEDULER WITH THE VALIDATION LOSS ———
            scheduler.step(val_loss)

            # ——— 4) PRINT YOUR METRICS ———
            if (epoch + 1) % 10 == 0:
                # Note: loss here is the last batch’s training loss
                print(f'Epoch [{epoch + 1}/{num_epochs}], '
                    f'Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')


    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).cpu().numpy().flatten()

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)


    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.normpath(os.path.join(script_dir, "../Ranknet/Saved_models/best_Rank_Net_model.pth"))
    scaler_save_path = os.path.normpath(os.path.join(script_dir, "../Ranknet/Saved_models/Rank_Net_scaler.joblib"))

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    dump(scaler, scaler_save_path)
    
# ---------------------------------------START OF  validation code-------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------------

    spearman_per_group = []
    top3_accuracy = []
    X_test_proof['PredictedRankScore'] = y_pred
    X_test_proof['ActualRankScore'] = y_test.values
    
    
    
    # ─────────────── GroupKFold with Scheduler ───────────────

    
    unique_groups = np.unique(groups_test)
    group_kfold = GroupKFold(n_splits=10)
    
    spearman_per_fold = []
    top3_accuracy_fold = []
    
    for fold_idx, (fold_train_idx, fold_test_idx) in enumerate(group_kfold.split(X_scaled, y, groups=groups)):
        
        # ----- Split into train‐fold and test‐fold using indices -----

        X_train_fold = X.iloc[fold_train_idx]
        X_test_fold = X.iloc[fold_test_idx]
        y_train_fold = y.iloc[fold_train_idx]
        y_test_fold = y.iloc[fold_test_idx]
        race_ids_train_fold = race_ids.iloc[fold_train_idx]
        race_ids_test_fold = race_ids.iloc[fold_test_idx]


        # ----- Scale the fold’s train + test feature subsets -----

        scaler_fold = MinMaxScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        
        
        # ----- Impute missing values (mean strategy) -----
        imputer_fold =  SimpleImputer(strategy="mean") 

        X_train_fold_imputed = imputer_fold.fit_transform(X_train_fold_scaled)
        X_test_fold_imputed = imputer_fold.transform(X_test_fold_scaled)
        
        model_fold = RankNet(input_dim=X_train_fold_imputed.shape[1]).to(device)
        optimizer_fold = torch.optim.AdamW(model_fold.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler_fold = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_fold,
                                                       mode=mode_metric,
                                                       factor=factor_Metric,
                                                       patience=patience_Metric)

        
        X_train_tensor_fold = torch.tensor(X_train_fold_imputed, dtype=torch.float32).to(device)
        y_train_tensor_fold = torch.tensor(y_train_fold.values, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor_fold = torch.tensor(X_test_fold_imputed, dtype=torch.float32).to(device)
        y_test_tensor_fold = torch.tensor(y_test_fold.values, dtype=torch.float32).view(-1, 1).to(device)
        
        num_epochs = num_epochs  # Set a fixed number of epochs for training
        tensor_fold  = torch.utils.data.TensorDataset(X_train_tensor_fold, y_train_tensor_fold)
        loader_fold     = torch.utils.data.DataLoader(tensor_fold, batch_size=512, shuffle=True)
        for epoch in range(num_epochs):
            model_fold.train()
            optimizer_fold.zero_grad()
            outputs = model_fold(X_train_tensor_fold)
            loss = pairwise_loss(outputs, y_train_tensor_fold)
            loss.backward()
            optimizer_fold.step()
               # ——— 2) Compute validation loss ———
            model_fold.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in loader_fold:
                    out_val = model_fold(x_val)
                    l = pairwise_loss(out_val, y_val)
                    val_losses.append(l.item() if torch.is_tensor(l) else l)
            val_loss = sum(val_losses) / len(val_losses)

            # ——— 3) Step the scheduler with val_loss ———
            scheduler_fold.step(val_loss)


            # ——— 4) Print both train & val losses every epoch ———
            print(f"Fold {fold_train_idx[fold_idx]} | Epoch {epoch+1:3d} | "
                f"Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")





        model_fold.eval()
        with torch.no_grad():
            y_pred = model_fold(X_test_tensor_fold).cpu().numpy().flatten()
        
        X_test_fold['PredictedRankScore'] = y_pred
        X_test_fold['ActualRankScore'] = y_test_fold.values
        

        
        spearman_per_group = []
        top3_accuracy = []
        
        for group in np.unique(race_ids_test_fold):
            group_data = X_test_fold[race_ids_test_fold == group]
            group_data_sorted = group_data.sort_values(by='PredictedRankScore', ascending=False)
            
            group_spearman, _ = spearmanr(group_data_sorted["ActualRankScore"], group_data_sorted["PredictedRankScore"])
            spearman_per_group.append(group_spearman)
            
            actual_top3 = group_data_sorted.sort_values("ActualRankScore", ascending=False).head(3).index
            predicted_top3 = group_data_sorted.sort_values("PredictedRankScore", ascending=False).head(3).index
            top3_match = len(set(actual_top3).intersection(set(predicted_top3))) / 3
            top3_accuracy.append(top3_match)
        
        spearman_avg = np.mean(spearman_per_group)
        top3_avg = np.mean(top3_accuracy) * 100  # expressed as percentage
        spearman_per_fold.append(spearman_avg)
        top3_accuracy_fold.append(top3_avg)

    final_spearman = np.mean(spearman_per_fold)
    final_top3 = np.mean(top3_accuracy_fold)
    
    
    # ─────────────── Bootstrap with Scheduler ───────────────


    bootstrap_rmse = []
    unique_groups = np.unique(groups_train)
    n_iterations = 30
    input_dim = X_train_imputed.shape[1]

    for i in range(n_iterations):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = [idx for idx, grp in enumerate(groups_train) if grp in sampled_groups]
        
        # Use .iloc if pandas objects
        X_boot = X_train_imputed[indices]
        y_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        # Debug: check distribution of y_boot
        print(f"Iteration {i+1}: y_boot min: {np.min(y_boot.values)}, max: {np.max(y_boot.values)}, std: {np.std(y_boot.values)}")
        
        # Skip iteration if sample is degenerate
        if np.std(y_boot.values) < 1e-6 or len(y_boot) < 2:
            print(f"Skipping iteration {i+1} due to degenerate sample")
            continue
        
        model_boot= RankNet(input_dim=input_dim).to(device)
        optimizer_boot = torch.optim.AdamW(model_boot.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler_boot = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_boot,
                                                       mode=mode_metric,
                                                       factor=factor_Metric,
                                                       patience=patience_Metric)
        
        X_train_tensor_boot = torch.tensor(np.array(X_boot), dtype=torch.float32).to(device)
        y_train_tensor_boot = torch.tensor(y_boot.values, dtype=torch.float32).view(-1, 1).to(device)
        X_test_tensor_boot = torch.tensor(X_test_imputed, dtype=torch.float32).to(device)
        y_test_tensor_boot = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)
        
        num_epochs = num_epochs
        tensor_boot  = torch.utils.data.TensorDataset(X_train_tensor_boot, y_train_tensor_boot)
        loader_boot     = torch.utils.data.DataLoader(tensor_boot, batch_size=512, shuffle=True)
        
        for epoch in range(num_epochs):
            model_boot.train()
            optimizer_boot.zero_grad()
            outputs = model_boot(X_train_tensor_boot)
            loss = pairwise_loss(outputs, y_train_tensor_boot)
            if torch.isnan(loss):
                print("Loss is NaN at epoch", epoch, "in iteration", i+1)
                break
            loss.backward()
            optimizer_boot.step()
            
            
                # ---- now run a quick validation pass to get val_loss ----
            model_boot.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in loader_boot:
                    out_val = model_boot(x_val)
                    l = pairwise_loss(out_val, y_val)
                    val_losses.append(l.item() if torch.is_tensor(l) else l)
            val_loss = sum(val_losses) / len(val_losses)

            # ——— 3) Step the scheduler with val_loss ———
            scheduler_boot.step(val_loss)


            if (epoch + 1) % 20 == 0:
                print(f"Iteration {i+1}, Epoch {epoch+1}, "
                    f"Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

        
        model_boot.eval()
        with torch.no_grad():
            y_pred = model_boot(X_test_tensor_boot).cpu().numpy().flatten()

            print("Initial outputs stats: min =", y_pred.min().item(), "max =", y_pred.max().item())

        y_test_np = y_test_tensor.cpu().numpy().flatten()
        rmse = spearmanr(y_test_np, y_pred)
        bootstrap_rmse.append(rmse[0])

    mean_rmse = np.mean(bootstrap_rmse)
    std_rmse = np.std(bootstrap_rmse)
    # print("Mean RMSE:", mean_rmse, "Std RMSE:", std_rmse)

    
    
    # ─────────────── Learning Curve with Scheduler ───────────────

    train_errors = []
    test_errors = []

    unique_groups = np.unique(groups_train)
    n_groups = len(unique_groups)
    fractions=np.linspace(0.1, 1.0, 10)
     
    for frac in fractions:
        
        # ---- learning cruve setup for training and test errors ----
        n_sample_groups = max(1, int(frac * n_groups))
        sampled_groups_lc = np.random.choice(unique_groups, size=n_sample_groups, replace=False)
        indices_lc = [idx for idx, grp in enumerate(groups_train) if grp in sampled_groups_lc]

        X_train_subset = X_train_imputed[indices_lc]

        y_train_subset = y_train.iloc[indices_lc]
        group_subset   = np.array([groups_train[i] for i in indices_lc])  # shape: (N_subset,)


        gss_small = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        idx_tr_sub, idx_val_sub = next(gss_small.split(X_train_subset, y_train_subset, groups=group_subset))

        X_tr_sub = X_train_subset[idx_tr_sub]
        y_tr_sub = y_train_subset.iloc[idx_tr_sub]

        X_val_sub = X_train_subset[idx_val_sub]
        y_val_sub = y_train_subset.iloc[idx_val_sub]


        print(f"Fraction: {frac:.2f} | Sampled groups: {n_sample_groups} | "
          f"Train subset shape: {X_train_subset.shape} | "
          f"y_train subset std: {np.std(y_train_subset.values):.4f}")
        
        model_lc = RankNet(input_dim=input_dim).to(device)
        optimizer_lc = torch.optim.AdamW(model_lc.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler_lc = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_lc,
                                                       mode=mode_metric,
                                                       factor=factor_Metric,
                                                       patience=patience_Metric)
    
        X_tr_tensor  = torch.tensor(X_tr_sub, dtype=torch.float32).to(device)
        y_tr_tensor  = torch.tensor(y_tr_sub.values, dtype=torch.float32).view(-1, 1).to(device)
        X_val_tensor = torch.tensor(X_val_sub, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_sub.values, dtype=torch.float32).view(-1, 1).to(device)
        X_te_tensor  = torch.tensor(X_test_imputed, dtype=torch.float32).to(device)
        y_te_tensor  = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

        tensor_lc  = torch.utils.data.TensorDataset(X_tr_tensor, y_tr_tensor)
        loader_lc     = torch.utils.data.DataLoader(tensor_lc, batch_size=512, shuffle=True)
        for epoch in range(num_epochs):
            model_lc.train()
            optimizer_lc.zero_grad()

            outputs = model_lc(X_tr_tensor)
            loss = pairwise_loss(outputs, y_tr_tensor)
            loss.backward()
            optimizer_lc.step()
                # 2) compute validation loss to feed the scheduler:
            model_lc.eval()
            val_losses = []
            with torch.no_grad():
                for x_val, y_val in loader_lc:
                    out_val = model_lc(x_val)
                    l = pairwise_loss(out_val, y_val)
                    val_losses.append(l.item() if torch.is_tensor(l) else l)
            val_loss = sum(val_losses) / len(val_losses)

            # ——— 3) Step the scheduler with val_loss ———
            scheduler_lc.step(val_loss)
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1:3d} | Train Loss: {loss.item():.6f} | Val Loss: {val_loss:.6f}")


        

        model_lc.eval()
        with torch.no_grad():
        # Evaluate on the training subset.
        #    X_train_tensor_subset = torch.tensor(np.array(X_train_subset), dtype=torch.float32).to(device)
           y_val_pred   = model_lc(X_val_tensor).cpu().numpy().flatten()


        # ---- Evaluate on the val_sub set spearman ----
        corr_train, _ = spearmanr(y_val_sub, y_val_pred )   # not RMSE!
        train_errors.append(1 - corr_train)
        
        
        with torch.no_grad():
            y_test_pred = model_lc(X_te_tensor).cpu().numpy().flatten()
            

        # ---- Evaluate on the test set spearman ----
        corr_test, _   = spearmanr(y_te_tensor.cpu().numpy().flatten(), y_test_pred)
        test_errors.append(1.0 - corr_test)
        
        

    model.cpu()
    X_test_cpu = X_test_tensor.cpu()

    model_cpu = RankNet(input_dim); model_cpu.load_state_dict(model.state_dict())
    model_cpu.eval()
    explainer = shap.GradientExplainer(model_cpu, X_test_cpu)

    shap_values = explainer.shap_values(X_test_cpu)

    # print(f"SHAP values shape before correction: {shap_values.shape}")
    shap_values = np.array(shap_values).reshape((X_test_cpu.shape[0], X_test_cpu.shape[1]))
    # print(f"SHAP values shape after correction: {shap_values.shape}")

    shap_values = shap_values.reshape((X_test_cpu.shape[0], X_test_cpu.shape[1]))


    return  final_spearman, final_top3,rmse,mae,spearman_corr,shap_values,X_test_imputed,  features,mean_rmse,std_rmse, train_errors , test_errors

