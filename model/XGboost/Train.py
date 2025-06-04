import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.discriminant_analysis import StandardScaler
from sklearn.isotonic import spearmanr
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRanker, plot_importance
from joblib import dump
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os


def train(data):



    scaler = MinMaxScaler(feature_range=(0, 1))

    features = data.columns.tolist()  # Convert Index object to list
    featurestest = data.columns.tolist()  # Convert Index object to list

    race_ids = data['RaceID']



    featurestest.remove('Drv_RankScoreAdjusted')
    featurestest.remove('RankScoreAdjusted_norm')




    features.remove('RaceID')
    features.remove('CarIdx')
    features.remove('Drv_RankScoreAdjusted')
    features.remove('RankScoreAdjusted_norm')
    features.remove('Race_RaceDate')
    featurestest.remove('Race_RaceDate')
    

    groups = data['RaceID'].values
    X = data[features]
    Xproof = data[featurestest]
    y = data['RankScoreAdjusted_norm']

    

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

   # Group-aware train-test split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X_scaled, y, groups))


    train_idx, test_idx = next(gss.split(X, y, groups=race_ids))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]
    
    race_ids_train = race_ids.iloc[train_idx]
    race_ids_test = race_ids.iloc[test_idx] 

    
    train_idx_proof, test_idx_proof = next(gss.split(Xproof, y, groups))


    X_test_proof =  Xproof.iloc[test_idx_proof]


    # Impute missing values in X_train and X_test
    imputer = SimpleImputer(strategy="mean") 


    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    np.set_printoptions(suppress=True)

    

    params = {
    "learning_rate": 0.01,         # Lower learning rate for smoother updates
    "max_depth": 1,                # Reduce tree depth to prevent overly complex trees
    "min_child_weight": 10,        # Increase to make splits more conservative
    "subsample": 0.7,              # Use a larger fraction of data for each tree to reduce variance
    "colsample_bytree": 0.6,       # Sample a higher fraction of features per tree
    "colsample_bylevel": 0.6,      # 50% per tree level
    "reg_alpha": 4,               # Stronger L1 regularization to encourage sparsity
    "reg_lambda": 4,              # Stronger L2 regularization for more penalty on complexity
    "gamma": 4,                    # Require a larger reduction in loss for splits
    "objective": "rank:pairwise",  # Ranking objective for pairwise ranking
    "eval_metric": "rmse",         # Evaluation metric for regression error
    "tree_method": "hist",         # GPU tree builder
    "device": "gpu:0",             # GPU acceleration
    "random_state": 42
}


    # Define and train the XGBRanker model
    xgb_model = XGBRanker(
        n_estimators=2000,
        **params
    )

    unique_groups, group_sizes = np.unique(groups_train, return_counts=True)

    xgb_model.fit(X_train_imputed, y_train, group=group_sizes)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.normpath(os.path.join(script_dir, "../XGboost/Saved_models/best_xgboost_model_rank.joblib"))
    file_path2 = os.path.normpath(os.path.join(script_dir, "../XGboost/Saved_models/xgboost_scaler.joblib"))
    file_path3 = os.path.normpath(os.path.join(script_dir, "../XGboost/Saved_models/xgboost_imputer.joblib"))
    # Save the trained model
    dump(xgb_model, file_path)
    
    dump(scaler, file_path2)
    dump(imputer, file_path3)



    # Predict and evaluate the model on the test set
    y_pred = xgb_model.predict(X_test_imputed)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)


        # Calculate SHAP values
     # Calculate SHAP values
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_train_imputed)



    dtrain = xgb.DMatrix(X_train_imputed, label=y_train)
    dtrain.set_group(group_sizes)


# ------------------START OF  validation code-------------------


    
    spearman_per_group = []
    top3_accuracy = []
    
    X_test_proof = X_test_proof.copy()  # Defragment the DataFrame to avoid PerformanceWarning
    X_test_proof['PredictedRankScore'] = y_pred
    X_test_proof['ActualRankScore'] = y_test.values  # Ensure alignment
    
    
    
    unique_groups = np.unique(groups_test)
    # print(unique_groups)
    group_kfold = GroupKFold(n_splits=10)
    
    spearman_per_fold = []
    top3_accuracy_fold = []
    
    unique_groups, group_sizes = np.unique(race_ids_train.values, return_counts=True)
    # print("Total rows from groups:", sum(group_sizes))  # Should equal X_train.shape[0]

    
    for fold_train_idx, fold_test_idx  in group_kfold.split(X_scaled, y, groups=race_ids):
        X_train_fold = X.iloc[fold_train_idx]
        X_test_fold = X.iloc[fold_test_idx]
        y_train_fold = y.iloc[fold_train_idx]
        y_test_fold = y.iloc[fold_test_idx]
        race_ids_train_fold = race_ids.iloc[fold_train_idx]

        scaler_fold = MinMaxScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        
        # imputer_fold = IterativeImputer(random_state=42)
        imputer_fold = SimpleImputer(strategy="mean") 


        X_train_fold_imputed = imputer_fold.fit_transform(X_train_fold_scaled)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        X_test_fold_imputed = imputer_fold.transform(X_test_fold_scaled)
         # Train the model on the training fold
         
        _, group_sizes_fold = np.unique(race_ids_train_fold.values, return_counts=True)
        # print("Group sizes for fold:", group_sizes_fold, "Total rows:", sum(group_sizes_fold))
    
    # Train the model on this fold:
        xgb_model.fit(X_train_fold_imputed, y_train_fold, group=group_sizes_fold)
        
        # Predict for the test fold
        y_pred = xgb_model.predict(X_test)
        
        # Create a DataFrame with your predictions and actual values.
        # Ensure that X_test has a 'RaceID' column to group by
        X_test_proof = pd.DataFrame(X_test.copy(), columns=X.columns)
        X_test_proof['RaceID'] = race_ids_test.values

        X_test_proof['PredictedRankScore'] = y_pred

        X_test_proof['ActualRankScore'] = y_test.values
        
        spearman_per_group = []
        top3_accuracy = []
        
        unique_groups = np.unique(groups_test)

        for group in unique_groups:
            
            group_data_proof = X_test_proof[X_test_proof['RaceID'] == group]
            
            group_data_sorted = group_data_proof.sort_values(by='PredictedRankScore', ascending=False)

            # print(f"Group {group} Rankings:")
            # print(group_data_sorted[['CarIdx', 'ActualRankScore', 'PredictedRankScore']])

            # Compute Spearman Rank Correlation
            group_spearman, _ = spearmanr(group_data_sorted["ActualRankScore"], group_data_sorted["PredictedRankScore"])
            spearman_per_group.append(group_spearman)
            
            # Check Top-3 Accuracy
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



    # Feature importance visualization
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)

    importances = (importances - importances.min()) / (importances.max() - importances.min())
    
    bootstrap_rmse = []
    unique_groups = np.unique(groups_train)
    n_iterations=30
    
    for i in range(n_iterations):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        # Get indices for rows that belong to these sampled groups
        indices = [idx for idx, grp in enumerate(groups_train) if grp in sampled_groups]
        
        # Create bootstrap sample
        X_boot = X_train[indices]
        y_boot = np.array(y_train)[indices]  # Ensure y is array-like
        
        _, group_sizes = np.unique(np.array(groups_train)[indices], return_counts=True)

        # Train on the bootstrap sample
        xgb_model.fit(X_boot, y_boot, group=group_sizes)
        y_pred = xgb_model.predict(X_test)
        
        rmse = spearmanr(y_test, y_pred)
        bootstrap_rmse.append(rmse[0])
    
    mean_rmse = np.mean(bootstrap_rmse)
    std_rmse = np.std(bootstrap_rmse)
    
    
    train_errors = []
    test_errors = []

    unique_groups = np.unique(groups_train)
    n_groups = len(unique_groups)
    
    fractions=np.linspace(0.1, 1.0, 10)
     
    for frac in fractions:
        
        # learning cruve setup
        
        # Determine number of groups to use for this fraction
        n_sample_groups = max(1, int(frac * n_groups))
        # Randomly sample groups (without replacement)
        sampled_groups = np.random.choice(unique_groups, size=n_sample_groups, replace=False)
        indices = [idx for idx, grp in enumerate(groups_train) if grp in sampled_groups]
        
        X_train_subset = X_train[indices]
        y_train_subset = np.array(y_train)[indices]

        _, group_sizes = np.unique(np.array(groups_train)[indices], return_counts=True)

        # Train the model on this subset
        xgb_model.fit(X_train_subset, y_train_subset, group=group_sizes)
        
        # Evaluate on the full train set for spearman correlation and top-3 accuracy
        y_train_pred = xgb_model.predict(X_train_subset)
        if np.all(y_train_subset == y_train_subset[0]) or np.all(y_train_pred == y_train_pred[0]):
            train_corr = np.nan
        else:
            train_corr, _ = spearmanr(y_train_subset, y_train_pred)
        train_errors.append(1 - (train_corr if not np.isnan(train_corr) else 0))
        
        
        # Evaluate on the full test set for spearman correlation and top-3 accuracy
        y_test_pred = xgb_model.predict(X_test)
        if np.all(y_test == y_test.iloc[0]) or np.all(y_test_pred == y_test_pred[0]):
            test_corr = np.nan
        else:
            test_corr, _ = spearmanr(y_test, y_test_pred)
        test_errors.append(1 - (test_corr if not np.isnan(test_corr) else 0))
        
        
        

    
    return  final_spearman, final_top3,rmse,mae,spearman_corr,shap_values,X_train_imputed, importances, indices, features, mean_rmse, std_rmse , train_errors, test_errors, xgb_model


