import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.isotonic import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
from sklearn.impute import IterativeImputer, SimpleImputer



def trainRa(data):

    # Define features and target

    features = data.columns.tolist()  # Convert Index object to list
    featurestest = data.columns.tolist()  # Con    features.remove('Race_RaceDate')

    race_ids = data['RaceID']

    features.remove('CarIdx')
    features.remove('Drv_RankScoreAdjusted')
    features.remove('RankScoreAdjusted_norm')
    features.remove('Race_RaceDate')


    Modelfeatures = features

    features.remove('RaceID')

    featurestest.remove('Drv_RankScoreAdjusted')
    featurestest.remove('RankScoreAdjusted_norm')
    featurestest.remove('Race_RaceDate')
    
    

    target = 'RankScoreAdjusted_norm'  # Define your target column

    # ------------------------------------------------------------
    unique_races = data['RaceID'].unique()
    np.random.shuffle(unique_races)

    # For an 80/20 split
    train_size = int(len(unique_races) * 0.7)
    train_races = unique_races[:train_size]
    test_races = unique_races[train_size:]

    train_df = data[data['RaceID'].isin(train_races)]
    test_df  = data[data['RaceID'].isin(test_races)]

    # Prepare the data splits
    X_train = train_df[features]
    y_train = train_df[target]
    X_test  = test_df[features]
    y_test  = test_df[target]
    
    race_ids_train = train_df['RaceID']
    race_ids_test = test_df['RaceID']

    
    #getting a copy for validation purposes
    X_test_proof  = test_df[featurestest]


    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)



    # imputer = IterativeImputer(random_state=42)
    imputer = SimpleImputer(strategy="mean") 


    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_test_imputed  = imputer.transform(X_test_scaled)


    # Define and train the RandomForestRegressor model
    rf_model = RandomForestRegressor(
        n_estimators=2000,      # lots of trees for stability
        max_depth=6,            # prevent overly deep, specialized splits
        min_samples_split=22,   # only split when ≥22 samples in a node
        min_samples_leaf=5,     # each leaf must have ≥5 samples
        max_features=0.5,       # only 50% of features considered at each split
        random_state=42,
        n_jobs=-1
    )



    # Train the model
    rf_model.fit(X_train_imputed, y_train)
    
    
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path = os.path.normpath(os.path.join(script_dir, "../RandomForest/Saved_models/best_random_forest_model.joblib"))
    file_path2 = os.path.normpath(os.path.join(script_dir, "../RandomForest/Saved_models/random_forest_scaler.joblib"))

    # Save the trained model
    dump(rf_model, file_path)
    dump(scaler,file_path2)
    dump(imputer,os.path.normpath(os.path.join(script_dir, "../RandomForest/Saved_models/random_forest_imputer.joblib")))

    # Predict and evaluate the model on the test set
    y_pred = rf_model.predict(X_test_imputed)
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)

    
    
    
# ------------------START OF  validation code-------------------

    
        # Calculate SHAP values
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer(X_train_imputed)

    # Plot SHAP summary plot
    # shap.summary_plot(shap_values, X_train_imputed, feature_names=features)
    group_kfold = GroupKFold(n_splits=10)
    
    spearman_per_fold = []
    top3_accuracy_fold = []
    
    for fold_train_idx, fold_test_idx in group_kfold.split(X_train, y_train, groups=race_ids_train):
        X_train_fold = X_train_imputed[fold_train_idx]
        X_test_fold = X_train_imputed[fold_test_idx]
        y_train_fold = y_train.iloc[fold_train_idx]
        y_test_fold = y_train.iloc[fold_test_idx]
        race_ids_train_fold = race_ids_train.iloc[fold_train_idx]
        race_ids_test_fold = race_ids_train.iloc[fold_test_idx].reset_index(drop=True)
        # Train the model on the training fold
        
        # Extract RaceID from the original data for the test fold,
        # then reset the index to ensure alignment with predictions
        X_test_proof = pd.DataFrame(X_test_fold.copy(), columns=Modelfeatures)
        X_test_proof['RaceID'] = race_ids_test_fold.reset_index(drop=True).values
        # Scale and impute the fold data
        scaler_fold = MinMaxScaler()
        X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler_fold.transform(X_test_fold)
        
        # imputer_fold = IterativeImputer(random_state=42)
        imputer_fold = SimpleImputer(strategy="mean") 


        X_train_fold_imputed = imputer_fold.fit_transform(X_train_fold_scaled)
        X_test_fold_imputed = imputer_fold.transform(X_test_fold_scaled)
        _, group_sizes_fold = np.unique(race_ids_train_fold.values, return_counts=True)

        print("Group sizes for fold:", group_sizes_fold, "Total rows:", sum(group_sizes_fold))

        rf_model.fit(X_train_fold_imputed, y_train_fold)
        
        # Predict for the test fold
        y_pred_fold = rf_model.predict(X_test_fold_imputed)
        
        X_test_proof_fold = X_test_proof.copy()
        X_test_proof_fold['PredictedRankScore'] = y_pred_fold
        X_test_proof_fold['ActualRankScore'] = y_test_fold.values
        
        spearman_per_group = []
        top3_accuracy = []
        
        unique_groups = np.unique(race_ids_test_fold)

        for group in unique_groups:
            group_data_proof = X_test_proof_fold[X_test_proof_fold['RaceID'] == group]
            group_data_sorted = group_data_proof.sort_values(by='PredictedRankScore', ascending=False)

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
    


    importances = rf_model.feature_importances_
    indices = np.argsort(importances)

    importances = (importances - importances.min()) / (importances.max() - importances.min())


    bootstrap_rmse = []
    unique_groups = race_ids_train.unique()
    unque_groups_test =  race_ids_test.unique()  # Sort for consistent sampling
    n_iterations=30
    
    for i in range(n_iterations):
        # Sample groups with replacement
        sampled_groups = np.random.choice(unique_groups, size=len(unique_groups), replace=True)
        # Get indices for rows that belong to these sampled groups
        indices = [idx for idx, grp in enumerate(race_ids_train) if grp in sampled_groups]
        
        # Create bootstrap sample
        X_boot = X_train.iloc[indices]
        y_boot = np.array(y_train)[indices]  # Ensure y is array-like
        
        _, group_sizes = np.unique(np.array(race_ids_train)[indices], return_counts=True)

        print("Group sizes for bootstrap:", group_sizes, "Total rows:", sum(group_sizes))

        # Train on the bootstrap sample
        rf_model.fit(X_boot, y_boot)
        y_pred = rf_model.predict(X_test)
        
        rmse = spearmanr(y_test, y_pred)
        bootstrap_rmse.append(rmse[0])
    
    mean_rmse = np.mean(bootstrap_rmse)
    std_rmse = np.std(bootstrap_rmse)
    
    
    train_errors = []
    test_errors = []
    train_top3_rf    = []
    test_top3_rf         = []
    n_groups = len(unique_groups)
    n_test_groups = len(unque_groups_test)
    fractions=np.linspace(0.1, 1.0, 10)
     
    for frac in fractions:
        # Determine number of groups to use for this fraction
        n_sample_groups = max(1, int(frac * n_groups))
        # Randomly sample groups (without replacement)
        sampled_groups = np.random.choice(unique_groups, size=n_sample_groups, replace=False)
        sampled_groups_test = np.random.choice(unque_groups_test, size=n_test_groups, replace=False)
        indices = [idx for idx, grp in enumerate(race_ids_train) if grp in sampled_groups]
        
        indicestest = [idx for idx, grp in enumerate(race_ids_test) if grp in sampled_groups_test]

        
        X_train_subset = X_train.iloc[indices]
        y_train_subset = np.array(y_train)[indices]
        grp_subset     = np.array(race_ids_train)[indices]  # array length N_subset
        grp_testsubset = np.array(race_ids_test)[indicestest] 
        
        _, group_sizes = np.unique(np.array(race_ids_train)[indices], return_counts=True)

        print("Group sizes for learning curve:", group_sizes, "Total rows:", sum(group_sizes))
        # Train the model on this subset
        rf_model.fit(X_train_subset, y_train_subset)

        # Evaluate on training subset
        y_train_pred = rf_model.predict(X_train_subset)
        train_rmse = spearmanr(y_train_subset, y_train_pred)
        train_errors.append(1 - train_rmse[0])
        
        
        # Evaluate on the full test set
        y_test_pred = rf_model.predict(X_test)
        test_rmse = spearmanr(y_test, y_test_pred)
        test_errors.append(1 - test_rmse[0])
        
  

    return  final_spearman, final_top3,rmse,mae,spearman_corr,shap_values,X_train_imputed, importances, indices, features, mean_rmse, std_rmse , train_errors, test_errors

