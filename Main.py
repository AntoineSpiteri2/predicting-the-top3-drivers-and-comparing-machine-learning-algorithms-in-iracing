#!/usr/bin/env python3
import asyncio
import numpy as np
import matplotlib.pyplot as plt
import shap
from collections import Counter

from utils.IracingSDKConnection import get_iracing_sdk_connection
from utils.DatabaseConnection import set_database
from Data_processing_and_inputting.historical_data_pipeline import ProcessIracingHistoicalData
from Data_processing_and_inputting.live_data_pipeline import ProcessIracingLiveData
from model.FetchDatabaseData import process_all_races
from utils.NormTarget import normalize_group
from model.XGboost.Train import train as train_xgb
from model.RandomForest.Train import trainRa as train_rf
from model.Ranknet.Train import trainRank as train_ranknet
from Data_fetching.FetchLive import IRacingDataHandler
from utils.DatabaseConnection import execute_query


def collect_data(test_mode: bool, save_historical: bool) -> None:
    """
    Connects to the iRacing SDK and collects either live or historical data.
    """
    sdk = get_iracing_sdk_connection()
    if sdk is None:
        print("Failed to connect to iRacing SDK.")
        return

    if test_mode:
        
        asyncio.run(ProcessIracingLiveData(0, True))
    else:
        # 1) Early‐exit if any driver is already in DB
        handler  = IRacingDataHandler()
        upcoming = [int(d['custid']) for d in handler.fetch_driver_names_and_caridx()]
        rows     = execute_query("SELECT CustID FROM Driver")

        dupes_incoming = [cid for cid, cnt in Counter(upcoming).items() if cnt > 1]
        if dupes_incoming:
            print(f"⚠️ Duplicate CustID(s) detected in DB: {dupes_incoming}. Aborting data collection.")
            return
                # 2) fetch existing IDs from the DB (as ints)
        existing = {int(r[0]) for r in execute_query("SELECT CustID FROM Driver")}

        # 3) detect overlaps between upcoming and existing
        dupes_in_db = [cid for cid in upcoming if cid in existing]
        if dupes_in_db:
            print(f"⚠️ Duplicate CustID(s) detected in DB: {dupes_in_db}. Aborting data collection.")
            return
        race_id = asyncio.run(ProcessIracingHistoicalData(save_historical))
        if race_id is not None:
               # 3) Find which upcoming drivers didn’t get any historical rows
            rows_live = execute_query(f"SELECT CustID FROM Driver WHERE RaceID = {race_id}")
            valid     = [int(r[0]) for r in rows_live]

            missing = [cid for cid in upcoming if cid not in valid]
            if missing:
                print(f"⚠️  No historical data for CustID(s): {missing}.  aborting data collection  \n as driver has disabled all kinds of data collection even live in thier personal settings")
                execute_query(f"DELETE FROM Driver where raceid = {race_id}")
                execute_query(f"delete Race where raceid = {race_id}")
                return

            asyncio.run(ProcessIracingLiveData(race_id, True,missing))

def train_models() -> None:
    # Data preparation
    data = process_all_races(True, False)
    data = normalize_group(data)
    # data = data[~data['Drv_Disqualified']]
    
    
    # Fill NaN values in event columns with 0 
    logical_suffixes = ["_incident", "_inpits", "_offtrack", '_inpits_']
    
    incident_cols = [
        col for col in data.columns if any(col.endswith(suf) for suf in logical_suffixes)
    ]
    
    data[incident_cols] = data[incident_cols].fillna(0)

    
    # data = data[np.isfinite(data['RankScoreAdjusted_norm'])] # remove drivers with no data

    print("\n\ntraining xgboost")
    xgboost_metrics = train_xgb(data)
    print("\n\ntraining Random forest")
    rf_metrics = train_rf(data)
    print("\n\nTraining Ranknet")
    ranknet_metrics = train_ranknet(data)
    print("\n\nComparing metrics of models")

    # Extract metrics for each model
    def extract_metrics(metrics, idx_map):
        return {name: metrics[i] for name, i in idx_map.items()}

    idx_map_xgb = {
        'spearman': 0, 'top3': 1, 'rmse': 2, 'mae': 3,
        'shap': 5, 'X': 6, 'features': 9,
        'bootstrap_rmse': 10, 'bootstrap_std_rmse': 11,
        'lc_train': 12, 'lc_test': 13
    }
    idx_map_rf = idx_map_xgb
    idx_map_ranknet = {
        'spearman': 0, 'top3': 1, 'rmse': 2, 'mae': 3,
        'shap': 5, 'X': 6, 'features': 7,
        'bootstrap_rmse': 8, 'bootstrap_std_rmse': 9,
        'lc_train': 10, 'lc_test': 11
    }

    xgb = extract_metrics(xgboost_metrics, idx_map_xgb)
    rf = extract_metrics(rf_metrics, idx_map_rf)
    rn = extract_metrics(ranknet_metrics, idx_map_ranknet)

    metrics = ['RMSE', 'MAE', 'Spearman', 'Top 3 Accuracy']
    xgboost_values = [round(xgb['rmse'].statistic, 4), xgb['mae'], xgb['spearman'], xgb['top3']]
    rf_values = [round(rf['rmse'].statistic, 4), rf['mae'], rf['spearman'], rf['top3']]
    rn_values = [round(rn['rmse'].statistic, 4), rn['mae'], rn['spearman'], rn['top3']]

    # --- Plots (unchanged) ---
    x = np.arange(len(metrics))
    width = 0.2

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle('Comparison of Model Metrics')

    def add_value_labels(ax, spacing=5):
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2
            space = spacing
            va = 'bottom'
            if y_value < 0:
                space *= -1
                va = 'top'
            label = "{:.2f}".format(y_value)
            ax.annotate(
                label,
                (x_value, y_value),
                xytext=(0, space),
                textcoords="offset points",
                ha='center',
                va=va)

    # RMSE
    axs[0, 0].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgboost_values[0], rf_values[0], rn_values[0]], color=['blue', 'green', 'red'])
    axs[0, 0].set_title('RMSE')
    axs[0, 0].set_ylabel('Values')
    add_value_labels(axs[0, 0])

    # MAE
    axs[0, 1].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgboost_values[1], rf_values[1], rn_values[1]], color=['blue', 'green', 'red'])
    axs[0, 1].set_title('MAE')
    axs[0, 1].set_ylabel('Values')
    add_value_labels(axs[0, 1])

    # Spearman Test
    axs[1, 0].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgboost_values[2], rf_values[2], rn_values[2]], color=['blue', 'green', 'red'])
    axs[1, 0].set_title('Spearman Test Kfold avg')
    axs[1, 0].set_ylabel('Values')
    add_value_labels(axs[1, 0])

    # Top 3 Accuracy
    axs[1, 1].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgboost_values[3], rf_values[3], rn_values[3]], color=['blue', 'green', 'red'])
    axs[1, 1].set_title('Top 3 Accuracy Kfold avg')
    axs[1, 1].set_ylabel('Values')
    add_value_labels(axs[1, 1])

    # Bootstrap RMSE
    axs[2, 0].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgb['bootstrap_rmse'], rf['bootstrap_rmse'], rn['bootstrap_rmse']], color=['blue', 'green', 'red'])
    axs[2, 0].set_title('Bootstrap spearmanr')
    axs[2, 0].set_ylabel('Values')
    add_value_labels(axs[2, 0])

    # Bootstrap Std RMSE
    axs[2, 1].bar(['XGBoost', 'Random Forest', 'RankNet'], [xgb['bootstrap_std_rmse'], rf['bootstrap_std_rmse'], rn['bootstrap_std_rmse']], color=['blue', 'green', 'red'])
    axs[2, 1].set_title('Bootstrap Std spearmanr')
    axs[2, 1].set_ylabel('Values')
    add_value_labels(axs[2, 1])
    
      # <-- layout tweak here
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()

    # Learning Curve Train and Test Errors - Separate Plots for Each Model
    models = [
        ('XGBoost', xgb, 'blue'),
        ('Random Forest', rf, 'green'),
        ('RankNet', rn, 'red')
    ]

    for name, model, color in models:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle(f'{name} Learning Curves')

        # Train Error
        axs[0].plot(model['lc_train'], label=f'{name} Train', color=color)
        axs[0].set_title('Learning Curve Train Error')
        axs[0].set_ylabel('1 - Spearman Correlation (Lower is Better)')
        axs[0].legend()
        for i, v in enumerate(model['lc_train']):
            axs[0].text(i, v, f"{v:.2f}", ha='center', va='bottom', color=color)

        # Test Error
        axs[1].plot(model['lc_test'], label=f'{name} Test', color=color)
        axs[1].set_title('Learning Curve Test Error')
        axs[1].set_ylabel('1 - Spearman Correlation (Lower is Better)')
        axs[1].set_xlabel('Fraction Index (10%, 20%, …, 100%)')
        axs[1].legend()
        for i, v in enumerate(model['lc_test']):
            axs[1].text(i, v, f"{v:.2f}", ha='center', va='bottom', color=color)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    

    # Create tables of differences between models
    differences = {
        'Metric': metrics,
        'XGBoost - Random Forest': np.array(xgboost_values) - np.array(rf_values),
        'XGBoost - RankNet': np.array(xgboost_values) - np.array(rn_values),
        'Random Forest - RankNet': np.array(rf_values) - np.array(rn_values)
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    table_data = [
        [metric] + list(diff)
        for metric, diff in zip(
            differences['Metric'],
            zip(differences['XGBoost - Random Forest'],
                differences['XGBoost - RankNet'],
                differences['Random Forest - RankNet'])
        )
    ]
    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'XGBoost - Random Forest', 'XGBoost - RankNet', 'Random Forest - RankNet'],
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title('Differences Between Models')
    fig.subplots_adjust(bottom=0.2, top=0.85)
    fig.tight_layout()
    plt.show()

    # SHAP plots
    shap.summary_plot(xgb['shap'], xgb['X'], feature_names=xgb['features'], max_display=50)
    shap.summary_plot(rf['shap'], rf['X'], feature_names=rf['features'], max_display=50)
    shap.summary_plot(rn['shap'], rn['X'], feature_names=rn['features'], max_display=50)
    shap.summary_plot(xgb['shap'], xgb['X'], feature_names=xgb['features'], plot_type="bar", max_display=50)
    shap.summary_plot(rf['shap'], rf['X'], feature_names=rf['features'], plot_type="bar", max_display=50)
    shap.summary_plot(rn['shap'], rn['X'], feature_names=rn['features'], plot_type="bar", max_display=50)

def main():
    while True:
        print("\n===== iRacing Menu =====")
        print("1. Test-mode data collection (no DB write)")
        print("2. Live data collection (save to DB)")
        print("3. Train & evaluate models")
        print("4. Exit")
        choice = input("Select an option [1-4]: ").strip()

        if choice == '1':
            collect_data(test_mode=True, save_historical=False)
        elif choice == '2':
            set_database('racing_data')
            collect_data(test_mode=False, save_historical=False)
        elif choice == '3':
            train_models()
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice, please enter 1, 2, 3, or 4.")



if __name__ == "__main__":
    main()
