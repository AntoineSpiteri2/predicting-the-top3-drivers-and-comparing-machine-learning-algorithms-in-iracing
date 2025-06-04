# Predicting Top 3 Drivers and Comparing Machine Learning Algorithms in iRacing

## Project Overview

This project implements a hybrid pipeline that merges historical iRacing driver data with live telemetry to predict the top 3 finishers in real-time. By engineering time-decayed features and training various machine learning models (XGBoost, Random Forest, RankNet), the system aims to provide accurate predictions, enabling esports analytics and broadcast overlays. The predictions update every 3 seconds. RankNet has shown the best accuracy in testing.  SHAP analysis is performed to identify key predictors of success.

*   **Repository URL:** [https://github.com/AntoineSpiteri2/predicting-the-top3-drivers-and-comparing-machine-learning-algorithms-in-iracing](https://github.com/AntoineSpiteri2/predicting-the-top3-drivers-and-comparing-machine-learning-algorithms-in-iracing)

## Features and Functionality

*   **Data Fusion:** Combines historical driver performance data with live telemetry from the iRacing simulation.
*   **Feature Engineering:** Creates time-decayed features to emphasize recent performance and trends.
*   **Machine Learning Models:** Trains and compares XGBoost, Random Forest, and RankNet models for top-3 prediction.
*   **Real-time Prediction:** Provides updated predictions approximately every 3 seconds.
*   **SHAP Analysis:**  Identifies the most influential features used by the models.
*   **Database Integration:** Stores historical data and race results in a SQL Server database.
*   **Esports Applications:** Designed for use in esports analytics and broadcast overlays.
*   **Test Mode:** Allows running live data collection without writing to the database (for testing purposes).
*   **Data Normalization:** Normalizes target variables for better model performance.

## Technology Stack

*   **Python:** Primary programming language.
*   **iRacing SDK (irsdk):**  For accessing live telemetry data from iRacing.
*   **iRacing Data API:** For fetching historical driver and race data.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical computations.
*   **Scikit-learn:**  For machine learning algorithms (RandomForest) and data preprocessing (MinMaxScaler, IterativeImputer).
*   **XGBoost:** For gradient boosting models.
*   **PyTorch:**  For the RankNet model.
*   **Shap:** For SHAP (SHapley Additive exPlanations) analysis
*   **PyODBC:** For connecting to SQL Server databases.
*   **SQLAlchemy:** For database interaction, especially for bulk data insertion and complex queries.
*   **Matplotlib:** For creating visualizations (e.g., model comparison plots).
*   **Joblib:** For saving and loading trained models.
*   **Asyncio:** For asynchronous API calls.
*   **Threading:**  For running the asynchronous data insertion in the background.

## Prerequisites

1.  **iRacing Account:**  An active iRacing subscription is required to access the live telemetry and historical data.
2.  **iRacing SDK:**  The iRacing SDK must be installed and configured to allow the script to connect to the simulator.
3.  **iRacing Data API Credentials:** You need an iRacing username and password to authenticate with the iRacing Data API. Set these in `utils/IracingApiConnection.py`:

    ```python
    USERNAME = 'youremail'  # Replace with your iRacing username
    PASSWORD = 'yourpassword'  # Replace with your iRacing password
    ```

4.  **SQL Server Database:** A SQL Server database is required to store historical and live race data.
5.  **Python Dependencies:** Install the required Python packages using `pip`:

    ```bash
    pip install irsdk pandas numpy scikit-learn xgboost torch shap pyodbc sqlalchemy matplotlib joblib aiohttp
    ```

6.  **ODBC Driver:**  Install the ODBC Driver 17 for SQL Server.  The driver name should match what's configured in `utils/DatabaseConnection.py`.
7.  **Database Connection Settings:** Configure the database connection settings in `utils/DatabaseConnection.py`:

    ```python
    server = 'ANTOINEPC\\MSSQLSERVER01'  # Update with your server name
    database = 'racing_data'
    driver = '{ODBC Driver 17 for SQL Server}'
    ```

## Installation Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/AntoineSpiteri2/predicting-the-top3-drivers-and-comparing-machine-learning-algorithms-in-iracing.git
    cd predicting-the-top3-drivers-and-comparing-machine-learning-algorithms-in-iracing
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt # Create this file with all the dependencies you have installed above if missing
    ```

3.  **Configure iRacing API Credentials:**  Edit `utils/IracingApiConnection.py` and enter your iRacing username and password.
4.  **Configure Database Connection:** Edit `utils/DatabaseConnection.py` and enter your SQL Server connection details.
5.  **Optional: Remap CustID:** Run `remap_ids_and_drop_new_column_with_subsession.py` ONCE to remap CustID and subsessionID, which facilitates integration:

    ```bash
    python remap_ids_and_drop_new_column_with_subsession.py
    ```
    **Important:** This script modifies the database structure and data. **BACK UP YOUR DATABASE BEFORE RUNNING.**

## Usage Guide

1.  **Run the Main Script:** Execute `Main.py` to start the application:

    ```bash
    python Main.py
    ```

2.  **Select an Option:**  The script presents a menu with the following options:

    *   **1. Test-mode data collection (no DB write):** Collects live data but does not save it to the database.  Useful for testing the iRacing SDK connection and data fetching.
    *   **2. Live data collection (save to DB):** Collects live and historical data and saves it to the database. This requires the iRacing simulator to be running and in a race session.  Also collects all historical data for drivers that are not already in the database, using iRacing Data API.

        *   First it collects historical data based on the drivers participating in the active iRacing session.
        *   Then collects live data, updating RealTimeLapData and RealTimeEvents tables with real-time information.

    *   **3. Train & evaluate models:** Trains and evaluates the machine learning models (XGBoost, Random Forest, RankNet) using the data stored in the database. This option generates model comparison plots and SHAP analysis plots, displaying the results using matplotlib.  This step requires that you have run option 2 at least once to populate the database.
    *   **4. Exit:** Exits the application.

3.  **Data Collection:** When running in "Live data collection" mode (option 2), the script performs the following steps:

    *   Fetches driver names and CarIdx values from the iRacing SDK.
    *   Collects historical data for each driver using the iRacing Data API (`Data_fetching/FetchHistroical.py`).
    *   Processes the historical data, calculates performance indices, and inserts it into the `Driver` and `PastRaceTable` tables in the database (`Data_processing_and_inputting/historical_data_pipeline.py`).
    *   Collects live telemetry data from the iRacing SDK (`Data_fetching/FetchLive.py`).
    *   Processes the live data, calculates various features, and inserts it into the `RealTimeLapData` and `RealTimeEvents` tables in the database (`Data_processing_and_inputting/live_data_pipeline.py`).
    *   The live data collection runs continuously until a "Race ended successfully" message is detected or a keyboard interrupt (`Ctrl+C`) is issued.

4.  **Model Training:** The model training process (`model/XGboost/Train.py`, `model/RandomForest/Train.py`, `model/Ranknet/Train.py`) involves:

    *   Fetching race data from the database (`model/FetchDatabaseData.py`).
    *   Preprocessing the data, including normalization and handling missing values.
    *   Training XGBoost, Random Forest, and RankNet models.
    *   Evaluating the models using metrics like RMSE, MAE, Spearman correlation, and Top 3 accuracy.
    *   Generating and displaying model comparison plots and SHAP analysis plots.
    *   The code also uses learning curves to validate its performance.

## Code Structure

*   **`Main.py`:**  The main entry point for the application. Provides the user interface and orchestrates the data collection and model training processes.
*   **`Data_fetching/FetchHistroical.py`:**  Fetches historical driver and race data from the iRacing Data API.
*   **`Data_fetching/FetchLive.py`:**  Retrieves live telemetry data from the iRacing SDK.
*   **`Data_processing_and_inputting/historical_data_pipeline.py`:** Processes historical data, calculates performance indices, and inserts it into the database.
*   **`Data_processing_and_inputting/live_data_pipeline.py`:** Processes live telemetry data and inserts it into the database.
*   **`model/FetchDatabaseData.py`:**  Fetches race data from the database.
*   **`model/XGboost/Train.py`:**  Trains and evaluates the XGBoost model.
*   **`model/RandomForest/Train.py`:** Trains and evaluates the Random Forest model.
*   **`model/Ranknet/Train.py`:** Trains and evaluates the RankNet model.
*   **`utils/DatabaseConnection.py`:** Handles the connection to the SQL Server database and executes queries.
*   **`utils/IracingApiConnection.py`:** Manages the connection to the iRacing Data API.
*   **`utils/IracingSDKConnection.py`:** Manages the connection to the iRacing SDK.
*   **`utils/NormTarget.py`:**  Provides functions for normalizing target variables.
*   **`remap_ids_and_drop_new_column_with_subsession.py`:** Used to remap CustID and subsessionID, and to keep CustID as an integer instead of a string, for consistency.

## API Documentation

There is no dedicated API endpoint for this project since it uses desktop applications to function. However, the internal functions for collecting data and training the models could be adapted into APIs.

## Contributing Guidelines

1.  **Fork the Repository:** Fork the repository to your GitHub account.
2.  **Create a Branch:** Create a new branch for your feature or bug fix.
3.  **Make Changes:** Implement your changes, ensuring code quality and adherence to project standards.
4.  **Test Thoroughly:** Test your changes to ensure they work as expected and do not introduce new issues.
5.  **Submit a Pull Request:** Submit a pull request to the `main` branch of the original repository.
6.  **Code Review:**  Your pull request will be reviewed by project maintainers.  Address any feedback and make necessary changes.

## License Information

This project has no specified license. All rights are reserved by the authors.

## Contact/Support Information

*   **Primary Contact:** Antoine Spiteri
*   **GitHub:** [https://github.com/AntoineSpiteri2](https://github.com/AntoineSpiteri2)
*   For support or questions, please open an issue on the GitHub repository.