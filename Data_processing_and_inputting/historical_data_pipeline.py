import os
import pandas as pd
from Data_fetching.FetchHistroical import Collect_Historical
import asyncio
from utils.DatabaseConnection import insert_dataframe_to_db, execute_query
from Data_fetching.FetchLive import IRacingDataHandler

def Get_existing_custid():
    query = "SELECT CustID FROM Driver"
    result =  execute_query(query)
    return result



def LoadHistoricalDataframeFromFile():
    # Load the DataFrame from the pickle file
    file_path = "historical_data.pkl"
    if os.path.exists(file_path):
        historical_dataframe = pd.read_pickle(file_path)
        print(f"Historical dataframe loaded from '{file_path}'.")
        return historical_dataframe
    else:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

async def SaveHistoricalDataframeToFile(historical_dataframe):
    # Save the DataFrame to a pickle file
    file_path = "historical_data.pkl"
    historical_dataframe.to_pickle(file_path)
    print(f"Historical dataframe saved to '{file_path}'.")

async def ProcessIracingHistoicalData(use_file=True):
    if use_file:
        
        try:
            # Try loading the saved DataFrame
            historical_dataframe = LoadHistoricalDataframeFromFile()
        except FileNotFoundError:
            print("Saved historical data not found, fetching from API...")
            # Fetch from API and save if the file doesn't exist
            historical_dataframe = await Collect_Historical(logs=True)
            await SaveHistoricalDataframeToFile(historical_dataframe)
    else:
        print("Fetching data directly from the API...")
        try:
            historical_dataframe = await Collect_Historical(logs=True)
            if use_file:
                await SaveHistoricalDataframeToFile(historical_dataframe)
        except KeyboardInterrupt:
            print("Data fetching interrupted. stopping the process")
            historical_dataframe = None
            

    IRacingDataHandler1 = IRacingDataHandler()
    weather = IRacingDataHandler1.Fetch_Weather_Data()
    
    weatherdf = pd.DataFrame([weather])
    if historical_dataframe is not None and not historical_dataframe.empty:
        historical_dataframe = pd.DataFrame(historical_dataframe)
        past_races = historical_dataframe[['Races']].copy()
        past_races_df = past_races.explode('Races')
        past_races_normalized = pd.json_normalize(past_races_df['Races'])

        historical_dataframe = historical_dataframe.drop("Races", axis=1)
        
        
        tuple_cols = ['Driver_avg_Irating','DriverAvgTTRating','DriverAvgLic_Sr']

        for c in tuple_cols:
            # extract the first element of the tuple
            historical_dataframe[c] = historical_dataframe[c].apply(lambda x: x[0] if isinstance(x, tuple) else x)  
            # convert to numeric
            historical_dataframe[c] = historical_dataframe[c].astype(float)
                
        Total_irating = (historical_dataframe['Driver_avg_Irating'].sum() * 100) * len(historical_dataframe)
        count_drivers = len(historical_dataframe)

        StrengthOfField = Total_irating / count_drivers
        weatherdf.insert(loc=weatherdf.columns.get_loc('TrackLength') + 1, column='StrengthOfField', value=StrengthOfField)
        
        exitingdrivers = Get_existing_custid()
        exitingdrivers_flat = [int(custid[0]) for custid in exitingdrivers]

        duplicate_rows = historical_dataframe[historical_dataframe['CustID'].isin(exitingdrivers_flat)]

        if not duplicate_rows.empty:
            print(f"Duplicate drivers detected:\n{duplicate_rows}")
            print("Stopping data collection due to duplicates.")
            return None
        

        raceid = int(insert_dataframe_to_db(weatherdf, "Race", True))
        


        historical_dataframe.insert(loc=historical_dataframe.columns.get_loc('CustID') + 1, column='RaceID', value=raceid)
        performance_indices = []
        for record in historical_dataframe.itertuples(index=False):
            norm_wins = IRacingDataHandler1.normalize(record.num_official_wins, historical_dataframe['num_official_wins'].min(), historical_dataframe['num_official_wins'].max())
            norm_winsalt = IRacingDataHandler1.normalize(record.wins, historical_dataframe['wins'].min(), historical_dataframe['wins'].max())

            norm_poles  = IRacingDataHandler1.normalize(record.poles, historical_dataframe['poles'].min(), historical_dataframe['poles'].max())
            norm_avg_incidents  = IRacingDataHandler1.normalize(record.avg_incidents, historical_dataframe['avg_incidents'].min(), historical_dataframe['avg_incidents'].max())
            norm_avg_finish_position = IRacingDataHandler1.normalize(record.avg_finish_position, historical_dataframe['avg_finish_position'].max(), historical_dataframe['avg_finish_position'].min())
            norm_laps_led_pct = IRacingDataHandler1.normalize(record.laps_led_percentage, historical_dataframe['laps_led_percentage'].min(), historical_dataframe['laps_led_percentage'].max())
            norm_consistency = IRacingDataHandler1.normalize(
            max(historical_dataframe['PerformanceConsistency']) - record.PerformanceConsistency,
            0,
            max(historical_dataframe['PerformanceConsistency']) - min(historical_dataframe['PerformanceConsistency'])
        )
            
            PerformanceIndex = (
            (0.6 * (norm_wins if norm_winsalt > norm_wins else norm_winsalt) +
            (0.5 * norm_poles) +
            (0.4 * norm_avg_finish_position) +
            (0.3 * norm_laps_led_pct) +
            (0.2 * norm_consistency)) -
            (0.3 * norm_avg_incidents) 

            )
            performance_indices.append(round(PerformanceIndex, 3))

            

        historical_dataframe['PerformanceIndex'] = performance_indices


        insert_dataframe_to_db(historical_dataframe, "Driver", False)

        insert_dataframe_to_db(past_races_normalized, "PastRaceTable", False)

        print("Data inserted into the database")
        return raceid
    
