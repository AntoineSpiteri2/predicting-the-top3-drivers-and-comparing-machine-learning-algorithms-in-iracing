from Data_fetching.FetchLive import IRacingDataHandler
import pandas as pd
import asyncio
from utils.DatabaseConnection import insert_dataframe_to_db, execute_query
import time
    
import asyncio
import threading
import time

driver_missed = {}
# Declare _background_loop as a global variable
_background_loop = None
async def async_insert_dataframe_to_db(*args, **kwargs):
    return await asyncio.to_thread(insert_dataframe_to_db, *args, **kwargs)


def run_in_background(coro):
    """Fire-and-forget an async function by scheduling it on a background event loop."""
    global _background_loop
    if _background_loop is None:
        # Create an event loop in a background thread
        _background_loop = asyncio.new_event_loop()
        t = threading.Thread(target=_background_loop.run_forever, daemon=True)
        t.start()

    # Schedule the coroutine on that background loop
    asyncio.run_coroutine_threadsafe(coro, _background_loop)



def compute_live_order(df: pd.DataFrame, iracing) -> pd.DataFrame:

    if iracing.get_session_info()  != 5:

    
        # 1) filter out any garbage rows
        df = df[~df.TrackSurface.isin(["Unknown"])].copy()
        
        # 2) fill NaNs so our sort never ties
        df['lapdistpct'] = df['lapdistpct'].fillna(0.0)
        max_est = df['EstTime'].max(skipna=True) or 0
        df['EstTime'] = df['EstTime'].fillna(max_est + 1)
        
        # 3) global sort across ALL drivers
        df = df.sort_values(
            ['Lap', 'LapsBehind', 'lapdistpct', 'EstTime', 'CarIdx'],
            ascending=[False, True , False, True, True]
        ).reset_index(drop=True)
        
        # 4) assign 1…N
        df['LivePosition'] = df.index + 1
        return df

    else:
        df['LivePosition'] = df['Position']
        return df






async def updatelap(lap_data, record, df):
    # Calculate the max and min positions from the DataFrame.
    max_position = df['Position'].max()
    min_position = df['Position'].min()
    
    # Avoid division by zero by checking if the range is zero.
    if max_position == min_position:
        normalized_score = 0
    else:
        # Compute exactly as in SQL: (MaxPosition - record.Position) / (MaxPosition - MinPosition)
        normalized_score = (max_position - record.Position) / (max_position - min_position)
    

    # Multiply by the weight (which is 1) and round to 3 decimals.
    livescore_adjusted = round(normalized_score, 3)
    
    
    # Update lap_data with the computed LiveScoreAdjusted value and set incidents to 0.
    lap_data['LiveScoreAdjusted'] = livescore_adjusted
    
    # Insert the data into the RealTimeLapData table.
    await async_insert_dataframe_to_db(pd.DataFrame([lap_data]), "RealTimeLapData", False)
    
async def ProcessIracingLiveData(raceid, logs=False, driver_missed=None):
    iracing = IRacingDataHandler()  # Instantiate here
    # iracing.iraicngsdk.connect()
    previous_lap_dict = {}
    previous_incidents_dict = {}
    previous_position_dict = {}
    LapsBehind = {}
    inserted_laps = set()  # Maintain this outside the loop
    last_print_time = time.time()
    disqualified_drivers = {}
    previous_pit_status = {}
    driver_next_threshold = {} 
    Previous_LivPos = {}
    
    if driver_missed is None:
        driver_missed = {}
    # This will hold the normalization for driver gaps for the *current* lap snapshot
    last_live_positions: dict[str,int] = {}

    if logs:
        print("Starting live loop…")
    switch = False
    while True:
        try:
            try:
                live_data = iracing.get_positions()
            except TypeError as e:
                current_time = time.time()
                if current_time - last_print_time >= 10:
                    if logs:
                        print("no active race as still in qualifying or practice")
                    last_print_time = current_time
                continue
        
            flattened_data = [v for k, v in live_data.items()]
            df = pd.DataFrame(flattened_data)
            
            if (df['raceended'] == 1).all():
                if logs:
                    print("Race ended successfully.")
                break
            
            df = df.replace(-1, 0)
            if raceid != 0:
                df.insert(0, 'RaceID', raceid)

            current_time = time.time()
            
            if (df['disqaliyfiedFunction'] == False).mean() >= 0.75:
                if time.time() - last_print_time >= 60:  # Check if 120 seconds have passed
                    switch = True
                    last_print_time = time.time()  # Update the last_print_time
                else:
                    switch = False
            


            df = compute_live_order(df, iracing)

            if 'LivePosition' not in df.columns:
                df['LivePosition'] = (
                    df['CustId']
                    .map(last_live_positions)
                    .fillna(0)
                    .astype(int)
                )
            else:
                # persist the newly computed positions for next iteration
                last_live_positions.update(df.set_index('CustId')['LivePosition'].to_dict())

            for record in df.itertuples(index=False):
                
                if driver_missed is not None and record.CustId in driver_missed:
                    execute_query(f"""
                    UPDATE RealTimeLapData
                    SET CustID = {record.CustId}
                    WHERE CarIdx = {record.CarIdx} AND RaceID = {raceid};
                    """)
                
                if raceid != 0:
                    lap_data_columns = ['RaceID','CustId', 'CarIdx', 'Lap', 'LivePosition', 'DriverGapInFront', 'LapsBehind', 'LiveLapsLed', 'LiveLapsLedPercantage', 'isleaderorclose', 'PositionVolatility', 'OvertakePotential', 'ConsistencyScore', 'LiveScoreAdjusted', 'LiveLapsComplete', 'Position', 'InTop3Live', 'FastRepairsUsed', 'F2Time', 'BestLapTime', 'LastLapTime', 'PodiumProximity', 'LeadershipStrength', 'StableFrontRunner', 'AvgIncidentSeverity', 'ProgressConsistency']
                    event_data_columns = ['RaceID','CustId', 'Lap', 'TrackSurface', 'Speed', 'RPM', 'Gear', 'CarSteer', 'SteerIntensity' , 'SteerSpeedRatio' , 'lapdistpct' ,'LapPhase' , 'PitRoad', 'FastRepairsUsed', 'RpmPerGear', 'SpeedPerGear', 'AggressiveManeuver','CatchUpPotential','AvgOvertakeRate', 'EstTime']
                else:
                    lap_data_columns = ['CustId', 'CarIdx', 'Lap', 'LivePosition', 'DriverGapInFront', 'LapsBehind', 'LiveLapsLed', 'LiveLapsLedPercantage', 'isleaderorclose', 'PositionVolatility', 'OvertakePotential', 'ConsistencyScore', 'LiveScoreAdjusted', 'LiveLapsComplete', 'Position', 'InTop3Live', 'FastRepairsUsed', 'F2Time', 'BestLapTime', 'LastLapTime','PodiumProximity', 'LeadershipStrength', 'StableFrontRunner', 'AvgIncidentSeverity', 'ProgressConsistency']
                    event_data_columns = ['CustId', 'Lap', 'TrackSurface', 'Speed', 'RPM', 'Gear', 'CarSteer', 'SteerIntensity' , 'SteerSpeedRatio' , 'lapdistpct' ,'LapPhase' , 'PitRoad', 'FastRepairsUsed', 'RpmPerGear', 'SpeedPerGear', 'AggressiveManeuver','CatchUpPotential','AvgOvertakeRate', 'EstTime']
                
                cust_id = record.CustId

                
                lap_data = {col: getattr(record, col) for col in lap_data_columns}
                event_data = {col: getattr(record, col) for col in event_data_columns}
                
                
                record2 = df.loc[df.CustId == cust_id].iloc[0]
                lap_data['LivePosition'] = record2.LivePosition
                
                
                cust_id = record.CustId

                if cust_id not in previous_lap_dict:
                    previous_lap_dict[cust_id] = 0
                    previous_incidents_dict[cust_id] = 0
                    previous_position_dict[cust_id] = record.Position
                    LapsBehind[cust_id] = (record.LapsBehind, time.time())
                    previous_pit_status[cust_id] = None
                    driver_next_threshold[cust_id] = 0.01 # first 5% marker

                    Previous_LivPos[cust_id] = record.Position


                
          
                    
           
                
                lap_key = (raceid, record.CustId, record.Lap)

                if record.Lap != previous_lap_dict[cust_id] and record.Lap != 0 and record.TrackSurface not in ["Unknown", "In Pit Stall", "In Transition"] and switch == False:
                    if raceid != 0 and lap_key not in inserted_laps:
                        previous_lap_dict[cust_id]    = record.Lap
                        driver_next_threshold[cust_id] = 0.01   # reset to 5% at lap start

                        if cust_id not in driver_missed:
                            if logs:
                                print(f"inserting lap data for CarIdx {record.CarIdx}")

                            await  updatelap(lap_data, record, df)
                            inserted_laps.add(lap_key)
                            

                         
                         
                         
                                            

                # … later, your threshold logic …
                th = driver_next_threshold[cust_id]          # e.g. 0.05, 0.10, …, 0.95
                if (
                    record.lapdistpct >= th
                    and record.Lap != 0
                    and record.TrackSurface not in ["Unknown", "In Pit Stall", "In Transition"]
                    and not switch
                ):
                    pct = int(th * 100)                      # 5, 10, 15, …, 95
                    event_data['EventType'] = f"{pct}% Lap Complete"
                    # print(f"Lap {pct}% complete detected for CarIdx {record.CarIdx}")
                    await async_insert_dataframe_to_db(
                        pd.DataFrame([event_data]),
                        "RealTimeEvents",
                        False
                    )
                    # advance by 5%, wrap back into [0.0,1.0)
                    driver_next_threshold[cust_id] = (th + 0.01) % 1.0




                if lap_data['LivePosition']  != Previous_LivPos[cust_id]  and record.TrackSurface not in ["Unknown", "In Pit Stall", "In Transition"] and switch == False:
                    Previous_LivPos[cust_id] = lap_data['LivePosition']
                    if logs:
                        print(f"Live position change detected for CarIdx {record.CarIdx}")
                    if raceid != 0:
                        execute_query(f"""
                    UPDATE RealTimeLapData
                    SET LivePosition = {lap_data['LivePosition']}
                    WHERE RaceID = {raceid}
                    AND CustID = '{cust_id}'
                    AND CarIdx = {record.CarIdx}
                    AND Lap     = {record.Lap};
                    """)

           
                if previous_pit_status[cust_id] != "In Pit Stall" and (record.TrackSurface == "In Pit Stall" or record.TrackSurface == "In Transition")  and record.Lap != 0 and switch == False:
                    event_data['EventType'] = "in Pits"
                    if logs:
                        print(f"Pit Entry detected for CarIdx {record.CarIdx}")


                    execute_query(f"""
                    UPDATE RealTimeLapData
                    SET InPits = InPits + 1
                    WHERE RaceID = {raceid}
                    AND CustID = '{cust_id}'
                    AND Lap     = {record.Lap};
                """)

                    if raceid != 0 and cust_id not in driver_missed:
                        test = await async_insert_dataframe_to_db(pd.DataFrame([event_data]), "RealTimeEvents", False)

                        if test == 1:
                            event_data['Lap'] = record.Lap - 1 #fix database error 
                            await async_insert_dataframe_to_db(pd.DataFrame([event_data]), "RealTimeEvents", False)

                            
                    previous_pit_status[cust_id] = "In Pit Stall"

                if record.Incidents != previous_incidents_dict[cust_id] and record.TrackSurface != "Unknown" and record.TrackSurface != "In Pit Stall" and record.TrackSurface != "In Transition" and switch == False and record.Lap != 0:
                    previous_incidents_dict[cust_id] = record.Incidents
                    if record.incident_type == 'Off Track Incident (Spinout or Crash)' and record.RPM > 3500 or record.Speed > 150:
                        event_data['EventType'] = 'Off Track'
                        print(f"OffTrack increamented by 1 {record.CarIdx}")

                    else:
                        event_data['EventType'] = 'Incident'
                        
                        print(f"Incident increamented by 1 {record.CarIdx}")

                        
                    if logs:
                        print(f"Incident detected for CarIdx {record.CarIdx}")
                    if raceid != 0:
                        if cust_id not in driver_missed:
                            test = await async_insert_dataframe_to_db(pd.DataFrame([event_data]), "RealTimeEvents", False)


                        
                if record.Position != previous_position_dict[cust_id] and record.TrackSurface != "Unknown" and record.Lap != 0 and switch == False:
                    previous_position_dict[cust_id] = record.Position
                    event_data['EventType'] = 'Position Change'
                    if logs:
                        print(f"Position change detected for CarIdx {record.CarIdx}")
                    if raceid != 0:
                        if cust_id not in driver_missed:
                            test = await async_insert_dataframe_to_db(pd.DataFrame([event_data]), "RealTimeEvents", False)


                                    
                # redudant not anymore as it was used check for disqualification 
                
                
                # if record.TrackSurface == "Unknown" and record.disqaliyfiedFunction == True:
                #     current_time = time.time()
                #     if cust_id in disqualified_drivers and disqualified_drivers[cust_id] != float('inf'):
                #         if logs:
                #             print(record.CarIdx, " time left  ", time.time() - disqualified_drivers[cust_id])

                #     if cust_id not in disqualified_drivers:
                #         disqualified_drivers[cust_id] = time.time()  # Start the timer
                #         if logs:
                #             print(f"Started timer for CarIdx {record.CarIdx} at {current_time}")

                #     elif time.time() - disqualified_drivers[cust_id] >= 5:
                #         if logs:
                #             print(f"Disqualifying CarIdx {record.CarIdx} after 20 seconds.")
                #         execute_query(f"""
                #         UPDATE Driver
                #         SET 
                #             Disqualified = '{int(1)}'
                #         WHERE CustID = '{record.CustId}' AND RaceID = {raceid};
                #         """)
                #         disqualified_drivers[cust_id] = float('inf')  # Prevent further checks
                # elif record.disqaliyfiedFunction == True:
                #     if cust_id in disqualified_drivers:
                #         del disqualified_drivers[cust_id]
                #         if logs:
                #             print(f"removing disqualifican for CarIdx {record.CarIdx} ")
                #         timestamp = LapsBehind[cust_id]
                #         execute_query(f"""
                #         UPDATE Driver
                #         SET 
                #             Disqualified = '{int(0)}'
                #         WHERE CustID = '{record.CustId}' AND RaceID = {raceid};
                #         """)
                #         LapsBehind[cust_id] = (timestamp, time.time())


        except KeyboardInterrupt:
            if logs:
                print("KeyboardInterrupt detected. Exiting...")
            break
