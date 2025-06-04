import asyncio
import statistics
from utils.IracingApiConnection import return_iracing_api_client, AsyncIDClientWrapper
from Data_fetching.FetchLive import IRacingDataHandler
import numpy as np
import pandas as pd
idc = return_iracing_api_client()

async_idc =  AsyncIDClientWrapper(idc, max_concurrent_requests=100) 



async def calculate_average_for_key(driver_data, key):
    values = [entry[key] for entry in driver_data if key in entry and entry[key] != 0]
    if len(values) == 0:
        return 0
    return sum(values) / len(values)

async def TwoNumAvg(num1, num2):
    values = [num if num is not None else 0 for num in [num1, num2]]
    total = sum(values)
    avg = total / 2
    return round(avg, 3)


async def transform_driver_data(driver_data, value_multiplier=0.001):
    if 'data' in driver_data and driver_data['data']:
        current_value = driver_data['data'][-1]['value'] * value_multiplier
        return round(current_value, 2)
    else:
        print(f"Warning: 'data' is missing or empty in driver_data: {driver_data}")
        return 0
    
async def format_lap_time_as_minutes(number):
    number_str = str(number).zfill(7)  # Ensure the number has at least 7 digits
    milliseconds = int(number_str[-3:])
    seconds = int(number_str[-5:-3])
    minutes = int(number_str[:-5]) if len(number_str) > 5 else 0
    total_seconds = (minutes * 60) + seconds + (milliseconds / 1000)
    return total_seconds


async def transform_race_data(race_result, main_driver_name, subessionid, CustID):
    session_data = None

    for session in race_result.get('session_results', []) or race_result.get('session_results', []):
        simnm = session.get("simsession_number", [])
        if simnm == 0:
            # Filter results for the main driver
            session_results = [
                {
                    "finish_position": result.get("finish_position"),
                    "laps_lead": result.get("laps_lead"),
                    "average_lap": await format_lap_time_as_minutes(result.get("average_lap")),
                    "best_lap_time":  await format_lap_time_as_minutes(result.get("best_lap_time")),
                    "incidents": result.get("incidents"),
                    "starting_position": result.get("starting_position"),
                    "in_top_3": result.get("finish_position") <= 3
                }
                for result in session.get("results", [])
                if result.get("display_name") == main_driver_name or  result.get("cust_id") == CustID
            ]

            if session_results:

                # Flatten session-level data and driver results into a single row
                session_data = {
                    "session_type": session.get("simsession_type_name", ""),
                    "avg_temp": session.get("weather_result", {}).get("avg_temp", 0),
                    "avg_wind_speed": session.get("weather_result", {}).get("avg_wind_speed", 0),
                    "precip_mm": session.get("weather_result", {}).get("precip_mm", 0),
                    "avg_rel_humidity": session.get("weather_result", {}).get("avg_rel_humidity", 0),
                    "finish_position": session_results[0]["finish_position"],
                    "laps_lead": session_results[0]["laps_lead"],
                    "average_lap": session_results[0]["average_lap"],
                    "best_lap_time": session_results[0]["best_lap_time"],
                    "incidents": session_results[0]["incidents"],
                    "starting_position": session_results[0]["starting_position"],
                    "in_top_3": session_results[0]["in_top_3"]
                }
                break  # Since we need only one row, stop processing further sessions

    # If no session data is found, return empty dictionary
    if not session_data:
        return {}

    # Add race-level attributes to the single row
    race_data = {
        "subessionid": subessionid,
        "CustID": str(CustID),
        "corners_per_lap": race_result.get("corners_per_lap", 0),
        "event_strength_of_field": race_result.get("event_strength_of_field", 0),
        "num_caution_laps": race_result.get("num_caution_laps", 0),
        "num_cautions": race_result.get("num_cautions", 0),
        "avg_temp": session_data["avg_temp"],
        "avg_wind_speed": session_data["avg_wind_speed"],
        "precip_mm": session_data["precip_mm"],
        "avg_rel_humidity": session_data["avg_rel_humidity"],
        "finish_position": session_data["finish_position"],
        "laps_lead": session_data["laps_lead"],
        "average_lap": session_data["average_lap"],
        "best_lap_time": session_data["best_lap_time"],
        "incidents": session_data["incidents"],
        "starting_position": session_data["starting_position"],
        "in_top_3": session_data["in_top_3"],
        "Improvement":  abs((session_data["best_lap_time"] - session_data["average_lap"])),
        "BadWeatherIndex": (session_data["avg_temp"] + session_data["avg_wind_speed"] + session_data["precip_mm"]),
        "CornerEfficiency": (session_data["average_lap"] / race_result.get("corners_per_lap", 0)),
        "PitStopEfficiency": (session_data["best_lap_time"] / session_data["average_lap"] if session_data["average_lap"] != 0 else 0),
    }

    return pd.DataFrame([race_data])  # Return as a single-row DataFrame


async def test():
    # Properly await the asynchronous method
    test_result = await async_idc.stats_member_summary(cust_id="1011518")
    print(test_result)
    
    
async def Collect_Historical(custid=None,caridx=None, logs=False):
    driver_info=custid
    if custid is None:
        # Fetch all drivers when no custid is passed
        iracing_handler = IRacingDataHandler()
        driver_info = iracing_handler.fetch_driver_names_and_caridx()
    else:
        # Use the passed custid(s)
        driver_info = [{'custid': custid, 'UserName': f"Driver_{custid}", 'CarIdx': caridx}]

    all_data = []

    
    for driver in driver_info:
        if logs:
            print(driver['UserName'])
        drivers_info =  await  async_idc.lookup_drivers(search_term=driver['UserName'])
        
        if len(drivers_info) == 0:
           drivers_info.append(driver['custid']) # this is just in case the driver is not fouund in the iracing api via name so we use the custid directly as a plan B
           
        if drivers_info:
            cust_id = None
            try:
                cust_id = drivers_info[0]['cust_id']
            except TypeError:
                cust_id = drivers_info[0]

            driver_summary = await async_idc.stats_member_summary(cust_id=cust_id)
            num_official_sessions = driver_summary["this_year"]["num_official_sessions"]
            num_league_sessions = driver_summary["this_year"]["num_league_sessions"]
            num_official_wins = driver_summary["this_year"]["num_official_wins"]
            num_league_wins = driver_summary["this_year"]["num_league_wins"]
            career_stats = await async_idc.stats_member_career(cust_id=cust_id)
            result = [item for item in career_stats["stats"] if item["category_id"] in {5,1,6,2}]
            
            starts = await calculate_average_for_key(result, "starts")
            wins = await calculate_average_for_key(result, "wins")
            top5 = await calculate_average_for_key(result, "top5")
            poles = await calculate_average_for_key(result, "poles")
            avg_start_position =  await calculate_average_for_key(result, "avg_start_position")
            avg_finish_position = await calculate_average_for_key(result, "avg_start_position")
            laps = await calculate_average_for_key(result, "laps")
            laps_led = await calculate_average_for_key(result, "laps_led")
            avg_incidents = await calculate_average_for_key(result, "avg_incidents")
            avg_points = await calculate_average_for_key(result, "avg_points")
            win_percentage = await calculate_average_for_key(result, "win_percentage")
            top5_percentage = await calculate_average_for_key(result, "top5_percentage")
            laps_led_percentage = await calculate_average_for_key(result, "laps_led_percentage")
            total_club_points = await calculate_average_for_key(result, "total_club_points")
            poles_percentage = await calculate_average_for_key(result, "poles_percentage")
            recent_races = await async_idc.stats_member_recent_races(cust_id=cust_id)
            DriverAvgIrating = await transform_driver_data(await async_idc.member_chart_data(cust_id=cust_id, category_id=5, chart_type=1)),

            DriverAvgTTRating = await transform_driver_data(await async_idc.member_chart_data(cust_id=cust_id, category_id=5, chart_type=2)),
                
            
            DriverAvgLicSr = await transform_driver_data(await async_idc.member_chart_data(cust_id=cust_id, category_id=5, chart_type=3)),

            races = []
            for race in recent_races['races']:
                race_result = await async_idc.result(subsession_id=race['subsession_id'])
                transformed_data = await transform_race_data(race_result, driver['UserName'], race['subsession_id'], cust_id)
                if len(transformed_data) != 0:
                    races.append(transformed_data.to_dict(orient="records")[0])  # Extract the single dictionary
                else:
                    if logs:
                        print("Warning: No race on ",  race_result)
                    continue
            # Calculate metrics for each driver
            # Calculate metrics for each driver
            finish_positions = [race['finish_position'] for race in races if 'finish_position' in race]
            avg_finish_positions = sum(finish_positions) / len(finish_positions) if finish_positions else None

            starting_positions = [race['starting_position'] for race in races if 'starting_position' in race]
            RecoveryPerformance = (
                sum(finish - start for finish, start in zip(finish_positions, starting_positions)) / len(races)
                if len(finish_positions) == len(starting_positions) and len(races) else 0
            )

            incidentRace = [race['incidents'] for race in races if 'incidents' in race]
            RecentIncidentsPerRace = sum(incidentRace) / len(races) if len(races) else 0

            recentpodrace = [race['finish_position'] for race in races if 'finish_position' in race and race['finish_position'] <= 3]
            RecentPodiumRate = len(recentpodrace) / len(races) if len(races) else 0

            wins = [race['finish_position'] for race in races if 'finish_position' in race and race['finish_position'] == 1]
            RecentWinRate = len(wins) / len(races) if len(races) else 0
            wins = len(wins)
            qualifying_races = sum(1 for race in races if 'starting_position' in race and race['starting_position'] >= 5 and race['finish_position'] <= 3)
            ClutchPerformanceRate = qualifying_races / len(races) if len(races) else 0

            FinishPosSTDEV = statistics.stdev(finish_positions) if len(finish_positions) > 1 else 0

            rainy_races = [race for race in races if 'precip_mm' in race and race['precip_mm'] >= 1]
            AvgFinishInRain = (
                sum(race['finish_position'] for race in rainy_races) / len(rainy_races)
                if rainy_races else 0
            )




            badweatherindex = [race['BadWeatherIndex'] for race in races if 'BadWeatherIndex' in race]
            StressHandling = sum(badweatherindex) / len(badweatherindex) if badweatherindex else 0

            all_data.append({
                        "CustID": cust_id,
                        "CarIdx": driver['CarIdx'],
                        "num_official_sessions": num_official_sessions,
                        "num_league_sessions": num_league_sessions,
                        "num_official_wins": num_official_wins,
                        "num_league_wins": num_league_wins,
                        "starts": starts,
                        "wins": wins,
                        "top5": top5,
                        "poles": poles,
                        "avg_start_position": avg_start_position,
                        "avg_finish_position": avg_finish_position,
                        "laps": laps,
                        "laps_led": laps_led,
                        "avg_incidents": avg_incidents,
                        "avg_points": avg_points,
                        "win_percentage": win_percentage,
                        "top5_percentage": top5_percentage,
                        "laps_led_percentage": laps_led_percentage,
                        "total_club_points": total_club_points,
                        "poles_percentage": poles_percentage,
                        "Driver_avg_Irating": DriverAvgIrating,
                        "DriverAvgTTRating": DriverAvgTTRating,
                        "DriverAvgLic_Sr": DriverAvgLicSr,
                        "RecentForm": avg_finish_positions,
                        "RecoveryPerformance": RecoveryPerformance,
                        "RecentIncidentsPerRace": RecentIncidentsPerRace,
                        "RecentPodiumrate": RecentPodiumRate,
                        "RecentWinRate": RecentWinRate,
                        "ClutchPerformanceRate": ClutchPerformanceRate,
                        "PerformanceConsistency": FinishPosSTDEV,
                        "StressHandling": StressHandling,
                        "AvgFinishInRain": AvgFinishInRain,
                        "PerformanceIndex": None,
                        "RankScoreAdjusted": 0,
                        # "Disqualified": False,
                        "Races": races
                    })
            
    df = pd.DataFrame(all_data)



    return df



