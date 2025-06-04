import statistics
import time

import numpy as np
import pandas as pd
from utils.IracingSDKConnection import get_iracing_sdk_connection, IRacingSDK

class IRacingDataHandler:
    def  __init__(self):
        # ir = get_iracing_sdk_connection()
        self.previous_timestamp = time.time()
        self.incident_tracker = {}
        self.trackLength = self.get_track_length()
        self.disqaliyfiedFunction = True
        self.iraicngsdk = IRacingSDK()

        

    def get_session_info(self):
        
        return session_state
   
    
    
    def build_car_idx_mapping(self,results_positions):
        """
        Create a mapping from CarIdx to results data for quick lookups.
        """
        return {result['CarIdx']: result for result in results_positions if result['CarIdx'] != -1}


    def fetch_driver_names_and_caridx(self):
        """
        Fetches driver names and their corresponding CarIdx from both iRacing SDK and iRacing API.
        
        Returns:
            list: A list of dictionaries containing driver information with their CarIdx.
        """
        try:
            # Initialize SDK and API connections
            ir = get_iracing_sdk_connection()

            if not ir or not ir.is_connected:
                raise ConnectionError("Failed to connect to iRacing SDK.")

            driver_list = []

            # Fetch driver data from the SDK
            sdk_driver_info = ir['DriverInfo']['Drivers']
            for driver in sdk_driver_info:
                username = driver.get('UserName')
                car_idx = driver.get('CarIdx')
                custid = driver.get('UserID')

                # Filter out unwanted drivers like the Pace Car or specific exclusions
                if username not in ('Pace Car', 'Antoine Spiteri2'):
                    driver_list.append({
                        'UserName': username,
                        'CarIdx': car_idx,
                        'custid': custid
                    })


            return driver_list

        except KeyError as e:
            print(f"Error accessing driver information: {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return []

    def Fetch_Weather_Data(self):
        ir = get_iracing_sdk_connection()
        
        weather_keys = [
            "TrackTemp",
            "TrackTempCrew",
            "AirTemp",
            "RelativeHumidity",
            "WindVel",
            "WindDir",
            "Skies",
            "FogLevel",
            "Precipitation",
            "SolarAltitude",
            "SolarAzimuth",
            "WeatherDeclaredWet"
        ]

        weather_data = {}
        if ir.is_initialized and ir.is_connected:
            for key in weather_keys:
                try:
                    weather_data[key] = ir[key]
                except KeyError:
                    weather_data[key] = None  # Handle missing keys
                except Exception as e:
                    print(f"Error accessing {key}: {e}")
                    weather_data[key] = None
        else:
            print("IRSDK is not initialized or connected.")
            
        track_length = self.get_track_length()
        weather_data['TrackLength'] = track_length
        return weather_data
        
    def normalize(self,value, min_value, max_value):
        if max_value == min_value:
            return 0  # Prevent division by zero
        return (value - min_value) / (max_value - min_value)

    def get_track_length(self):
        
        
        """
        Retrieve track length from iRacing SDK.
        Returns track length in meters.
        """
        ir = get_iracing_sdk_connection()

        if ir.is_initialized:
            # Fetch track length from SessionInfoStr
            session_string = ir['WeekendInfo']['TrackLength']
            if session_string:
                session_text = session_string.decode('utf-8') if isinstance(session_string, bytes) else session_string
                for line in session_text.splitlines():
                    track_length_str = line.split(" ", 1)[1].strip()
                    if "km" in track_length_str.lower():#
                        return float(session_text.split(" ", 1)[0].strip()) * 1000
                    elif "mi" in track_length_str.lower():
                        return float(session_text.split(" ", 1)[0].strip()) * 1000
        return 5000  # Default track length in meters if unavailable
            
    def calculate_speed(self,lap_dist_pct, prev_lap_dist_pct, track_length, time_diff):
        """
        Calculate speed in km/h using lap distance percentage and time difference not super accurate but its better then nothing.
        """
        if time_diff > 0:
            # Distance traveled in meters
            distance_traveled = (lap_dist_pct - prev_lap_dist_pct) * track_length
            # Speed in m/s
            speed_m_per_s = distance_traveled / time_diff
            # Convert to km/h
            return max(0, speed_m_per_s * 3.6)  # Ensure no negative speed
        return 0

    def get_positions(self):
        global incident_tracker, previous_timestamp, session_state
        irsdk_instance = None

        while irsdk_instance is None:
            irsdk_instance = get_iracing_sdk_connection()
            if irsdk_instance is None:
                time.sleep(5)  # Wait for 1 second before retrying

        # Check if iRacing is running
        if not irsdk_instance.is_initialized and irsdk_instance.startup():
            print("iRacing is connected.")

        if irsdk_instance.is_initialized:
            # Fetch telemetry data
            car_idx_lap = irsdk_instance['CarIdxLap']
            car_idx_position = irsdk_instance['CarIdxPosition']
            track_surface = irsdk_instance['CarIdxTrackSurface']
            car_idx_lap_dist_pct = irsdk_instance['CarIdxLapDistPct']
            car_idx_rpm = irsdk_instance['CarIdxRPM']
            car_idx_gear = irsdk_instance['CarIdxGear']
            car_idx_best_lap_time = irsdk_instance['CarIdxBestLapTime']
            car_idx_last_lap_time = irsdk_instance['CarIdxLastLapTime']
            car_idx_est_time = irsdk_instance['CarIdxEstTime']
            driver_info = irsdk_instance['DriverInfo']['Drivers']
            session_state = irsdk_instance['SessionState']  # Get the current session state
            SessionTimeRemain = irsdk_instance['SessionTimeRemain']  # Get the current session state

            caridxsteer = irsdk_instance['CarIdxSteer']
            CarIdxOnPitRoad = irsdk_instance['CarIdxOnPitRoad']
            CarIdxFastRepairsUsed = irsdk_instance['CarIdxFastRepairsUsed']
            CarIdxF2Time = irsdk_instance['CarIdxF2Time']
            raceended = 0
            # Check if the race is in an active state
            if session_state == 6 and SessionTimeRemain >= 0:  # 4: Warmup, 5: Race
                raceended = 1

            if session_state == 5:
                self.disqaliyfiedFunction = False


                 
            session_info = irsdk_instance['SessionInfo']
            results_positions = session_info['Sessions'][2]['ResultsPositions']  # Get 2 for races or even 1 not 0 as thats pratice
            car_idx_to_results = self.build_car_idx_mapping(results_positions)
            # Retrieve track length dynamically

            # Current timestamp for speed calculation
            current_timestamp = time.time()
            time_diff = current_timestamp - self.previous_timestamp
            self.previous_timestamp = current_timestamp

            # Map data for all drivers
            current_data = {}
            if (
                car_idx_lap is not None
                and car_idx_position is not None
                and track_surface is not None
                and car_idx_lap_dist_pct is not None
            ):
                for driver in driver_info:
                    car_idx = driver['CarIdx']
                    if car_idx != -1 or car_idx != 0 or driver['UserID'] != -1  or driver['UserName'] != 'Pace Car':  # Valid car index
                        user_name = driver['UserName']
                        Car_idx = driver['CarIdx']
                        lap = car_idx_lap[car_idx]
                        driverid = driver['UserID']
                        position = car_idx_position[car_idx]
                        surface_status = track_surface[car_idx]
                        lap_dist_pct =  round(car_idx_lap_dist_pct[car_idx], 5) # Round to 5 decimal places as as is it has allot of decimal places
                        rpm = car_idx_rpm[car_idx]
                        gear = car_idx_gear[car_idx]
                        best_lap_time = car_idx_best_lap_time[car_idx]
                        last_lap_time = car_idx_last_lap_time[car_idx]
                        est_time = car_idx_est_time[car_idx]
                        carsteer = abs(caridxsteer[car_idx])
                        PitRoad = CarIdxOnPitRoad[car_idx]
                        FastRepairsUsed = CarIdxFastRepairsUsed[car_idx]
                        F2Time = CarIdxF2Time[car_idx]

                        # Define track surface statuses
                        if surface_status == 0:
                            track_status = "Off Track"
                        elif surface_status == 1:
                            track_status = "In Pit Stall"
                        elif surface_status == 2:
                            track_status = "In Transition"
                        elif surface_status == 3:
                            track_status = "On Track"
                        else:
                            track_status = "Unknown"

                        # Skip drivers with "Unknown" track status
                        if  driver['UserName'] == 'Pace Car' or driver['UserName'] == 'Antoine Spiteri2':
                            continue
                        # Initialize or update incident tracker
                        if car_idx not in self.incident_tracker:
                            self.incident_tracker[car_idx] = {
                                "last_lap_dist_pct": lap_dist_pct,
                                "last_position": position,
                                "last_rpm": rpm,
                                "incidents": 0,
                                "incident_type": "None",
                                "last_low_rpm_time": 0,  # Timestamp for low RPM cooldown
                                "last_speed": 0,
                                "last_gear": gear,
                                "last_off_track_time": 0,  # Timestamp for off-track cooldown
                                "race_start_time": time.time(),  # Timestamp for race start
                                "last_lap_time": last_lap_time,
                                "lap_times": [],
                                "lap_incidents": 0,  # Track incidents per lap
                                "current_lap": lap  # Track current lap
                            }

                        # Calculate speed
                        prev_lap_dist_pct = self.incident_tracker[car_idx]["last_lap_dist_pct"]
                        speed = self.calculate_speed(lap_dist_pct, prev_lap_dist_pct, self.trackLength, time_diff)
                        
                        # Incident detection logic
                        incidents = self.incident_tracker[car_idx]["incidents"]
                        incident_type = "None"
                        if track_status not in ["In Pit Stall", "In Transition", "Unknown"] and session_state != 5:
                            # Off-track and Low Speed/RPM detection logic
                            current_time = time.time()
                            race_start_delay = 10  # Delay in seconds after race start to prevent false positives
                            if current_time - self.incident_tracker[car_idx]["race_start_time"] > race_start_delay:
                                # Ensure the incident triggers if the condition persists for 3 seconds
                                if track_status == "Off Track":
                                    if current_time - self.incident_tracker[car_idx].get("last_off_track_time", 0) > 1:  # 3-second persistence
                                        incident_type = "Off Track Incident (Spinout or Crash)"
                                        incidents += 1
                                        self.incident_tracker[car_idx]["last_off_track_time"] = current_time + 2  # Add 5-second cooldown
                                    else:
                                        self.incident_tracker[car_idx]["last_off_track_time"] = current_time
                                elif speed <= 150 and rpm < 3500:
                                    if current_time - self.incident_tracker[car_idx].get("last_low_rpm_time", 0) > 1:  # 3-second persistence
                                        incident_type = "Low Speed/RPM (Stall or Spinout or Crash)"
                                        incidents += 1
                                        self.incident_tracker[car_idx]["last_low_rpm_time"] = current_time + 2  # Add 5-second cooldown
                                    else:
                                        self.incident_tracker[car_idx]["last_low_rpm_time"] = current_time

                        # Carry over incidents to the current lap
                        self.incident_tracker[car_idx]["lap_incidents"] = incidents


                        # Update incident tracker
                        self.incident_tracker[car_idx]["last_lap_dist_pct"] = lap_dist_pct
                        self.incident_tracker[car_idx]["last_position"] = position
                        self.incident_tracker[car_idx]["last_rpm"] = rpm
                        self.incident_tracker[car_idx]["incident_type"] = incident_type
                        self.incident_tracker[car_idx]["incidents"] = incidents
                        self.incident_tracker[car_idx]["last_speed"] = speed
                        self.incident_tracker[car_idx]["last_gear"] = gear
                        session_data = car_idx_to_results.get(car_idx, {})
                        intop3 = False
                        if position <= 3:
                            intop3 = True
                            
                        steer_intensity = carsteer / 3  # Normalize to range [0, 1]
                        steer_speed_ratio = steer_intensity * speed
                        rpm_per_gear = (rpm / gear) if gear > 0 else 0
                        speed_per_gear = (speed / gear) if gear > 0 else 0
                        aggressive_maneuver = (steer_intensity >= 1 or steer_intensity <= -1 and speed > 150)  # Example: steer > 33% of max
                        progress_consistency = 0
                        # Update lap times in the tracker
                        if last_lap_time > 0:  # Ensure valid lap time
                            self.incident_tracker[car_idx]["lap_times"].append(last_lap_time)
                            # Keep only the last 5 lap times for consistency calculation
                            if len(self.incident_tracker[car_idx]["lap_times"]) > 5:
                                self.incident_tracker[car_idx]["lap_times"].pop(0)
                                
                        lap_times = self.incident_tracker[car_idx]["lap_times"]
                        if len(lap_times) > 1:
                            lap_time_variance = statistics.stdev(lap_times)  # Use standard deviation for simplicity
                            progress_consistency = 1 / (1 + lap_time_variance)  # Inverse of variance for consistency
                        else:
                            progress_consistency = 0  # Not enough data to calculate consistency
                        
                        completed_laps = session_data.get('LapsComplete', 0) if len(session_data) != 0 else 0
                        gap_in_front = session_data.get('Time', 0) if len(session_data) != 0 else 0

                        # PodiumProximity
                        if position <= 3:
                            podium_proximity = 1.0
                        elif gap_in_front > 0:
                            podium_proximity = 1 / (gap_in_front + 1)
                        else:
                            podium_proximity = 0
                            
                        leader_avg_lap_time = min(
                            [car_idx_best_lap_time[car_idx] for car_idx in car_idx_position if car_idx_position[car_idx] == 1],
                            default=0
                        )

                        if session_data.get('Lap', 0) >= 1 and session_data['Time'] == 0:
                            adjusted_time = session_data.get('Lap', 0) * leader_avg_lap_time
                        elif len(session_data) > 0:
                            adjusted_time = session_data['Time']
                        else:
                            adjusted_time = 0
                            
                            
                            
                        LiveLapsLedPercantage = 0
                        ConsistencyScore = 0
                        OvertakePotential = 0
                        if len(session_data) != 0:
                            LiveLapsLedPercantage = (session_data.get('LapsLed', 0) / session_data.get('LapsComplete', 0)) if session_data.get('LapsComplete', 0) > 0 else 0
                            OvertakePotential = (adjusted_time / last_lap_time) if last_lap_time > 0 else 0
                            ConsistencyScore = (best_lap_time / last_lap_time) if last_lap_time > 0 else 0
                        # LeadershipStrength
                        if position == 1:
                            leadership_strength = LiveLapsLedPercantage * ConsistencyScore
                        else:
                            leadership_strength = max(0, adjusted_time / last_lap_time) * ConsistencyScore


                        catch_up_potential = (
                            (lap_dist_pct ** 2 * OvertakePotential) / (1 + gap_in_front)
                            if gap_in_front > 0 and position > 3 else 0
                        )

                        AvgOvertakeRate = (
                            lap_dist_pct / (gap_in_front + 1) if gap_in_front > 0 else 0
                        )

                        AvgIncidentSeverity = (
                            incidents / completed_laps if completed_laps > 0 else 0
                        )

                        


                        lap_phase = (
                            "Start" if lap_dist_pct <= 0.25 else
                            "Mid" if lap_dist_pct <= 0.50 else
                            "Near end" if lap_dist_pct <= 0.75 else
                            "end" 
                        )

                        # StableFrontRunner
                        if position <= 3:
                            incident_factor = (1 - (incidents / (incidents + 1)))  # Decreases with more incidents
                            stable_front_runner = ConsistencyScore * incident_factor * (LiveLapsLedPercantage + 1)
                        else:
                            stable_front_runner = 0
            

                       
                                
            
                        # Prepare current data
                        current_data[car_idx] = {
                            "CustId": driver['UserID'],
                            "CarIdx": Car_idx,
                            "Lap": lap,
                            "lapdistpct": lap_dist_pct,
                            "LapPhase": lap_phase,  # New feature
                            "DriverGapInFront": gap_in_front,
                            "LapsBehind": session_data.get('Lap', 0) if len(session_data) != 0 else 0,
                            "LiveLapsLed": session_data.get('LapsLed', 0) if len(session_data) != 0 else 0,
                            "LiveLapsLedPercantage": LiveLapsLedPercantage,
                            "isleaderorclose": gap_in_front <= 1,
                            "PositionVolatility": None,
                            "OvertakePotential": OvertakePotential,
                            "ConsistencyScore": ConsistencyScore,
                            "LiveScoreAdjusted":  None,
                            "LiveLapsComplete": completed_laps,
                            "Position": position,
                            "InTop3Live": intop3,
                            "TrackSurface": track_status,
                            "Speed": round(speed, 2),
                            "RPM": rpm,
                            "Gear": gear,
                            "CarSteer": carsteer,
                            "PitRoad": PitRoad,
                            "FastRepairsUsed": FastRepairsUsed,
                            "F2Time": F2Time,
                            "BestLapTime": best_lap_time,
                            "LastLapTime": last_lap_time,
                            "EstTime": est_time,
                            "Incidents": incidents,
                            "SteerIntensity": steer_intensity,
                            "SteerSpeedRatio": steer_speed_ratio,
                            "RpmPerGear": rpm_per_gear,
                            "SpeedPerGear": speed_per_gear,
                            "AggressiveManeuver": aggressive_maneuver,
                            "PodiumProximity": podium_proximity,
                            "LeadershipStrength": leadership_strength,
                            "CatchUpPotential": catch_up_potential,
                            "StableFrontRunner": stable_front_runner,
                            "AvgOvertakeRate": AvgOvertakeRate,
                            "AvgIncidentSeverity": AvgIncidentSeverity,
                            "ProgressConsistency": progress_consistency,  # Adjusted feature
                            "disqaliyfiedFunction": self.disqaliyfiedFunction,
                            "raceended": raceended,
                            "incident_type": incident_type
                        }

            # Sort data by race position (ascending) and update the table
            sorted_data = {k: v for k, v in sorted(current_data.items(), key=lambda item: item[1]['Position'])}
            
            df = pd.DataFrame(sorted_data)

                    
            return df

    
    
    
