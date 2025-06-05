"""
dynamic_race_processing.py
───────────────────────────────────────────────────────────────────────────────
End-to-end feature-engineering pipeline for the SQL-Server schema in
racing_data / racing_data_Live.

• All columns from *Race* and *Driver* tables are copied verbatim
  (prefixed Race_… and Drv_…).

• ALL numeric columns from RealTimeLapData, RealTimeEvents and
  PastRaceTable are summarised with a dynamic *bucket + decay* strategy:
      – buckets are centred on equally-spaced fractions of race distance
      – within each bucket we take the mean of each numeric column
      – that mean is multiplied by the bucket’s DECAY weight

• Event counts (Incidents / in Pits / Off Track) are aggregated the same
  way; they already respect the 1 % Lap-progress trigger that the live
  collector writes.

Add or remove columns in any table and they will appear / disappear
automatically in the output feature-set – no code changes needed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import re

from utils.DatabaseConnection import execute_query


# ──────────────────────────────── configuration ─────────────────────────────

# bucket anchor-points (fraction of total race distance)
BUCKETS: Dict[str, float] = {
    "First": 0.00,   # 0 %
    "25":    0.25,   # 25 %
    "50":    0.50,   # 50 %
    "75":    0.75,   # 75 %
    # "Last":  1.00,   # 100 %
}

# how many consecutive laps make up each bucket window
WINDOW = 3           # 1-2-3, 3-4-5, … (feel free to shrink / enlarge)

WINDOW_FRAC = 0.06
MIN_WINDOW  = 3  
# decay weight per bucket (used to down-weight early laps)
DECAY: Dict[str, float] = {
    "First": 1 / 6,
    "25":    2 / 6,
    "50":    3 / 6,
    "75":    4 / 6,
    # "Last":  6 / 6,
}

def window_size(total_laps: int) -> int:
    """Return an integer window length proportional to race distance."""
    return max(MIN_WINDOW, int(round(total_laps * WINDOW_FRAC)))


# ───────────────────────────── DB helpers (thin wrappers) ────────────────────


def _fetch_dataframe(query: str) -> pd.DataFrame:
    data = execute_query(query)
    return pd.DataFrame(data or [])


def fetch_races() -> pd.DataFrame:
    return _fetch_dataframe("SELECT * FROM Race;")


def fetch_drivers(race_id: int, livepred = False) -> pd.DataFrame:
    
    if livepred == True:
        return _fetch_dataframe(f"SELECT * FROM Driver WHERE RaceID = '{race_id}';")
    else:
        return _fetch_dataframe(
            f"""
        SELECT DISTINCT
                d.*
            FROM Driver AS d
            INNER JOIN RealTimeLapData AS l
                ON l.RaceID = d.RaceID
            AND l.CustID = d.CustID
            WHERE d.RaceID = {race_id}
            ORDER BY d.RankScoreAdjusted DESC;
            """
        )


def fetch_lap_data(cust_id: str) -> pd.DataFrame:
    return _fetch_dataframe(
        f"SELECT * FROM RealTimeLapData WHERE CustId = '{cust_id}';"
    )


def fetch_events_for_laps(cust_id: str, laps: np.ndarray) -> pd.DataFrame:
    # guard against empty lap lists
    if len(laps) == 0:
        return pd.DataFrame()
    lap_list = ",".join(map(str, laps))
    return _fetch_dataframe(
        f"""
        SELECT *
        FROM   RealTimeEvents
        WHERE  CustId = '{cust_id}'
          AND  Lap    IN ({lap_list})
          AND  EventType != 'Position Change'
        """
    )

        #   AND  EventType != 'Position Change';


def fetch_past_races(cust_id: str) -> pd.DataFrame:
    return _fetch_dataframe(
        f"SELECT * FROM PastRaceTable WHERE CustID = '{cust_id}';"
    )


# ───────────────────────────── feature helpers ──────────────────────────────

COUNT_PATTERNS = re.compile(
    r"(?:^Cnt_|_count$|_last3$|incident|offtrack|inpits|pit_stop)",
    flags=re.IGNORECASE,
)

def zero_fill_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace NaNs with 0 in any column whose name matches COUNT_PATTERNS.
    """
    count_cols = [c for c in df.columns if COUNT_PATTERNS.search(c)]
    if count_cols:
        df[count_cols] = df[count_cols].fillna(0)
    return df


def fill_other_missing(df: pd.DataFrame) -> pd.DataFrame:
    # — numeric columns: fill with 0 (or you could use median if 0 is not sensible)
    num = df.select_dtypes(include="number").columns
    df[num] = df[num].fillna(0)

    # — categorical/text columns: fill with a sentinel
    cat = df.select_dtypes(include="object").columns
    df[cat] = df[cat].fillna("Missing")

    return df

def bucket_window(
    lap_df: pd.DataFrame, frac: float, total_laps: int, window: int = WINDOW
) -> pd.DataFrame:
    """
    Return the *window*-sized slice of *lap_df* starting at
    floor(frac · max_lap). Example: window=3, frac=0.25, max_lap=20 → laps 5-6-7.
    """
    anchor = max(1, int(np.floor(frac * total_laps)))
    return lap_df.query(f"Lap >= {anchor} and Lap < {anchor + window}")


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric columns of *df*."""
    return df.select_dtypes(include="number")





def add_recency_features(
    lap_df: pd.DataFrame,
    cust_id: str,
    recency: int = 5
) -> Dict[str, float]:
    """
    Count incident, off-track, in-pits and lap-complete events in the last N laps.

    Parameters
    ----------
    lap_df : DataFrame
        RealTimeLapData for one driver, must have column "Lap".
    cust_id : str
        The driver ID, used to fetch RealTimeEvents.
    recency : int
        Number of most recent laps to look back over.

    Returns
    -------
    Dict[str,float]
        {
          'Cnt_incident_last5': ...,
          'Cnt_inpits_last5':   ...,
          'Cnt_offtrack_last5': ...,
          'Pct_lapcomplete_last5': ...
        }
    """
    # 1) figure out which laps to inspect
    max_lap = int(lap_df['Lap'].max())
    laps = list(range(max(1, max_lap - recency + 1), max_lap + 1))

    # 2) fetch only those events
    evt = fetch_events_for_laps(cust_id, np.array(laps))
    if evt.empty:
        return {
            f"Cnt_incident_last{recency}":    0.0,
            f"Cnt_inpits_last{recency}":      0.0,
            f"Cnt_offtrack_last{recency}":    0.0,
            f"Pct_lapcomplete_last{recency}": 0.0,
        }

    # 3) count by pattern in EventType
    et = evt['EventType'].str.lower()
    incident_count    = et.str.contains('incident').sum()
    pits_count        = et.str.contains('in pits').sum()
    offtrack_count    = et.str.contains('off track').sum()
    lapcomplete_count = et.str.contains('lap') & et.str.contains('complete')
    lapcomplete_count = lapcomplete_count.sum()

    # 4) lap-complete percentage
    lap_completion_rate = lapcomplete_count / recency

    return {
        f"Cnt_incident_last{recency}":    float(incident_count),
        f"Cnt_inpits_last{recency}":      float(pits_count),
        f"Cnt_offtrack_last{recency}":    float(offtrack_count),
        f"Pct_lapcomplete_last{recency}": lap_completion_rate,
    }

def add_checkpoint_event_features(df):
    """
    Given a DataFrame with one row per (RaceID,CarIdx) that has columns like:
       Incident_{First,25,50,75,Last},
       OffTrack_{…}, Overtakes_{…}, PositionsLost_{…}, PitStop_{…} (if available)
    this will:
      - Sum each event across all checkpoints to get a cumulative count
      - Sum each event over the last 3 checkpoints to get a recency-window count
      - Compute laps_since_last_pit if you have a PitStop flag + a 'RaceLap' column
    """
    # 1. define your checkpoints ,"Last"
    segs = ["First","25","50","75"]
    events = {
        "incident":        "Incident",
        "offtrack":        "OffTrack",
        "overtake":        "Overtakes",
        "pos_lost":        "PositionsLost",
        # if you have a raw pit-stop flag, include it here:
        "pit_stop":        "PitStop"  
    }

    # 2. cumulative & recency sums
    for evt_key, evt_prefix in events.items():
        cols = [f"{evt_prefix}_{s}" for s in segs if f"{evt_prefix}_{s}" in df.columns]

        # cumulative count
        df[f"{evt_key}_count"] = df[cols].sum(axis=1)

        # last-3 checkpoints
        last3 = cols[-3:]  # e.g. ['50','75','Last']
        df[f"{evt_key}_last3"] = df[last3].sum(axis=1)

    # 3. laps since last pit (if you have a RaceLap + pit_stop_flag at each seg)
    if "RaceLap" in df.columns and any(f"PitStop_{s}" in df.columns for s in segs):
        # first create a “pit_stop_flag” per lap by expanding out your checkpoint flags
        # here we’ll assume each row is the “current lap,” so you already have pit-stop=1 if they pitted
        df["pit_lap"] = df["RaceLap"].where(df[[f"PitStop_{s}" for s in segs if f"PitStop_{s}" in df.columns]].sum(axis=1)>0)

        # now shift per CarIdx
        last_pit_lap = (
            df
            .groupby("CarIdx")["pit_lap"]
            .apply(lambda x: x.shift().ffill())
        )
        df["laps_since_pit"] = df["RaceLap"] - last_pit_lap.fillna(df["RaceLap"])

        # clean up
        df.drop(columns=["pit_lap"], inplace=True)

    return df





def _bucket_aggregate(df: pd.DataFrame, label: str) -> Dict[str, float]:
    """
    Mean-aggregate every numeric column in *df* and apply DECAY[label].
    Identifier columns (Lap, CustId, …) are ignored.
    """
    if df.empty:
        return {}

    ignore = {"lap", "raceid", "custid", "caridx", "eventid", "RaceDate"}
    out: Dict[str, float] = {}
    for col, val in _numeric(df).mean().items():
        if col.lower() in ignore:
            continue
        out[f"{col}_{label}"] = val * DECAY[label]
    return out


# ─────────────────────────── driver-level aggregation ───────────────────────


def build_driver_features(driver_row: pd.Series, race_row: pd.Series) -> pd.Series:
    cust_id: str = driver_row.CustID
    car_idx: int = int(driver_row.CarIdx)
    
    total_laps_planned = race_row.TotalLaps  or 0

    win = window_size(total_laps_planned)
    # -------------------- copy ALL Race + Driver columns --------------------
    race_feats = {f"Race_{c}": race_row[c] for c in race_row.index if c != "RaceID"}
    drv_feats = {
        f"Drv_{c}": driver_row[c]
        for c in driver_row.index
        if c not in {"CustID", "RaceID", "CarIdx"}
    }

    lap_df = fetch_lap_data(cust_id)
    # Drop 'LiveScoreAdjusted' and 'LiveLapsComplete' 
    if lap_df.empty:
        return pd.Series({"RaceID": race_row.RaceID, "CarIdx": car_idx, **race_feats, **drv_feats})

    

    # max_lap = int(lap_df.Lap.max())
    
    
    # last_lap_rows = lap_df[lap_df['Lap'] == max_lap]
    
    # # 2) pick one live position (e.g. the last one)
    # if not last_lap_rows.empty:
    #     lap_live_pos = last_lap_rows['LivePosition'].iloc[-1]
    # else:
    #     lap_live_pos = np.nan   # or some default


    lap_df = lap_df.drop(columns=[c for c in ["LiveScoreAdjusted","Position","LivePosition",  "F2Time",  "DriverGapInFront","PodiumProximity"] if c in lap_df.columns])

    # -------------------- bucket windows once ------------------------------
    bucket_laps = {
        label: bucket_window(lap_df, frac, total_laps_planned,win) for label, frac in BUCKETS.items()
    }

    bucket_feats: Dict[str, float] = {}
    for label, df_slice in bucket_laps.items():
        # ‣ RealTimeLapData stats
        bucket_feats.update(_bucket_aggregate(df_slice, label))

        # ‣ RealTimeEvents stats by event group
        evt = fetch_events_for_laps(cust_id, df_slice.Lap.unique())
        if not evt.empty:
            # Define flexible matching rules
            event_groups = {
                "lapcomplete": lambda s: "lap" in s.lower() and "complete" in s.lower(),
                "positionchange": lambda s: "position" in s.lower() and "change" in s.lower(),
                "inpits": lambda s: "pit" in s.lower(),
                "incident": lambda s: "incident" in s.lower(),
                "offtrack": lambda s: "off" in s.lower() and "track" in s.lower(),
            }

            for group_name, match_fn in event_groups.items():
                matching_events = evt[evt["EventType"].apply(lambda s: match_fn(str(s)))]
                if not matching_events.empty:
                    # Aggregate numeric fields in this group and bucket
                    raw_bucket = _bucket_aggregate(matching_events, label)  # normal decay
                    # Rename keys with group_name prefix
                    sub_bucket = {f"{col}_{group_name}": val for col, val in raw_bucket.items()}
                    bucket_feats.update(sub_bucket)

                    # Count of rows (decay applied separately)
                    bucket_feats[f"Cnt_{group_name}_{label}"] = len(matching_events) * DECAY[label]
                    bucket_feats.update(sub_bucket)

                    # Count of how many rows of this type in this bucket
                    bucket_feats[f"Cnt_{group_name}_{label}"] = len(matching_events) * DECAY[label]

    # -------------------- historic (PastRaceTable) averages -----------------
    hist_feats: Dict[str, float] = {}
    past = fetch_past_races(cust_id)
    if not past.empty:
        for col, val in _numeric(past).mean().items():
            hist_feats[f"HistAvg_{col}"] = val
            
    recency_feats = add_recency_features(lap_df, cust_id, recency=5)
    bucket_feats.update(recency_feats)



    # -----------------------------------------------------------------------
    return pd.Series({
        "RaceID": race_row.RaceID,
        "CarIdx": car_idx,
        **race_feats,
        **drv_feats,
        **bucket_feats,
        **hist_feats
        # "LivePosition": lap_live_pos

        
    })


# ─────────────────────────── race-level aggregation ─────────────────────────


def process_race(race_row: pd.Series, livepred = False) -> pd.DataFrame:
    drivers = fetch_drivers(race_row.RaceID, livepred)
    if drivers.empty:
        return pd.DataFrame()

    rows = [build_driver_features(d, race_row) for _, d in drivers.iterrows()]
    
    
    rows = pd.DataFrame(rows)
    
    
    # -------------------- extra features ----------------------
    rows = add_checkpoint_event_features(rows)
    
    rows = zero_fill_counts(rows)
    
    rows = fill_other_missing(rows)
    
    return rows


# ──────────────────────────── top-level entry point ─────────────────────────


def process_all_races(verbose: bool = True, livepred = False) -> pd.DataFrame:
    races = fetch_races()
    out_frames: List[pd.DataFrame] = []

    for _, race_row in races.iterrows():
        if verbose:
            print(f"▸ Processing race {race_row.RaceID}")
        df = process_race(race_row, livepred)
        if not df.empty:
            out_frames.append(df)

    if out_frames:
        return pd.concat(out_frames, ignore_index=True)
    
    
    
    
    return pd.DataFrame()


if __name__ == "__main__":
    features_df = process_all_races()
    print(features_df.head())
