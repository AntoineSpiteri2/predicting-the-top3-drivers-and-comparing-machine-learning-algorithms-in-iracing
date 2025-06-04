"""
File: remap_ids_and_drop_new_column_with_subsession.py

Description:
    Extends remapping of CustID → integer by also remapping PastRaceTable.subessionid → integer.
    1. Drops existing triggers on RealTimeLapData and RealTimeEvents (so they don’t fire mid‐update).
    2. Adds NewCustId to Driver, RealTimeLapData, RealTimeEvents, PastRaceTable. 
    3. Remaps old CustID → new_int (1..N) in Driver, propagates into children, overwrites CustID, drops NewCustId. 
    4. Adds NewSubsessionId to PastRaceTable, remaps old subessionid → new_int (1..M), overwrites subessionid, drops NewSubsessionId.
    5. Re-creates all original triggers.
    6. Commits at the end, or rolls back on error.

Prerequisites:
    • utils/DatabaseConnection.py (in your PYTHONPATH) must define:
          def ConnectToDatabase() -> pyodbc.Connection
      which returns a live connection to the target database (e.g. racing_data_Live).
    • pyodbc installed (pip install pyodbc).
    • BACK UP your database before running—this operation is destructive for CustID and subsessionid values.
"""

import sys
import pyodbc
from utils.DatabaseConnection import ConnectToDatabase

def remap_ids_and_drop_new_column_with_subsession():
    # 1. CONNECT
    conn = ConnectToDatabase()
    if conn is None:
        print("❌ Could not establish a database connection. Aborting.")
        sys.exit(1)

    cursor = conn.cursor()
    try:
        # Ensure all steps run in a single transaction:
        conn.autocommit = False

        # ----------------------------------------------------------------
        # 2. DROP EXISTING TRIGGERS ON RealTimeLapData AND RealTimeEvents
        # ----------------------------------------------------------------
        print("► Dropping existing triggers on RealTimeLapData and RealTimeEvents…")
        drop_stmts = [
            "IF OBJECT_ID('trg_normalize_livescoreadjusted','TR') IS NOT NULL DROP TRIGGER trg_normalize_livescoreadjusted;",
            "IF OBJECT_ID('trg_update_live_features','TR') IS NOT NULL DROP TRIGGER trg_update_live_features;",
            "IF OBJECT_ID('trg_update_max_totallaps','TR') IS NOT NULL DROP TRIGGER trg_update_max_totallaps;",
            "IF OBJECT_ID('trg_Incident_Lap_Increment','TR') IS NOT NULL DROP TRIGGER trg_Incident_Lap_Increment;",
            "IF OBJECT_ID('trg_OffTrack_SmallPenalty','TR') IS NOT NULL DROP TRIGGER trg_OffTrack_SmallPenalty;",
            "IF OBJECT_ID('trg_UpdateOvertakesOnly','TR') IS NOT NULL DROP TRIGGER trg_UpdateOvertakesOnly;",
            "IF OBJECT_ID('trg_UpdatePositionsLost','TR') IS NOT NULL DROP TRIGGER trg_UpdatePositionsLost;",
            "IF OBJECT_ID('trg_LapTimeDelta','TR') IS NOT NULL DROP TRIGGER trg_LapTimeDelta;",
            "IF OBJECT_ID('trg_NormDriverGapToLeader','TR') IS NOT NULL DROP TRIGGER trg_NormDriverGapToLeader;",
            "IF OBJECT_ID('trg_FixLapPositions','TR') IS NOT NULL DROP TRIGGER trg_FixLapPositions;",
            "IF OBJECT_ID('trg_PitStop_Penalty','TR') IS NOT NULL DROP TRIGGER trg_PitStop_Penalty;"
        ]
        for stmt in drop_stmts:
            cursor.execute(stmt)
        print("    → All listed triggers dropped (if they existed).")

        # ----------------------------------------------------------------
        # 3. ADD NewCustId COLUMN (IF MISSING) TO EACH TABLE INVOLVED IN CustID REMAP
        # ----------------------------------------------------------------
        print("\n► Adding NewCustId column to each table (if missing)…")
        tables_with_cust = ["Driver", "RealTimeLapData", "RealTimeEvents", "PastRaceTable"]
        for tbl in tables_with_cust:
            cursor.execute(f"""
                IF COL_LENGTH('{tbl}', 'NewCustId') IS NULL
                BEGIN
                    ALTER TABLE {tbl} ADD NewCustId INT NULL;
                END
            """)
        print("    → Finished adding NewCustId where needed.")

        # ----------------------------------------------------------------
        # 4. BUILD MAPPING old CustId → new integer IN Driver
        # ----------------------------------------------------------------
        print("\n► Building mapping of old CustId → new integer in Driver …")
        cursor.execute("SELECT DISTINCT CustId FROM Driver ORDER BY CustId;")
        distinct_rows = cursor.fetchall()
        custid_to_int = { row.CustId: idx for idx, row in enumerate(distinct_rows, start=1) }
        print(f"    → Found {len(custid_to_int)} distinct CustId values.")

        # 5. POPULATE Driver.NewCustId
        print("► Populating Driver.NewCustId …")
        for old_str, new_int in custid_to_int.items():
            cursor.execute(
                "UPDATE Driver SET NewCustId = ? WHERE CustId = ?;",
                new_int,
                old_str
            )
        print("    → Driver.NewCustId populated.")

        # 6. PROPAGATE NewCustId INTO CHILD TABLES (join on old CustId)
        print("\n► Propagating NewCustId into RealTimeLapData …")
        cursor.execute("""
            UPDATE L
               SET L.NewCustId = D.NewCustId
            FROM RealTimeLapData AS L
            JOIN Driver            AS D
              ON L.CustId = D.CustId;
        """)
        print("    → RealTimeLapData.NewCustId populated.")

        print("► Propagating NewCustId into RealTimeEvents …")
        cursor.execute("""
            UPDATE E
               SET E.NewCustId = D.NewCustId
            FROM RealTimeEvents AS E
            JOIN Driver          AS D
              ON E.CustId = D.CustId;
        """)
        print("    → RealTimeEvents.NewCustId populated.")

        print("► Propagating NewCustId into PastRaceTable …")
        cursor.execute("""
            UPDATE P
               SET P.NewCustId = D.NewCustId
            FROM PastRaceTable AS P
            JOIN Driver         AS D
              ON P.CustId = D.CustId;
        """)
        print("    → PastRaceTable.NewCustId populated.")

        # ----------------------------------------------------------------
        # 7. DISABLE FKs on child tables before overwriting CustId
        # ----------------------------------------------------------------
        print("\n► Disabling foreign-key constraints on child tables …")
        cursor.execute("ALTER TABLE RealTimeEvents    NOCHECK CONSTRAINT ALL;")
        cursor.execute("ALTER TABLE RealTimeLapData   NOCHECK CONSTRAINT ALL;")
        cursor.execute("ALTER TABLE PastRaceTable     NOCHECK CONSTRAINT ALL;")
        print("    → Foreign-key constraints disabled.")

        # ----------------------------------------------------------------
        # 8. OVERWRITE CustId (bottom-up: children first, then Driver)
        #    CAST INT → NVARCHAR(256) to match original datatype
        # ----------------------------------------------------------------
        print("\n► Updating RealTimeLapData.CustId ← NewCustId …")
        cursor.execute("""
            UPDATE RealTimeLapData
               SET CustId = CAST(NewCustId AS NVARCHAR(256))
             WHERE NewCustId IS NOT NULL;
        """)
        print("    → RealTimeLapData.CustId updated.")

        print("► Updating RealTimeEvents.CustId ← NewCustId …")
        cursor.execute("""
            UPDATE RealTimeEvents
               SET CustId = CAST(NewCustId AS NVARCHAR(256))
             WHERE NewCustId IS NOT NULL;
        """)
        print("    → RealTimeEvents.CustId updated.")

        print("► Updating PastRaceTable.CustId ← NewCustId …")
        cursor.execute("""
            UPDATE PastRaceTable
               SET CustId = CAST(NewCustId AS NVARCHAR(256))
             WHERE NewCustId IS NOT NULL;
        """)
        print("    → PastRaceTable.CustId updated.")

        print("► Updating Driver.CustId ← NewCustId …")
        cursor.execute("""
            UPDATE Driver
               SET CustId = CAST(NewCustId AS NVARCHAR(256))
             WHERE NewCustId IS NOT NULL;
        """)
        print("    → Driver.CustId updated.")

        # ----------------------------------------------------------------
        # 9. RE-ENABLE FKs (WITH CHECK) so SQL Server re-validates children
        # ----------------------------------------------------------------
        print("\n► Re-enabling foreign-key constraints (WITH CHECK) …")
        cursor.execute("ALTER TABLE RealTimeEvents    WITH CHECK CHECK CONSTRAINT ALL;")
        cursor.execute("ALTER TABLE RealTimeLapData   WITH CHECK CHECK CONSTRAINT ALL;")
        cursor.execute("ALTER TABLE PastRaceTable     WITH CHECK CHECK CONSTRAINT ALL;")
        print("    → Foreign-key constraints re-enabled and validated.")

        # ----------------------------------------------------------------
        # 10. DROP NewCustId columns from every table
        # ----------------------------------------------------------------
        print("\n► Dropping NewCustId columns from all tables …")
        for tbl in tables_with_cust:
            cursor.execute(f"ALTER TABLE {tbl} DROP COLUMN NewCustId;")
            print(f"    → Dropped NewCustId from {tbl}")

        # ----------------------------------------------------------------
        # 11. NOW REPEAT SIMILAR STEPS FOR subessionid IN PastRaceTable
        # ----------------------------------------------------------------

        # 11a) ADD NewSubsessionId (if missing)
        print("\n► Adding NewSubsessionId column to PastRaceTable (if missing)…")
        cursor.execute("""
            IF COL_LENGTH('PastRaceTable', 'NewSubsessionId') IS NULL
            BEGIN
                ALTER TABLE PastRaceTable ADD NewSubsessionId INT NULL;
            END
        """)
        print("    → NewSubsessionId added to PastRaceTable (if it did not already exist).")

        # 11b) BUILD MAPPING old subessionid → new integer
        print("\n► Building mapping of old subessionid → new integer in PastRaceTable …")
        cursor.execute("SELECT DISTINCT subessionid FROM PastRaceTable ORDER BY subessionid;")
        distinct_ss = cursor.fetchall()
        ss_to_int = { row.subessionid: idx for idx, row in enumerate(distinct_ss, start=1) }
        print(f"    → Found {len(ss_to_int)} distinct subessionid values in PastRaceTable.")

        # 11c) POPULATE PastRaceTable.NewSubsessionId
        print("► Populating PastRaceTable.NewSubsessionId …")
        for old_ss, new_ss in ss_to_int.items():
            cursor.execute(
                "UPDATE PastRaceTable SET NewSubsessionId = ? WHERE subessionid = ?;",
                new_ss,
                old_ss
            )
        print("    → PastRaceTable.NewSubsessionId populated.")

        # 11d) OVERWRITE PastRaceTable.subessionid ← NewSubsessionId
        #      (No FK to worry about, but it is part of the primary key. Because each old_ss maps to a unique new_ss,
        #       the (NewSubsessionId, CustID) pairs remain unique.)
        print("\n► Updating PastRaceTable.subessionid ← NewSubsessionId …")
        cursor.execute("""
            UPDATE PastRaceTable
               SET subessionid = NewSubsessionId
             WHERE NewSubsessionId IS NOT NULL;
        """)
        print("    → PastRaceTable.subessionid updated.")

        # 11e) DROP the NewSubsessionId column
        print("\n► Dropping NewSubsessionId column from PastRaceTable …")
        cursor.execute("ALTER TABLE PastRaceTable DROP COLUMN NewSubsessionId;")
        print("    → NewSubsessionId dropped from PastRaceTable.")

        # ----------------------------------------------------------------
        # 12. RE-CREATE ALL TRIGGERS (exactly as they were originally)
        # ----------------------------------------------------------------
        print("\n► Re-creating all triggers on RealTimeLapData and RealTimeEvents…")

        # 12a) trg_normalize_livescoreadjusted
        cursor.execute("""
        CREATE TRIGGER trg_normalize_livescoreadjusted
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            -- Prevent recursive trigger firing
            IF TRIGGER_NESTLEVEL() > 1 
                RETURN;

            WITH RaceAgg AS (
                SELECT
                    RaceID,
                    MAX(Position) AS MaxPosition,
                    MIN(Position) AS MinPosition
                FROM RealTimeLapData
                GROUP BY RaceID
            )
            UPDATE rtd
            SET LiveScoreAdjusted = ROUND(
                  1 * (
                    CAST(ra.MaxPosition AS DECIMAL(38,18)) - CAST(rtd.Position AS DECIMAL(38,18))
                  )
                  / NULLIF(
                    CAST(ra.MaxPosition AS DECIMAL(38,18)) - CAST(ra.MinPosition AS DECIMAL(38,18)), 0
                  )
            , 3)
            FROM RealTimeLapData rtd
            INNER JOIN RaceAgg ra ON rtd.RaceID = ra.RaceID;
        END;
        """)

        # 12b) trg_update_live_features
        cursor.execute("""
        CREATE TRIGGER trg_update_live_features
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            -- Prevent recursive firing
            IF TRIGGER_NESTLEVEL() > 1 
                RETURN;

            ----------------------------------------
            -- Step 1: Update PositionVolatility
            ----------------------------------------
            WITH PositionStats AS (
                SELECT
                    r.CustID,
                    STDEV(r.Position) AS PositionVolatility
                FROM RealTimeLapData r
                GROUP BY r.CustID
            )
            UPDATE r
            SET r.PositionVolatility = ISNULL(ps.PositionVolatility, 0)
            FROM RealTimeLapData r
            INNER JOIN PositionStats ps 
                ON r.CustID = ps.CustID
            WHERE r.CustID IN (
                SELECT DISTINCT CustID
                FROM inserted
            );

            ----------------------------------------
            -- Step 2-5: Compute LapCompletionRate and Final Score
            ----------------------------------------
            ;WITH LapCompletionRateCalc AS (
                SELECT
                    rtd.CustID,
                    rtd.RaceID,
                    rtd.LiveScoreAdjusted,
                    rtd.Position,
                    rtd.Lap,
                    CASE
                        WHEN rtd.Lap = MAX(rtd.Lap) OVER (PARTITION BY rtd.RaceID)
                        THEN (rtd.Lap * 2.5) / MAX(rtd.Lap) OVER (PARTITION BY rtd.RaceID)
                        ELSE (rtd.Lap * 1.0) / MAX(rtd.Lap) OVER (PARTITION BY rtd.RaceID)
                    END AS LapCompletionRate
                FROM RealTimeLapData rtd
            ),
            LastInsertedWeightedLiveScore AS (
                SELECT
                    lcr.CustID,
                    lcr.RaceID,
                    lcr.LiveScoreAdjusted,
                    lcr.Position,
                    lcr.LapCompletionRate,
                    lcr.Lap,
                    lcr.LiveScoreAdjusted * lcr.LapCompletionRate AS WeightedLiveScoreAdjusted
                FROM LapCompletionRateCalc lcr
                WHERE lcr.Lap = (
                    SELECT MAX(rtd.Lap)
                    FROM RealTimeLapData rtd
                    WHERE rtd.RaceID = lcr.RaceID
                      AND rtd.CustID = lcr.CustID
                )
            ),
            RankScoreCalc AS (
                SELECT
                    d.CustID,
                    COALESCE(MAX(wls.WeightedLiveScoreAdjusted), 0) AS FinalWeightedScore
                FROM LastInsertedWeightedLiveScore wls
                RIGHT JOIN Driver d 
                    ON wls.CustID = d.CustID
                GROUP BY d.CustID
            )
            UPDATE d
            SET d.RankScoreAdjusted = rsc.FinalWeightedScore
            FROM Driver d
            INNER JOIN RankScoreCalc rsc 
                ON d.CustID = rsc.CustID;
        END;
        """)

        # 12c) trg_update_max_totallaps
        cursor.execute("""
        CREATE TRIGGER trg_update_max_totallaps
        ON RealTimeLapData
        AFTER INSERT
        AS
        BEGIN
            SET NOCOUNT ON;

            -- Update the MaxTotalLaps in the Driver table based on the associated RaceID
            UPDATE r
            SET r.TotalLaps = (
                SELECT MAX(rtd.Lap)
                FROM RealTimeLapData rtd
                WHERE rtd.RaceID = i.RaceID
            )
            FROM Race r
            INNER JOIN inserted i ON r.RaceID = i.RaceID;
        END;
        """)

        # 12d) trg_Incident_Lap_Increment
        cursor.execute("""
        CREATE TRIGGER trg_Incident_Lap_Increment
        ON RealTimeEvents
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            -- Update the Incident count in RealTimeLapData
            UPDATE L
            SET L.Incident = E.IncidentCount
            FROM RealTimeLapData L
            INNER JOIN (
                 SELECT RaceID, CustId, Lap, COUNT(*) AS IncidentCount
                 FROM RealTimeEvents
                 WHERE (EventType LIKE 'Incident%' OR (RPM < 3500 AND Speed < 125))
                 GROUP BY RaceID, CustId, Lap
            ) E
              ON E.RaceID = L.RaceID
             AND E.CustId = L.CustId
             AND E.Lap = L.Lap
            WHERE EXISTS (
                 SELECT 1 
                 FROM inserted I
                 WHERE (I.RaceID = L.RaceID AND I.CustId = L.CustId AND I.Lap = L.Lap)
                   AND (I.EventType LIKE 'Incident%' OR (I.RPM < 3500 AND I.Speed < 125))
            );
        END;
        """)

        # 12e) trg_OffTrack_SmallPenalty
        cursor.execute("""
        CREATE TRIGGER trg_OffTrack_SmallPenalty
        ON RealTimeEvents
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            DECLARE @BasePenalty FLOAT = 0.15;  

            UPDATE L
            SET L.OffTrack = IncidentData.OffTrackCount
            FROM RealTimeLapData L
            CROSS APPLY (
                SELECT COUNT(*) AS OffTrackCount
                FROM RealTimeEvents E
                WHERE E.RaceID = L.RaceID
                  AND E.CustId = L.CustId
                  AND E.Lap = L.Lap
                  AND E.EventType LIKE 'Off Track%'
            ) IncidentData
            CROSS APPLY (
                SELECT COUNT(DISTINCT D.CustId) AS TotalDrivers
                FROM Driver D
                WHERE D.RaceID = L.RaceID
            ) DriverCount
            WHERE EXISTS (
                SELECT 1 
                FROM inserted I
                WHERE I.RaceID = L.RaceID
                  AND I.CustId = L.CustId
                  AND I.Lap = L.Lap
                  AND I.EventType LIKE 'Off Track%'
            );
        END;
        """)

        # 12f) trg_UpdateOvertakesOnly
        cursor.execute("""
        CREATE TRIGGER trg_UpdateOvertakesOnly
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            IF TRIGGER_NESTLEVEL() > 1 RETURN;

            UPDATE i
            SET i.Overtakes = CASE 
                                    WHEN i.Lap = 1 THEN 0
                                    WHEN previous.Position > i.Position 
                                         THEN previous.Position - i.Position
                                    ELSE 0
                                 END
            FROM RealTimeLapData AS i
            LEFT JOIN RealTimeLapData AS previous
                 ON i.CustId = previous.CustId
                 AND i.RaceID = previous.RaceID
                 AND i.Lap = previous.Lap + 1
            WHERE i.CustId IN (SELECT DISTINCT CustId FROM inserted);
        END;
        """)

        # 12g) trg_UpdatePositionsLost
        cursor.execute("""
        CREATE TRIGGER trg_UpdatePositionsLost
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            IF TRIGGER_NESTLEVEL() > 1 RETURN;

            UPDATE i
            SET i.PositionsLost = CASE 
                                      WHEN i.Lap = 1 THEN 0
                                      WHEN previous.Position < i.Position 
                                           THEN i.Position - previous.Position
                                      ELSE 0
                                  END
            FROM RealTimeLapData AS i
            LEFT JOIN RealTimeLapData AS previous
                 ON i.CustId = previous.CustId
                 AND i.RaceID = previous.RaceID
                 AND i.Lap = previous.Lap + 1
            WHERE i.CustId IN (SELECT DISTINCT CustId FROM inserted);
        END;
        """)

        # 12h) trg_LapTimeDelta
        cursor.execute("""
        CREATE TRIGGER trg_LapTimeDelta
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            IF TRIGGER_NESTLEVEL() > 1 RETURN;

            UPDATE r
            SET r.LapTimeDelta = 
                CASE 
                    WHEN i.LastLapTime = 0 OR i.BestLapTime = 0 THEN 0
                    ELSE i.BestLapTime - i.LastLapTime 
                END
            FROM RealTimeLapData AS r
            INNER JOIN inserted AS i
                ON r.CustId = i.CustId
               AND r.RaceID = i.RaceID
               AND r.Lap = i.Lap;
        END;
        """)

        # 12i) trg_NormDriverGapToLeader
        cursor.execute("""
        CREATE TRIGGER trg_NormDriverGapToLeader
        ON RealTimeLapData
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            IF TRIGGER_NESTLEVEL() > 1 RETURN;

            ;WITH CTE AS (
                SELECT 
                    r.CustId,
                    r.RaceID,
                    r.Lap,
                    r.Position,
                    r.DriverGapInFront,
                    SUM(r.DriverGapInFront) OVER (
                        PARTITION BY r.RaceID, r.Lap 
                        ORDER BY r.Position 
                        ROWS UNBOUNDED PRECEDING
                    ) AS CumGap
                FROM RealTimeLapData r
                INNER JOIN inserted i
                    ON r.RaceID = i.RaceID
                   AND r.Lap = i.Lap
            ),
            RaceAgg AS (
                SELECT 
                    RaceID,
                    Lap,
                    MAX(CumGap) AS MaxCumGap,
                    MIN(CumGap) AS MinCumGap
                FROM CTE
                GROUP BY RaceID, Lap
            ),
            normgap AS (
                SELECT 
                    c.CustId,
                    c.RaceID,
                    c.Lap,
                    CASE 
                        WHEN ra.MaxCumGap = ra.MinCumGap THEN 1.0
                        ELSE (ra.MaxCumGap - c.CumGap) / (ra.MaxCumGap - ra.MinCumGap)
                    END AS NormScore
                FROM CTE c
                INNER JOIN RaceAgg ra
                    ON c.RaceID = ra.RaceID
                   AND c.Lap = ra.Lap
            )
            UPDATE r
            SET r.NormDriverGapToLeader = n.NormScore
            FROM RealTimeLapData r
            INNER JOIN normgap n
                ON r.CustId = n.CustId
               AND r.RaceID = n.RaceID
               AND r.Lap = n.Lap;
        END;
        """)

        # 12j) trg_FixLapPositions
        cursor.execute("""
        CREATE TRIGGER trg_FixLapPositions
          ON RealTimeLapData
          AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            ;WITH ChangedLaps AS (
                SELECT DISTINCT RaceID, Lap
                FROM inserted
            ),
            RankedPositions AS (
                SELECT
                    r.RaceID,
                    r.Lap,
                    r.CustId,
                    ROW_NUMBER()
                      OVER (
                        PARTITION BY r.RaceID, r.Lap
                        ORDER BY r.NormDriverGapToLeader DESC
                      ) AS NewPos
                FROM dbo.RealTimeLapData AS r
                INNER JOIN ChangedLaps AS c
                  ON r.RaceID = c.RaceID
                 AND r.Lap    = c.Lap
            )
            UPDATE rtd
            SET rtd.Position = rp.NewPos
            FROM dbo.RealTimeLapData AS rtd
            INNER JOIN RankedPositions AS rp
              ON rtd.RaceID = rp.RaceID
             AND rtd.Lap    = rp.Lap
             AND rtd.CustId = rp.CustId;
        END;
        """)

        # 12k) trg_PitStop_Penalty
        cursor.execute("""
        CREATE TRIGGER trg_PitStop_Penalty
        ON RealTimeEvents
        AFTER INSERT, UPDATE
        AS
        BEGIN
            SET NOCOUNT ON;

            UPDATE L
            SET 
                L.inpits += 1
            FROM RealTimeLapData L
            WHERE EXISTS (
                SELECT 1 
                FROM inserted I
                WHERE I.RaceID = L.RaceID
                  AND I.CustId = L.CustId
                  AND I.Lap = L.Lap
                  AND I.TrackSurface = 'In Pit Stall'
            );
        END;
        """)
        print("    → All triggers recreated successfully.")

        # ----------------------------------------------------------------
        # 13. COMMIT ALL CHANGES
        # ----------------------------------------------------------------
        conn.commit()
        print("\n✔ Remapping complete—CustID & subessionid replaced; helper columns dropped; triggers re-created. Transaction committed.")

    except Exception as ex:
        # Roll back everything on error
        print("‼ Error occurred – rolling back changes:\n", ex)
        conn.rollback()
    finally:
        cursor.close()
        conn.close()
        print("Connection closed.")


if __name__ == "__main__":
    remap_ids_and_drop_new_column_with_subsession()
