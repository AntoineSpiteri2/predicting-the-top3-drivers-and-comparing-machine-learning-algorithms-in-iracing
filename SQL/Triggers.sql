
USE racing_data;
use racing_data_Live
go
drop trigger trg_normalize_livescoreadjusted

GO
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
GO


GO
drop trigger trg_update_live_features
GO

GO
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

GO



GO
drop trigger trg_update_max_totallaps
GO


go
CREATE TRIGGER trg_update_max_totallaps
ON RealTimeLapData
AFTER INSERT
AS
BEGIN
    SET NOCOUNT ON;

    -- Update the MaxTotalLaps in the Driver table based on the associated RaceID and CustID
    UPDATE r
    SET r.TotalLaps = (
        SELECT MAX(rtd.Lap)
        FROM RealTimeLapData rtd
        WHERE rtd.RaceID = i.RaceID
    )
    FROM Race r
    INNER JOIN inserted i ON r.RaceID = i.RaceID;
END;

GO

drop trigger trg_Incident_Lap_Increment
GO
CREATE TRIGGER trg_Incident_Lap_Increment
ON RealTimeEvents
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Parameter for tuning the base penalty weight.
    --DECLARE @BasePenalty FLOAT = 3.5;  -- Adjust this value as needed

    -------------------------------------
    -- 1. Update the Incident count first
    -------------------------------------
    UPDATE L
    SET L.Incident = E.IncidentCount
    FROM RealTimeLapData L
    INNER JOIN (
         SELECT RaceID, CustId, Lap, COUNT(*) AS IncidentCount
         FROM RealTimeEvents
         WHERE EventType LIKE 'Incident%'
           OR RPM < 3500
           AND Speed < 125
         GROUP BY RaceID, CustId, Lap
    ) E
      ON E.RaceID = L.RaceID
     AND E.CustId = L.CustId
     AND E.Lap = L.Lap
    WHERE EXISTS (
         SELECT 1 
         FROM inserted I
         WHERE I.RaceID = L.RaceID
           AND I.CustId = L.CustId
           AND I.Lap = L.Lap
           AND I.EventType LIKE 'Incident%'
           or I.RPM < 3500
           AND I.Speed < 125
    );

END;
GO

DROP TRIGGER trg_OffTrack_SmallPenalty
GO
CREATE TRIGGER trg_OffTrack_SmallPenalty
ON RealTimeEvents
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Use a smaller base penalty for off track events
    DECLARE @BasePenalty FLOAT = 0.15;  

    UPDATE L
    SET 
		L.OffTrack = IncidentData.OffTrackCount

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
GO

drop trigger trg_UpdateOvertakesOnly

GO
CREATE TRIGGER trg_UpdateOvertakesOnly
ON RealTimeLapData
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Prevent recursive trigger firing
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
GO

drop trigger trg_UpdatePositionsLost
GO
CREATE TRIGGER trg_UpdatePositionsLost
ON RealTimeLapData
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Prevent recursive trigger firing
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
    WHERE i.CustId IN (SELECT DISTINCT CustID FROM inserted);
END;
GO

drop trigger trg_LapTimeDelta
go
CREATE TRIGGER trg_LapTimeDelta
ON RealTimeLapData
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Prevent recursive trigger firing
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
GO
drop trigger trg_NormDriverGapToLeader
go
CREATE TRIGGER trg_NormDriverGapToLeader
ON RealTimeLapData
AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- Prevent recursive trigger firing
    IF TRIGGER_NESTLEVEL() > 1 RETURN;

    ;WITH CTE AS (
        -- Calculate the cumulative gap from the leader for each row
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
        -- For each RaceID and Lap, get the minimum and maximum cumulative gap
        SELECT 
            RaceID,
            Lap,
            MAX(CumGap) AS MaxCumGap,
            MIN(CumGap) AS MinCumGap
        FROM CTE
        GROUP BY RaceID, Lap
    ),
    normgap AS (
        -- Normalize the cumulative gap using min–max normalization:
        --   NormScore = (MaxCumGap - CumGap) / (MaxCumGap - MinCumGap)
        -- When all drivers have the same gap, default the score to 1.
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
GO


GO
drop trigger trg_FixLapPositions
go

CREATE TRIGGER dbo.trg_FixLapPositions
  ON dbo.RealTimeLapData
  AFTER INSERT, UPDATE
AS
BEGIN
    SET NOCOUNT ON;

    -- 1) Which Race/Laps have changed?
    WITH ChangedLaps AS (
        SELECT DISTINCT RaceID, Lap
        FROM inserted
    )
    -- 2) Re-rank only those laps
    , RankedPositions AS (
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
    -- 3) Bulk-update the Position column
    UPDATE rtd
    SET rtd.Position = rp.NewPos
    FROM dbo.RealTimeLapData AS rtd
    INNER JOIN RankedPositions AS rp
      ON rtd.RaceID = rp.RaceID
     AND rtd.Lap    = rp.Lap
     AND rtd.CustId = rp.CustId;
END;
GO
drop trigger trg_PitStop_Penalty

GO
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
GO

