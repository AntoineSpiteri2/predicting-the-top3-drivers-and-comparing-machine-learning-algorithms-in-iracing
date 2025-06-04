USE racing_data;
use racing_data_Live



-- Select queries to verify the data
SELECT * FROM Race;

SELECT * FROM Driver   order by RaceID, RankScoreAdjusted desc


select * from PastRaceTable
SELECT *   FROM RealTimeLapData  where RaceID = 48   order by RaceID, Lap, Position desc 
select * from RealTimeEvents where RaceID = 43  order by lap, EventType


---- fix live pos dup pos

--WITH MaxLaps AS (
--    SELECT
--        RaceID,
--        MAX(Lap) AS MaxLap
--    FROM RealTimeLapData
--    GROUP BY RaceID
--),
--RankedPositions AS (
--    SELECT
--        r.RaceID,
--        r.Lap,
--        r.CustId,
--        ROW_NUMBER() OVER (
--            PARTITION BY r.RaceID, r.Lap
--            ORDER BY r.NormDriverGapToLeader DESC
--        ) AS NewPos
--    FROM RealTimeLapData r
--    INNER JOIN MaxLaps m
--        ON r.RaceID = m.RaceID
--       AND r.Lap    = m.MaxLap
--)
--UPDATE rtd
--SET LivePosition = rp.NewPos
--FROM RealTimeLapData rtd
--INNER JOIN RankedPositions rp
--    ON rtd.RaceID = rp.RaceID
--   AND rtd.Lap    = rp.Lap
--   AND rtd.CustId = rp.CustId;


--SELECT * 
--FROM PastRaceTable 
--WHERE CustID IN (SELECT CustID 
--                FROM Driver 
--                WHERE raceid = 1072);

--SELECT * FROM Driver where RaceID = 1 order by  RankScoreAdjusted desc


--SELECT * from  RealTimeLapData   order by  lap, Position

--SELECT CarIdx, custid ,RankScoreAdjusted, Disqualified FROM Driver      order by  RankScoreAdjusted desc;



-- fix duplicated posistions in dataset
--WITH RankedPositions AS (
--    SELECT 
--        RaceID,
--        Lap,
--        CustId,
--        ROW_NUMBER() OVER (PARTITION BY RaceID, Lap ORDER BY NormDriverGapToLeader DESC) AS NewPos
--    FROM RealTimeLapData
--)
--UPDATE rtd
--SET Position = rp.NewPos
--FROM RealTimeLapData rtd
--INNER JOIN RankedPositions rp
--    ON rtd.RaceID = rp.RaceID
--   AND rtd.Lap = rp.Lap
--   AND rtd.CustId = rp.CustId;


--UPDATE RealTimeEvents
--SET EventType = 
--    CASE 
--        WHEN RPM < 3000 AND Speed < 150 THEN 'Incident'
--        ELSE 'Off Track'
--    END
--where EventType =  'Incident'


--INSERT INTO RealTimeEvents (RaceID, CustId, Lap, TrackSurface, Speed, RPM, Gear, CarSteer,
--    SteerIntensity, SteerSpeedRatio, lapdistpct, LapPhase, PitRoad, FastRepairsUsed,
--    RpmPerGear, SpeedPerGear, AggressiveManeuver, CatchUpPotential, AvgOvertakeRate, EventType, EventTimestamp)
--SELECT 
--    e.RaceID, e.CustId, e.Lap, e.TrackSurface, e.Speed, e.RPM, e.Gear, e.CarSteer,
--    e.SteerIntensity, e.SteerSpeedRatio, e.lapdistpct, e.LapPhase, e.PitRoad, e.FastRepairsUsed,
--    e.RpmPerGear, e.SpeedPerGear, e.AggressiveManeuver, e.CatchUpPotential, e.AvgOvertakeRate,
--    'In Pits', GETDATE()
--FROM RealTimeEvents e
--WHERE e.EventType = 'Position Change' and e.PitRoad = 1
--AND NOT EXISTS (
--    SELECT 1 FROM RealTimeEvents e2
--    WHERE e2.CustId = e.CustId 
--    AND e2.Lap = e.Lap 
--    AND e2.EventType = 'in Pits'
--);



---- manually activate triggers 
--UPDATE RealTimeLapData
--SET inpits  = 0;

--UPDATE RealTimeLapData
--SET OffTrack  = 0;

--UPDATE RealTimeLapData
--SET Incidents = 0;


--UPDATE RealTimeLapData
--SET Position = Position;

--UPDATE RealTimeEvents
--SET EventType = EventType 





------------------------------------




