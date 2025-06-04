USE racing_data;
use racing_data_Live

------ Step 1: Drop tables if necessary (Uncomment if you need to reset the schema)
 DROP TABLE RealTimeEvents;
 DROP TABLE RealTimeLapData;
 DROP TABLE PastRaceTable;
 DROP TABLE Driver;
 DROP TABLE Race;

-- Step 2: Delete all data from dependent tables
-- Ensure ON DELETE CASCADE is set for child tables; otherwise, delete child records manually.

DELETE FROM RealTimeEvents where raceid = 33
Delete from RealTimeLapData where raceid = 33
DELETE FROM Driver where raceid = 33
delete Race where raceid = 33
-- Delete all data from Driver (child records will be deleted automatically via ON DELETE CASCADE)



--Delete from RealTimeEvents where RaceID =  1083 
--DELETE FROM RealTimeLapData where RaceID =  1083 
--DELETE FROM Driver where RaceID =  1083 
--DELETE FROM Race where RaceID =  1083 



DELETE FROM RealTimeEvents 
Delete from RealTimeLapData 
DELETE FROM PastRaceTable 
DELETE FROM Driver 
delete Race 

DBCC CHECKIDENT ('Race', RESEED, 0);               -- Reset RaceID to start from 1
DBCC CHECKIDENT ('RealTimeEvents', RESEED, 0);    -- Reset EventID to start from 1

-- Step 4: Confirm identity resets and data cleanup
--SELECT * FROM Race;
--SELECT * FROM Driver;
--SELECT * FROM RealTimeLapData;
--SELECT * FROM RealTimeEvents;




