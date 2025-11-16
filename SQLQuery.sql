USE PakWheelsDB;
GO

SELECT COUNT(*) AS TotalCars FROM Cars;
SELECT TOP 10 * FROM Cars;

SELECT COUNT(*) AS TotalAnalysis FROM AnalysisResults;
SELECT TOP 10 * FROM AnalysisResults ORDER BY InsertedAt DESC;


USE PakWheelsDB;
GO

IF OBJECT_ID('Analysis') IS NOT NULL 
    DROP TABLE Analysis;
GO

CREATE TABLE Analysis (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    MetricName NVARCHAR(200),
    MetricValue NVARCHAR(200),
    Description NVARCHAR(500),
    InsertedAt DATETIME2 DEFAULT GETDATE()
);
GO



USE PakWheelsDB;
GO

IF OBJECT_ID('DataFrameAnalysis') IS NOT NULL 
    DROP TABLE DataFrameAnalysis;
GO

CREATE TABLE DataFrameAnalysis (
    Id INT IDENTITY(1,1) PRIMARY KEY,
    ColumnName NVARCHAR(200),
    StatType NVARCHAR(100),     -- e.g., mean, max, min, corr
    StatValue NVARCHAR(200),
    InsertedAt DATETIME2 DEFAULT GETDATE()
);
GO


SELECT TOP 10 * FROM Cars;


USE PakWheelsDB;
GO

SELECT * FROM Analysis ORDER BY InsertedAt DESC;

SELECT * FROM DataFrameAnalysis ORDER BY InsertedAt DESC;
