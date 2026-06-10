USE [LAB_REPAIR];
GO

-- 0. Odświeżamy słownik słów
IF OBJECT_ID('tempdb..#DostepneSlowa') IS NOT NULL DROP TABLE #DostepneSlowa;
SELECT DISTINCT display_term 
INTO #DostepneSlowa
FROM sys.dm_fts_index_keywords(DB_ID('LAB_REPAIR'), OBJECT_ID('dbo.Files'));

-- 1. Struktura raportu
IF OBJECT_ID('tempdb..#FinalReport') IS NOT NULL DROP TABLE #FinalReport;
CREATE TABLE #FinalReport (
    file_name NVARCHAR(255),
    file_type NVARCHAR(10),
    full_name NVARCHAR(255),
    first_name NVARCHAR(100),
    last_name NVARCHAR(100),
    forms_count INT,
    searched_at DATETIME2(3)
);

DECLARE @F NVARCHAR(100), @L NVARCHAR(100), @Cmd NVARCHAR(MAX);

-- 2. Kursor - dodajemy DISTINCT, żeby uniknąć duplikowania par Imię-Nazwisko
DECLARE FinalCursor CURSOR FOR 
SELECT DISTINCT i.Imie, n.Nazwisko
FROM dbo.v_RankingOsob i
CROSS JOIN dbo.v_RankingNazwisk n
WHERE i.Imie IN (SELECT display_term FROM #DostepneSlowa)
  AND n.Nazwisko IN (SELECT display_term FROM #DostepneSlowa);

OPEN FinalCursor;
FETCH NEXT FROM FinalCursor INTO @F, @L;

WHILE @@FETCH_STATUS = 0
BEGIN
    SET @Cmd = '
    INSERT INTO #FinalReport (file_name, file_type, full_name, first_name, last_name, forms_count, searched_at)
    SELECT f.name, f.file_type, ''' + @F + ' ' + @L + ''', ''' + @F + ''', ''' + @L + ''', ct.[RANK], SYSDATETIME()
    FROM dbo.Files f
    INNER JOIN CONTAINSTABLE(dbo.Files, file_stream, ''NEAR(("' + @F + '", "' + @L + '"), 1, TRUE)'') AS ct
    ON f.path_locator = ct.[KEY]';
    
    EXEC sp_executesql @Cmd;
    FETCH NEXT FROM FinalCursor INTO @F, @L;
END;

CLOSE FinalCursor;
DEALLOCATE FinalCursor;

-- 3. Czysty wynik z grupowaniem duplikatów
SELECT 
    r.file_name, 
    MAX(r.file_type) AS file_type,
    UPPER(r.first_name + ' ' + r.last_name) AS full_name,
    r.first_name, 
    r.last_name, 
    (SELECT STRING_AGG(display_term, ', ') 
     FROM (SELECT DISTINCT display_term FROM #DostepneSlowa) AS ds
     WHERE ds.display_term LIKE LOWER(r.first_name) + '%' 
        OR ds.display_term LIKE LOWER(r.last_name) + '%') AS forms,
    MAX(r.forms_count) AS max_rank, 
    MIN(r.searched_at) AS searched_at 
FROM #FinalReport r
GROUP BY r.file_name, r.first_name, r.last_name
ORDER BY r.file_name, max_rank DESC;