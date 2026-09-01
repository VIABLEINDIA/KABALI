@echo off
REM Live order-path probe. Places TWO real orders when authorised.
REM
REM Refuses harmlessly unless state\SMOKE_TEST.json exists with the exact
REM confirmation phrase, so scheduling this is safe: without that file it logs a
REM refusal and exits.

setlocal
set "REPO=D:\KABALI"
cd /d "%REPO%" || exit /b 1

for /f "tokens=1-3 delims=/- " %%a in ("%date%") do set "STAMP=%%c%%b%%a"
set "LOG=%REPO%\runs\smoke_%STAMP%.log"

echo ============================================================ >> "%LOG%"
echo smoke probe started %date% %time% >> "%LOG%"
C:\Python314\python.exe "%REPO%\scripts\live_smoke.py" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo smoke probe exited with %RC% at %time% >> "%LOG%"

REM Exit code 1 means the position did not verify flat. Leave a file whose name
REM is impossible to miss, because nobody reads a scheduled task's log.
if "%RC%"=="1" (
  echo POSITION NOT FLAT after the smoke probe - close it manually. See %LOG% > "%REPO%\runs\SMOKE_POSITION_NOT_FLAT.txt"
)
endlocal
