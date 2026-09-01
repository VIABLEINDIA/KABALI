@echo off
REM Forward paper session -- one trading day on live data, no real orders.
REM
REM Wrapper for Task Scheduler. Exists because scheduled tasks start in
REM %SystemRoot%, not the project, and daily.py resolves its config and state
REM relative to the repo root.
REM
REM Paper by default. Adding --live here would only REQUEST live trading; the
REM gate is re-evaluated inside and refuses unless the evidence supports it.

setlocal
set "REPO=D:\KABALI"
cd /d "%REPO%" || exit /b 1

for /f "tokens=1-3 delims=/- " %%a in ("%date%") do set "STAMP=%%c%%b%%a"
set "LOG=%REPO%\runs\forward_%STAMP%.log"

echo ============================================================ >> "%LOG%"
echo forward session started %date% %time% >> "%LOG%"
C:\Python314\python.exe "%REPO%\scripts\daily.py" >> "%LOG%" 2>&1
echo forward session exited with %ERRORLEVEL% at %time% >> "%LOG%"

REM Summarise into runs\LATEST_SESSION.txt and raise a desktop toast. A
REM scheduled task finishes into an empty room; the result is four screens up
REM from the end of a log nobody has open.
C:\Python314\python.exe "%REPO%\scripts\session_summary.py" --notify >> "%LOG%" 2>&1
endlocal
