@echo off
setlocal EnableDelayedExpansion

title BTC Advanced Charting - Auto Runner

cd /d "%~dp0"

:loop
cls
echo ========================================================
echo   BTC AUTO RUNNER - Every 15 Minutes (00, 15, 30, 45)
echo ========================================================
echo.

rem Get current time
for /f "tokens=1-3 delims=:.," %%a in ("%time: =0%") do (
    set "hh=%%a"
    set "mm=%%b"
    set "ss=%%c"
)

rem Calculate minutes
set /a "min_num=1!mm!-100"
set /a "remainder=min_num %% 15"

rem If at 00, 15, 30, 45 minute mark (with some tolerance, but wait handles double run)
if !remainder! equ 0 (
    echo   [!hh!:!mm!:!ss!] === RUNNING ANALYSIS ===
    echo.
    
    python run.py
    
    echo.
    echo   [+] Done! Waiting 65s to avoid double run...
    timeout /t 65 /nobreak >nul
    goto loop
)

rem Calculate time until next run
set /a "wait_min=15-remainder"
title BTC Runner - Next in !wait_min! min

echo   [!hh!:!mm!:!ss!] Next run in !wait_min! min...
echo.
echo   Press Ctrl+C to EXIT

rem Wait 30 seconds then refresh check
timeout /t 30 /nobreak >nul

goto loop
