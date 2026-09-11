@echo off
REM Double-click this to run Firewatch on Windows.
REM Assumes: conda env create -f environment.yml  has already been done.

call conda activate firewatch
if errorlevel 1 (
    echo.
    echo Could not activate the 'firewatch' conda environment.
    echo Run this first, from the Anaconda Prompt:
    echo     conda env create -f environment.yml
    echo.
    pause
    exit /b 1
)

python main.py %*
pause
