@echo off
REM Build script for Windows .exe

echo Building cubiomes DLL...
cd cubiomes
if not exist libcubiomes.dll (
    echo libcubiomes.dll not found. Please ensure it is built.
    pause
    exit /b 1
)
cd ..

echo Installing dependencies...
pip install pyinstaller
poetry install

echo Building executable with PyInstaller...
pyinstaller --onefile --hidden-import=numba --add-data "cubiomes/libcubiomes.dll;." main.py

echo Build complete. Check dist\main.exe
pause