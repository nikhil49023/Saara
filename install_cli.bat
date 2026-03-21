@echo off
echo 🧠 Installing Saara CLI Environment...
echo.

:: Install the package in editable mode
pip install -e .

echo.
echo ===================================================
echo  ✅ Installation Complete!
echo ===================================================
echo.
echo You can now use the 'saara' command from anywhere if your Python Scripts folder is in PATH.
echo.
echo Example Commands:
echo   saara run           (Start Interactive Wizard)
echo   saara --help        (Show Help)
echo.
echo If 'saara' command is not found, use the local shortcut:
echo   .\saara.bat run
echo.
pause
