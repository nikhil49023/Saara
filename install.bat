@echo off
echo ============================================
echo   🧠 Saara CLI Installation Script
echo ============================================
echo.

:: Get the Python user scripts path
for /f "tokens=*" %%i in ('python -c "import sysconfig; print(sysconfig.get_path('scripts', 'nt_user'))"') do set SCRIPTS_PATH=%%i

echo 📂 Python Scripts Path: %SCRIPTS_PATH%
echo.

:: Install the package
echo 📦 Installing Saara package...
pip install -e .
echo.

:: Check if path is already in user PATH
echo %PATH% | findstr /I "%SCRIPTS_PATH%" >nul
if %ERRORLEVEL% == 0 (
    echo ✅ Scripts path is already in PATH.
) else (
    echo 🔧 Adding Scripts path to user PATH...
    setx PATH "%PATH%;%SCRIPTS_PATH%"
    echo.
    echo ⚠️  PATH has been updated. Please RESTART your terminal for changes to take effect.
)

echo.
echo ============================================
echo   Installation Complete!
echo ============================================
echo.
echo After restarting your terminal, you can run:
echo   saara --help
echo   saara run
echo.
pause
