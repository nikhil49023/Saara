@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Pulling required Ollama models...
echo Pulling Granite 4.0 (Labeling Model)...
ollama pull granite-code:8b
rem Note: Adjust model name if needed, using generic granite-code for now or user specified granite4
ollama pull moondream

echo.
echo Setup complete!
echo Run 'run_pipeline.bat' to start the wizard.
pause
