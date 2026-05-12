@echo off
REM Matches C:\woundmodel\run.bat — starts llama.cpp llama-server (vision) on port 11435.
REM FIRSTSIGHT / woundwatch.py default to http://127.0.0.1:11435/v1/chat/completions
REM Edit paths here if your GGUF folder is not C:\woundmodel

start "LlamaServer" llama-server.exe -m C:\woundmodel\gemma-4-e2b-it.Q4_K_M.gguf --mmproj C:\woundmodel\gemma-4-e2b-it.F16-mmproj.gguf --port 11435
echo.
echo When the server is ready, run from this project folder:
echo   py final_pipeline.py --serve
echo.
pause
