@echo off
REM Force llama-server URLs for this session (overrides a stale WOUNDWATCH_VISION_URL=11434 in Windows).
REM Start C:\woundmodel\run.bat or start_llama_vision_server.bat first, wait until llama-server is ready.

set "WOUNDWATCH_VISION_URL=http://127.0.0.1:11435/v1/chat/completions"
set "WOUNDWATCH_MODEL=gpt-3.5-turbo"

cd /d "%~dp0"
py -3 final_pipeline.py --serve --port 5050
pause
