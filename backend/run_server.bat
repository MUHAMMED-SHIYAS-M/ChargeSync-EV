@echo off
echo Starting EV ChargeSync API with Python 3.11...
.venv311\Scripts\python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001
pause

