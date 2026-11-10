@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYTHONPATH=%~dp0src"
py -3.12 -m tgbot.main
if errorlevel 1 (
    echo Ошибка запуска. Проверьте Python 3.12, зависимости и локальный файл .env.
    pause
)
