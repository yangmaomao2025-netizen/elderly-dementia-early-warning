@echo off
chcp 65001 >nul
echo ========================================
echo   批量热力图生成器
echo ========================================
echo.

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到 Python
    pause
    exit /b 1
)

REM 运行
python heatmap_generator.py %*
pause
