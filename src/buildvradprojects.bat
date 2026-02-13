@echo off
echo ========================================
echo  Building vrad.sln (Release x64)
echo ========================================
msbuild "%~dp0vrad.sln" /p:Configuration=Release /p:Platform=x64 /m
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo BUILD SUCCEEDED
