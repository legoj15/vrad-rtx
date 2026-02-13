@echo off
echo ========================================
echo  Cleaning vrad.sln (Release x64)
echo ========================================
msbuild "%~dp0vrad.sln" /t:Clean /p:Configuration=Release /p:Platform=x64
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo CLEAN FAILED with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo ========================================
echo  Rebuilding vrad.sln (Release x64)
echo ========================================
msbuild "%~dp0vrad.sln" /t:Build /p:Configuration=Release /p:Platform=x64 /m
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo BUILD FAILED with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo.
echo REBUILD SUCCEEDED
