@echo off
REM Run the already-built appOrange (Release) without rebuilding.
REM Usage: runwithoutbuild.bat [--vulkan]  (any args are forwarded to appOrange)
setlocal
set ROOT=%~dp0
if %ROOT:~-1%==\ set ROOT=%ROOT:~0,-1%
set EXE=%ROOT%\build\bin\Release\appOrange.exe

if not exist "%EXE%" (
    echo appOrange.exe not found at "%EXE%".
    echo Build it first with run.bat.
    exit /b 1
)

"%EXE%" %*
