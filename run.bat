@echo off
REM Build Orange in Release and run appOrange (OpenGL by default).
REM Usage: run.bat [--vulkan]  (any args are forwarded to appOrange)
setlocal
set ROOT=%~dp0
if %ROOT:~-1%==\ set ROOT=%ROOT:~0,-1%
set BUILD=%ROOT%\build

cmake -S "%ROOT%" -B "%BUILD%" -G "Visual Studio 16 2019" -A x64 || goto :error
cmake --build "%BUILD%" --config Release || goto :error

"%BUILD%\bin\Release\appOrange.exe" %*
goto :eof

:error
echo.
echo Build failed (exit %errorlevel%).
exit /b %errorlevel%
