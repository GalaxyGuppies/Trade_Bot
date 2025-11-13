@echo off
echo Resetting Windows DNS Client Service...
echo.

REM Stop DNS Client Service
net stop dnscache

REM Clear DNS cache
ipconfig /flushdns

REM Start DNS Client Service
net start dnscache

echo.
echo Done! Now test with: python test_dns.py
pause
