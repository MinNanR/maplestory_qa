@echo off
set "URL=%~1"
if "%URL%"=="" set "URL=about:blank"
start "" "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe" --remote-debugging-port=9333 --remote-allow-origins=* --user-data-dir="D:\code\maplestory_qa\.edge-cdp" --no-first-run --no-default-browser-check "%URL%"
echo.
echo Debug Edge launched (port 9333). Open the page you want me to read in THIS window.
echo You can close this black window now.
pause
