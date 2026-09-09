icacls setting.ini /c /t /grant Everyone:F
ie4uinit.exe -ClearIconCache
netsh advfirewall firewall add rule name="picobusinesssuit" dir=in action=allow program="%CUR_DIR%RoboticsServiceProcess.exe" enable=yes profile=any protocol=any localport=any
rd /s /q db