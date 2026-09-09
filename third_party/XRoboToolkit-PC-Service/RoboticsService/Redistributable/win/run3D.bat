pushd %~dp0

start RoboticsServiceProcess.exe

pushd SDKDemo\RobotUnityDemo
start PXREAClientUnity.exe
popd

popd


