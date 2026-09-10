@echo off
REM 从 AutoDL 拉取训练权重到本地（Windows）
REM 用法：双击运行，或命令行传参
REM   pull_weights.bat <端口> <实例地址>
REM 例：pull_weights.bat 12345 connect.autodl.com

setlocal

if "%~1"=="" (
    echo 用法: pull_weights.bat ^<SSH端口^> ^<实例地址^>
    echo 例:   pull_weights.bat 12345 connect.autodl.com
    exit /b 1
)

set PORT=%~1
set HOST=%~2

echo 正在从 %HOST%:%PORT% 拉取权重到 outputs\ckpts\ ...
scp -P %PORT% root@%HOST%:/root/denoising-dirty-documents/outputs/ckpts/cloudA_v2_seed*.pt outputs\ckpts\
scp -P %PORT% root@%HOST%:/root/denoising-dirty-documents/outputs/ckpts/cloudB_v2_seed*.pt outputs\ckpts\

echo 拉取完成，当前 ckpts 目录：
dir outputs\ckpts\cloud*.pt

endlocal
