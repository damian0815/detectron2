@echo off
call :treeProcess
goto :eof

:treeProcess
rem Do whatever you want here over the files of this subdir, for example:
for %%i in ("z:transcoded\*.mp4" "z:transcoded\*.mov") do (
  if not exist "%%i.detectron2.json" (
    python apply_net.py write_rects -v configs/cse/densepose_rcnn_R_50_FPN_s1x.yaml workspace/models/densepose_rcnn_R_50_FPN_s1x-CSE-c4ea5f.pkl "%%i" cse --is_video --rects_config workspace/detection_rects_config.yml --output "%%i.detectron2.json"
  )
)
for /D %%d in (z:*) do (
    z:
    echo entering "%%d"
    cd %%d
    c:
    call :treeProcess
    z:
    cd ..
    c:
)
exit /b
