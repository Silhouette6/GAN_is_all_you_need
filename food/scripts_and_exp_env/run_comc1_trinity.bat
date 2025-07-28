@echo off
set LOGFILE=comc1food_trinity_log.txt
echo. > %LOGFILE%

REM Baseline 模型
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0 >> %LOGFILE% 2>&1
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0.1 >> %LOGFILE% 2>&1
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0.2 >> %LOGFILE% 2>&1
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0.3 >> %LOGFILE% 2>&1
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0.4 >> %LOGFILE% 2>&1
python Comc1_food_trinity.py --model_path "models/com_c1.pth" --running_mod "eval" --plot_only "False" --erase_ratio 0.5 >> %LOGFILE% 2>&1

echo Done! All evaluations have been completed.
pause