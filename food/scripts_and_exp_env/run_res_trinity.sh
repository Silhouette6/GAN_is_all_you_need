#!/bin/bash
LOGFILE="resfood_trinity_log.txt"
echo "" > "$LOGFILE"
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only False --erase_ratio 0 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.1 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.2 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.3 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.4 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.5 >> "$LOGFILE" 2>&1

python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only False --erase_ratio 0 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.1 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.2 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.3 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.4 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_finetuned.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.5 >> "$LOGFILE" 2>&1

python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only False --erase_ratio 0 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.1 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.2 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.3 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.4 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_18_pretrain_baseline.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.5 >> "$LOGFILE" 2>&1

python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only False --erase_ratio 0 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.1 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.2 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.3 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.4 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_2.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.5 >> "$LOGFILE" 2>&1

python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only False --erase_ratio 0 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.1 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.2 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.3 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.4 >> "$LOGFILE" 2>&1
python Res18_food_trinity.py --model_path 'models/saved/resnet_mutual_learning_11_r18-vbasepre.pth' --running_mod 'eval' --plot_only True --erase_ratio 0.5 >> "$LOGFILE" 2>&1

