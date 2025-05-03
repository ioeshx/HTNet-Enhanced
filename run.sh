export CUDA_VISIBLE_DEVICES=0

current_time=$(date +"%m-%d_%H-%M")

nohup python main_HTNet.py \
--epochs 400 \
--train True > logs/output_${current_time}_v3.log
