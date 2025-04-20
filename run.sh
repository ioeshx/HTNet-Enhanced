export CUDA_VISIBLE_DEVICES=1

current_time=$(date +"%m-%d_%H-%M")

stdbuf -oL nohup python main_HTNet.py \
--train True > logs/output_${current_time}.log