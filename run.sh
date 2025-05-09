export CUDA_VISIBLE_DEVICES=0

current_time=$(date +"%m-%d_%H-%M")
model_name="HTNet_Enhanced_v6"

nohup python main_HTNet.py \
--epochs 300 \
--gb_tf_channels 64 \
--gb_heads 4 \
--model_type $model_name \
--train True > logs/output_${current_time}_${model_name}.log
