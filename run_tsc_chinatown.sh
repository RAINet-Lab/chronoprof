dataset=chinatown
classes=2
lookback=24 
export CUDA_VISIBLE_DEVICES=1
for model in patchtst linear
do
  if [ "$model" = "patchtst" ]; then
    bs=30000
  else
    bs=80000
  fi
python -u exp_tsc.py \
    --model $model \
    --dataset $dataset\
    --seq_len $lookback\
    --pred_len $classes \
    --calculate_shap True\
    --save_train_info True\
    --max_evals 674\
    --enc_in 1 \
    --e_layers 3 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 2\
    --epochs 100\
    --patience_scheduler 15\
    --batch_size 20\
    --background_samples 20\
    --shap_batch_size $bs >logs/$dataset'_'$model'_tsc.log' 
done