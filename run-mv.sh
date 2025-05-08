dataset=users
lookback=60
horizon=20

export CUDA_VISIBLE_DEVICES=0
for model in patchtst 
do
python -u exp.py \
    --model $model \
    --dataset $dataset\
    --seq_len $lookback \
    --pred_len $horizon \
    --univariate False\
    --calculate_shap True\
    --save_train_info False\
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
    --epochs 10\
    --patience_scheduler 16\
    --batch_size 40\
    --background_samples 20\
    --epochs 50\
    --shap_batch_size 2500 >logs/$dataset'_'$model'_tsf.log' 
done