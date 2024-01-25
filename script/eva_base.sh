OMP_NUM_THREADS=1 python  main_finetune.py  \
    --model convnextv2_base  \
    --batch_size 128 --update_freq 1 \
    --blr 8e-4 \
    --epochs 6 \
    --warmup_epochs 1 \
    --layer_decay_type 'single' \
    --layer_decay 1. \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --model_ema True --model_ema_eval True \
    --use_amp False \
    --mixup 0. \
    --num_view 2 \
    --max_len 25 \
    --nb_classes 71 \
    --smoothing 0. \
    --strides 4_4__2_1__2_1__1_1 \
    --input_h 64 \
    --input_w 256 \
    \
    --loss_weight_binary 1. \
    --binary_loss_type BanlanceMultiClassCrossEntropyLoss \
    --voc_type LOWERCASE \
    \
    --evaluation True \
    --dataset_dir ./ \
    --remote_folders test_data/test \
    --data_path  test_data/IIIT5k/  \
    --eval_data_path  test_data/IIIT5k/ \
    --font_path ./dataset/arial.ttf \
    --evaluation True \
    --output_dir outputs \
    --resume /home/kas/byang/convnextv2_ocr/checkpoint-9_7000.pth \
    \
    --use_more_unet \
    --decoder_type small_tf_decoder \
    --mid_size False \