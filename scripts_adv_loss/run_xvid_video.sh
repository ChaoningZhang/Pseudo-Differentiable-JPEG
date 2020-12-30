#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'xvid_cover_dependent_video_trained' \
  --noise 'xvid' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 32 \
  --adv_loss 1e-2 \
  --video_dataset 1 \
  --save-dir 'runs_adv_loss/1e-2'