#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'different_corruptions_cover_dependent_adv_loss_1e-2' \
  --noise 'diff_corruptions' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 48 \
  --cover-dependent 1 \
  --adv_loss 1e-2 \
  --save-dir 'runs_adv_loss/1e-2'