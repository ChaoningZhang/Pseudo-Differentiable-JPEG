#### NON-ROBUST DATASET ####
python3 main.py \
  'new' \
  --name 'jpeg_plus_quant_cover_dependent' \
  --noise 'jpeg+quant' \
  --data-dir '/media/user/SSD1TB-1/ImageNet/' \
  --batch-size 32 \
  --cover-dependent 1 \
  --adv_loss 1e-2 \