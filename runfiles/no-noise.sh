nohup python -u main.py new -d /data/coco/10K -e 200 -b 32 --name no-noise & 
sleep 1
tail -f nohup.out
