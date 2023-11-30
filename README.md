gdown 18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip 

python3 run_nerf.py --config configs/ship.txt --i_video=100000 --i_test=100000

python3 run_nerf.py --config configs/chair.txt --i_video=100001 --i_test=100001 --convert=1 --rounds=10
