CUDA_VISIBLE_DEVICES=0  python3 trainval_net.py --dataset pascal_voc --net vgg16 \
   --bs 1 --nw 6 --save_dir='results/test_single'  \
   --lr 0.001 --cuda --disp_interval 50 --vis \
	--checkpoint_interval=500 --model='oicr' --threshold=0.1




#CUDA_VISIBLE_DEVICES=1,2,3  python3 trainval_net.py --dataset pascal_voc --net vgg16 \
#   --bs 3 --nw 6 --save_dir='results/test_mult'  \
#   --lr 0.001 --cuda --disp_interval 50 --vis --mGPUs \
#	--checkpoint_interval=500 --model='oicr' --threshold=0.1
