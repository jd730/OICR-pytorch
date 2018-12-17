CUDA_VISIBLE_DEVICES=0 python3 test_oicr.py --dataset pascal_voc --net vgg16  --checkpoint 00500 --load_dir=results/test_single --cuda --output_dir="test" --model="oicr" --vis
