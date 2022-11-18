#!/bin/bash

python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate invert --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate invert blur --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate invert blur solarize --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate invert blur solarize grayscale --knn-monitor --knn-int 5
python train.py --submit --net resnet18 --gpu --data /home/gridsan/shan/MAML-Soljacic_shared --max-num-tf-combos 1 --tfs crop hflip vflip rotate invert blur solarize grayscale colorjitter --knn-monitor --knn-int 5