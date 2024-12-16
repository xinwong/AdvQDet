# custom config
DATA=imagenet       # imagenet, cifar, flowers-102, oxford-iiit-pet
DETECT=AdvQDet      # AdvQDet, blacklight, PIHA
ATTACK=hsja         # boundary, hsja, nesscore, qeba, square, surfree, zoo
TARGET=untargeted   # untargeted
S=standard          # adaptive or standard

python main.py \
--config ./configs/${DATA}/${DETECT}/${ATTACK}/${TARGET}/${S}/config.json \
--start_idx 0 \
--num_images 100 \
