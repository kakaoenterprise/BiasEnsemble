##### LfF + BE
python train.py --train_lff_be --dataset=bar --percent=1pct --lr=0.00001 --resnet_pretrained --exp=lff_be_bar_1pct --tensorboard
python train.py --train_lff_be --dataset=bar --percent=5pct --lr=0.00001 --resnet_pretrained --exp=lff_be_bar_5pct --tensorboard

##### DisEnt + BE
python train.py --train_disent_be --dataset=bar --percent=1pct --lr=0.00001 --resnet_pretrained --exp=disent_be_bar_1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard
python train.py --train_disent_be --dataset=bar --percent=5pct --lr=0.00001 --resnet_pretrained --exp=disent_be_bar_5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard