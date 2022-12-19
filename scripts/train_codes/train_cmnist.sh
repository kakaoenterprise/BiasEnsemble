### LfF + BE
python train.py --train_lff_be --dataset=cmnist --percent=0.5pct --lr=0.01 --exp=lff_be_cmnist_0.5pct --tensorboard
python train.py --train_lff_be --dataset=cmnist --percent=1pct --lr=0.01 --exp=lff_be_cmnist_1pct --tensorboard
python train.py --train_lff_be --dataset=cmnist --percent=2pct --lr=0.01 --exp=lff_be_cmnist_2pct --tensorboard
python train.py --train_lff_be --dataset=cmnist --percent=5pct --lr=0.01 --exp=lff_be_cmnist_5pct --tensorboard

### DisEnt + BE
python train.py --train_disent_be --dataset=cmnist --percent=0.5pct --lr=0.01 --exp=disent_be_cmnist_0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --tensorboard
python train.py --train_disent_be --dataset=cmnist --percent=1pct --lr=0.01 --exp=disent_be_cmnist_1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --tensorboard
python train.py --train_disent_be --dataset=cmnist --percent=2pct --lr=0.01 --exp=disent_be_cmnist_2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --tensorboard
python train.py --train_disent_be --dataset=cmnist --percent=5pct --lr=0.01 --exp=disent_be_cmnist_5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --tensorboard