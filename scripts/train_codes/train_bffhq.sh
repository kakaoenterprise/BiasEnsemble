##### LfF + BE
python train.py --train_lff_be --dataset=bffhq --percent=0.5pct --lr=0.0001 --exp=lff_be_bffhq_0.5pct --tensorboard
python train.py --train_lff_be --dataset=bffhq --percent=1pct --lr=0.0001 --exp=lff_be_bffhq_1pct --tensorboard
python train.py --train_lff_be --dataset=bffhq --percent=2pct --lr=0.0001 --exp=lff_be_bffhq_2pct --tensorboard
python train.py --train_lff_be --dataset=bffhq --percent=5pct --lr=0.0001 --exp=lff_be_bffhq_5pct --tensorboard

##### DisEnt + BE
python train.py --train_disent_be --dataset=bffhq --percent=0.5pct --lr=0.0001 --exp=disent_be_bffhq_0.5pct --curr_step=10000 --lambda_swap=0.1 --lambda_dis_align=2 --lambda_swap_align=2 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard
python train.py --train_disent_be --dataset=bffhq --percent=1pct --lr=0.0001 --exp=disent_be_bffhq_1pct --curr_step=10000 --lambda_swap=0.1 --lambda_dis_align=2 --lambda_swap_align=2 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard
python train.py --train_disent_be --dataset=bffhq --percent=2pct --lr=0.0001 --exp=disent_be_bffhq_2pct --curr_step=10000 --lambda_swap=0.1 --lambda_dis_align=2 --lambda_swap_align=2 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard
python train.py --train_disent_be --dataset=bffhq --percent=5pct --lr=0.0001 --exp=disent_be_bffhq_5pct --curr_step=10000 --lambda_swap=0.1 --lambda_dis_align=2 --lambda_swap_align=2 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.1 --tensorboard
