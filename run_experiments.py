import os
num_seeds = 3
for framestack in [1, 2, 3]:
    for data_augs in ["translate", "no_aug", "crop"]:
        for discount in [.99, .8]:
            for lr in [2e-4, 1e-3]:
                if data_augs == 'crop':
                    image_size = 84
                elif data_augs == 'translate':
                    image_size = 108
                else:
                    image_size = 100
                os.system(
                    "python kitchen_train.py \
                    --encoder_type pixel --work_dir data/kitchen_sc/ \
                    --action_repeat 1 --num_eval_episodes 5 \
                    --pre_transform_image_size 100 --image_size {image_size} \
                    --data_augs {data_augs} --discount {discount} --init_steps 2500 \
                    --agent rad_sac --frame_stack {framestack} \
                    --seed -1 --critic_lr {lr} --actor_lr {lr} --encoder_lr {lr} --eval_freq 1000 --batch_size 512 --num_train_steps 25000".format(
                        data_augs=data_augs, framestack=framestack, discount=discount, image_size=image_size, lr=lr,
                    )
                )
