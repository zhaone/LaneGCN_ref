debug() {
  horovodrun -np 4 python cli.py \
    --mixed_precision \
    --epochs 36 \
    --lr 0.001 \
    --lr_decay 0.90 \
    --min_decay 0.1 \
    --save_checkpoints_steps 100 \
    --print_steps 10 \
    --board_steps 50 \
    --eval_steps 1600 \
    --is_dist \
    --optimizer "adam" \
    --pin_memory \
    --name "train_exp" \
    --model "lanegcn_ori" \
    --hparams_path "hparams/lanegcn_ori.json" \
    --num_workers 0 \
    --data_name "lanegcn" \
    --data_version "debug" \
    --mode "train_eval" \
    --save_path "/workspace/expout/lanegcn/debug" \
    --batch_size 32 \
    --reload "latest"
}
train() {
  horovodrun -np 4 python cli.py \
    --mixed_precision \
    --epochs 36 \
    --lr 0.001 \
    --lr_decay 0.90 \
    --min_decay 0.1 \
    --save_checkpoints_steps 100 \
    --print_steps 10 \
    --board_steps 50 \
    --eval_steps 1600 \
    --is_dist \
    --optimizer "adam" \
    --pin_memory \
    --name "val_exp" \
    --model "lanegcn_ori" \
    --hparams_path "hparams/lanegcn_ori.json" \
    --num_workers 0 \
    --data_name "lanegcn" \
    --data_version "full" \
    --mode "train_eval" \
    --save_path "/workspace/expout/lanegcn/train" \
    --batch_size 32 \
    --reload "latest"
}
val() {
  horovodrun -np 4 python cli.py \
    --mixed_precision \
    --epochs 36 \
    --lr 0.001 \
    --lr_decay 0.90 \
    --min_decay 0.1 \
    --save_checkpoints_steps 100 \
    --print_steps 10 \
    --board_steps 50 \
    --eval_steps 1600 \
    --is_dist \
    --optimizer "adam" \
    --pin_memory \
    --name "val_exp" \
    --model "lanegcn_ori" \
    --hparams_path "hparams/lanegcn_ori.json" \
    --num_workers 0 \
    --data_name "lanegcn" \
    --data_version "full" \
    --mode "eval" \
    --save_path "/workspace/expout/lanegcn/val" \
    --batch_size 32 \
    --reload "latest" \
    --enable_hook
}
_test() {
  horovodrun -np 4 python cli.py \
    --mixed_precision \
    --epochs 36 \
    --lr 0.001 \
    --lr_decay 0.90 \
    --min_decay 0.1 \
    --save_checkpoints_steps 100 \
    --print_steps 10 \
    --board_steps 50 \
    --eval_steps 1600 \
    --is_dist \
    --optimizer "adam" \
    --pin_memory \
    --name "test_exp" \
    --model "lanegcn_ori" \
    --hparams_path "hparams/lanegcn_ori.json" \
    --num_workers 0 \
    --data_name "lanegcn" \
    --data_version "full" \
    --mode "test" \
    --save_path "/workspace/expout/lanegcn/test" \
    --batch_size 32 \
    --reload "latest" \
    --enable_hook
}