python3 -m torch.distributed.launch --nproc_per_node=1 \
train.py \
--do_train --num_thread_reader=0 \
--epochs=30 --batch_size=8 \
--n_display=10 \
--data_path /amax/cmy/train_valid_test.pt \
--output_dir /amax/cmy/ckpt_align \
--lr 5e-5  \
--visual_num_hidden_layers 4 \
--bert_num_hidden_layers 6 \
--audio_num_hidden_layers 4 
