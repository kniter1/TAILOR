from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import json
import torch
import numpy as np
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import argparse
from src.models.models import Tagging_UniVL
from src.models.optimization import BertAdam
from torch.utils.data import DataLoader
import torch.utils.data as data
from util import parallel_apply, get_logger
from src.dataloaders.cmu_dataloader import MOSEI_Dataset, MOSEI_Dataset_no_align
from src.utils.eval import get_metrics
#torch.distributed.init_process_group(backend="nccl")

global logger
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dataloader_test(args):
    test_dataset = MOSEI_Dataset(
        args.data_path,
        'test'
    )
    label_input, label_mask = test_dataset._get_label_input()
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=False,
        shuffle=False,
 #       sampler=val_sampler,
        drop_last=True   
    )
    test_length = len(test_dataset)
    return  label_input, label_mask, test_dataloader, test_length

def load_model(args, n_gpu, device, model_file=None): #模型加载
    logger.info("**** loading model_file=%s *****", model_file)

    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        if args.local_rank == 0:
            logger.info("Model loaded from %s", model_file)
        # Prepare model
        model = Tagging_UniVL.from_pretrained(args.bert_model, args.visual_model, args.audio_model, args.cross_model, args.decoder_model, state_dict=model_state_dict,task_config=args)


        model.to(device)
        logger.info('***** loading model successful! *****')
    else:
        model = None
    return model

#save_path = '/home/tione/notebook/test_5k_2nd_cross_embedding'
def model_test(model, test_dataloader, device, label_input, label_mask):
    model.eval()
    label_input = label_input.to(device)
    label_mask = label_mask.to(device)
    with torch.no_grad():
        total_pred = []
        total_true_label = []
        for _, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            text, text_mask, video, video_mask, audio, audio_mask, ground_trunth_labels = batch
            batch_pred, true_label = model.interfence(text, text_mask, video, video_mask, audio, audio_mask, label_input, label_mask, ground_trunth_labels=ground_trunth_labels)
            total_pred.append(batch_pred)
            total_true_label.append(true_label)
        
        total_pred=torch.cat(total_pred,0)
        total_true_label = torch.cat(total_true_label, 0)
    
        return  total_pred, total_true_label

parser = argparse.ArgumentParser(description="model interfence")
parser.add_argument("--do_test", action='store_true', help="whether to run test")
parser.add_argument("--data_path", type=str, help='cmu_mosei data_path')
parser.add_argument("--model_file", type=str, help="model store path")
parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
#parser.add_argument("--epoch", type=int, help="choice nums models")
parser.add_argument('--max_words', type=int, default=60, help='')
parser.add_argument('--max_frames', type=int, default=60, help='')
parser.add_argument('--max_sequence', type=int, default=60, help='')
parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
parser.add_argument('--audio_model', default="audio-base", type=str, required=False, help='AUdio module')
parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
parser.add_argument("--bert_model", default="bert-base", type=str, required=False,
                        help="Bert pre-trained model")
parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
parser.add_argument("--num_labels", type=int, default=6, required=False)
parser.add_argument('--video_dim', type=int, default=35, required=False,help='video feature dimension')
parser.add_argument('--audio_dim', type=int, default=74, required=False, help='')
parser.add_argument('--text_dim', type=int, default=300, help='text_feature_dimension') 
parser.add_argument('--bert_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
parser.add_argument('--visual_num_hidden_layers', type=int, default=4, help="Layer NO. of visual.")
parser.add_argument('--audio_num_hidden_layers', type=int, default=4, help="Layer NO. of audio")
parser.add_argument('--cross_num_hidden_layers', type=int, default=3, help="Layer NO. of cross.")
parser.add_argument('--decoder_num_hidden_layers', type=int, default=1, help="Layer NO. of decoder.")
parser.add_argument("--common_dim",type=int, default=256)
parser.add_argument('--batch_size', type=int, default=64, help='batch size')#训练集bh
parser.add_argument('--seed', type=int, default=42, help='random seed') 
args = parser.parse_args()
n_gpu = 1
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
logger = get_logger(os.path.join(args.output_dir, "log.txt"))

start=time.time()
if args.local_rank ==0:
    model = load_model(args, n_gpu, device, model_file=args.model_file)
    logger.info("***** dataloader loading *****")
    label_input, label_mask, test_dataloader, test_length = dataloader_test(args)
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch size = %d", 64)
    logger.info("  Num steps = %d", len(test_dataloader)) 
    total_pred, total_true_label = model_test(model, test_dataloader, device, label_input, label_mask)
    
    test_micro_f1, test_micro_precision, test_micro_recall, test_acc = get_metrics(total_pred, total_true_label)
    logger.info("----- micro_f1: %f, micro_precision: %f, micro_recall: %f,  acc: %f", \
                    test_micro_f1, test_micro_precision, test_micro_recall, test_acc)
    logger.info("inference time: {}".format(time.time() - start))



# %%

# %%
