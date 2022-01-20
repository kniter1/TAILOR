# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
#torch.set_printoptions(profile="full")
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import time

from .module_bert import BertModel, BertConfig, BertOnlyMLMHead
from .until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, Focalloss
from .module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from .module_audio import AudioModel, AudioConfig, AudioOnlyMLMHead
from .module_cross import CrossModel, CrossConfig
from .module_decoder import DecoderModel, DecoderConfig
from .until_module import getBinaryTensor, GradReverse, CTCModule
# from warpctc_pytorch import CTCLoss
import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

class EmotionClassifier(nn.Module):
    def __init__(self, input_dims, num_classes=1, dropout=0.1):
        super(EmotionClassifier, self).__init__()
        self.dense = nn.Linear(input_dims, num_classes)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq_input):

        output = self.dense(seq_input)
        output = self.dropout(output)
        output = self.activation(output)

        return output


class TAILORPreTrainedModel(PreTrainedModel, nn.Module):  
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config,*inputs, **kwargs):
        # utilize bert config as base config
        super(TAILORPreTrainedModel, self).__init__(visual_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.audio_config = audio_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.visual = None
        self.vat_cross = None
        self.audio = None
        self.bert = None
        self.decoder = None

    
    @classmethod
    def from_pretrained(cls, bert_model_name, visual_model_name,  audio_model_name, cross_model_name, decoder_model_name, 
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):


        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

 
        bert_config, _= BertConfig.get_config(bert_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        audio_config, _ = AudioConfig.get_config(audio_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, audio_config, cross_config, decoder_config, *inputs, **kwargs)
        assert model.bert is not None
        assert model.visual is not None
        assert model.audio is not None
        assert model.vat_cross is not None
        assert model.va_cross is not None
        assert model.pc_cross is not None
        assert model.decoder is not None

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)
            
        return model

class NormalizeText(nn.Module):   
    def __init__(self, task_config):
        super(NormalizeText, self).__init__()
        self.text_norm2d = LayerNorm(task_config.text_dim)

    def forward(self, text):
        text = torch.as_tensor(text).float()
        text = text.view(-1, text.shape[-2], text.shape[-1])
        text = self.text_norm2d(text)
        return text  

class NormalizeVideo(nn.Module):   
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, visual):
        visual = torch.as_tensor(visual).float()
        visual = visual.view(-1, visual.shape[-2], visual.shape[-1])
        visual = self.visual_norm2d(visual)
        return visual 

class NormalizeAudio(nn.Module):  
    def __init__(self, task_config):
        super(NormalizeAudio, self).__init__()
        self.audio_norm2d = LayerNorm(task_config.audio_dim)

    def forward(self, audio):
        audio = torch.as_tensor(audio).float()
        audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
        audio = self.audio_norm2d(audio)
        return audio  #输出：[B, L, D]

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config
    
def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]



class TAILOR(TAILORPreTrainedModel): 
    def __init__(self, bert_config, visual_config, audio_config, cross_config, decoder_config, task_config):
        super(TAILOR, self).__init__(bert_config, visual_config, audio_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1
        self.num_classes = task_config.num_classes
        self.aligned = task_config.aligned

        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_sequence <= audio_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        # Text Encoder ====>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "bert_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <==== End of Text Encoder
        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder
        # Audio Encoder ====>
        audio_config = update_attr("audio_config", audio_config, "num_hidden_layers",
                                    self.task_config, "audio_num_hidden_layers")
        self.audio = AudioModel(audio_config)
        audio_word_embedding_weight = self.audio.embeddings.word_embeddings.weight
        # <====End of Audio_Encoder

        # Cross Encoder ===>
        cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
        self.vat_cross = CrossModel(cross_config)
        self.va_cross = CrossModel(cross_config)
        self.pc_cross = CrossModel(cross_config)
     #  self.at_cross = CrossModel(cross_config)
         # <=== End of Cross Encoder
        
        # Label Decoder =====>
        decoder_config = update_attr("decoder_config", decoder_config, "num_decoder_layers",
                                        self.task_config, "decoder_num_hidden_layers")
        self.decoder = DecoderModel(decoder_config)
        # <===========  End of Decoder
        
        self.common_feature_extractor= nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )

        self.common_classfier = nn.Sequential(
            nn.Linear(task_config.hidden_size, self.num_classes),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )

        self.private_feature_extractor = nn.ModuleList([nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size),
            nn.Dropout(p=0.1),
            nn.Tanh()
        ) for _ in range(3)])

        self.modal_discriminator = nn.Sequential(
            nn.Linear(task_config.hidden_size, task_config.hidden_size // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(task_config.hidden_size // 2, 3),
        )

        self.cross_classifier = EmotionClassifier(cross_config.hidden_size, 1) 
        self.text_norm = NormalizeText(task_config)   
        self.visual_norm = NormalizeVideo(task_config)
        self.audio_norm = NormalizeAudio(task_config)
        self.ml_loss = nn.BCELoss()
        self.adv_loss = nn.CrossEntropyLoss()

        if self.aligned == False:
            self.a2t_ctc = CTCModule(task_config.audio_dim, 50)
            self.v2t_ctc = CTCModule(task_config.video_dim, 50)
            self.ctc_criterion = CTCLoss()  
        self.apply(self.init_weights)

    def forward(self, text, text_mask, visual, visual_mask, audio, audio_mask, 
                label_input, label_mask, groundTruth_labels=None, training=True):
        """
        text: [B, L, Dt]
        visual: [B, L, Dv]
        audio: [B, L, Da]
        """

        label_input = label_input.unsqueeze(0)
        batch = text.size(0)
        label_input = label_input.repeat(batch, 1)
        label_mask = label_mask.unsqueeze(0).repeat(batch, 1)
        text = self.text_norm(text)
        visual = self.visual_norm(visual)  
        audio = self.audio_norm(audio)
        # ========> aligned
        if self.aligned == False:
            visual, v2t_position = self.v2t_ctc(visual)
            audio, a2t_position = self.a2t_ctc(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask) #[B, L, D]
        
        # =========> common and private feature extractor
        private_text = self.private_feature_extractor[0](text_output)
        private_visual = self.private_feature_extractor[1](visual_output)
        private_audio = self.private_feature_extractor[2](audio_output)

        common_text = self.common_feature_extractor(text_output)
        common_visual = self.common_feature_extractor(visual_output)
        common_audio = self.common_feature_extractor(audio_output)

        common_feature = common_text + common_visual + common_audio 
        # <========= common and private feature extractor

        common_mask = torch.ones_like(text_mask)
        pooled_output, cross_output, cross_mask = self._get_cross_output(private_text, private_visual, private_audio, common_feature, text_mask, visual_mask, audio_mask, common_mask)

        # ==========> label modal alignment
        decoder_output = self.decoder(label_input, cross_output, label_mask, cross_mask)
        # <========== label modal alignment
        cross_predict_scores = self.cross_classifier(decoder_output)
        cross_predict_scores  = cross_predict_scores.view(-1, self.num_classes)     
        predict_scores = cross_predict_scores
        predict_labels = getBinaryTensor(predict_scores)
        groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)
                                                                  
        if training:
            text_modal = torch.zeros_like(common_mask).view(-1) #[B, L]
            visual_modal = torch.ones_like(common_mask).view(-1) #[B, L]
            audio_modal = visual_modal.data.new(visual_modal.size()).fill_(2) #[B, L]

            private_text_modal_pred = self.modal_discriminator(private_text).view(-1, 3)
            private_visual_modal_pred = self.modal_discriminator(private_visual).view(-1, 3)
            private_audio_modal_pred = self.modal_discriminator(private_audio).view(-1, 3)

            # ==========> adversial Training
            common_text_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_text, 1)).view(-1, 3)
            common_visual_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_visual, 1)).view(-1, 3)
            common_audio_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_audio, 1)).view(-1, 3)
            # <========== adversial Training

            all_loss = 0.
            pooled_common = common_feature[:, 0] #[B, D]
            common_pred = self.common_classfier(pooled_common)
            ml_loss = self.ml_loss(predict_scores, groundTruth_labels)
            cml_loss = self.ml_loss(common_pred, groundTruth_labels)
            preivate_diff_loss = self.calculate_orthogonality_loss(private_text, private_visual) + self.calculate_orthogonality_loss(private_text, private_audio) + self.calculate_orthogonality_loss(private_visual, private_audio)
            common_diff_loss = self.calculate_orthogonality_loss(common_text, private_text) + self.calculate_orthogonality_loss(common_visual, private_visual) + self.calculate_orthogonality_loss(common_audio, private_audio)
            adv_preivate_loss = self.adv_loss(private_text_modal_pred, text_modal) + self.adv_loss(private_visual_modal_pred, visual_modal) + self.adv_loss(private_audio_modal_pred, audio_modal)
            adv_common_loss = self.adv_loss(common_text_modal_pred, text_modal) + self.adv_loss(common_visual_modal_pred, visual_modal) + self.adv_loss(common_audio_modal_pred, audio_modal)

            if self.aligned == False:
                text_length , audio_length, visual_length = text.size(1), audio.size(1), visual.size(1)
                t_position = torch.tensor([i+1 for i in range(text_length)] * batch, device=text.device).int().cpu()
                t_length = torch.tensor([text_length] * batch).int().cpu()
                a_length = torch.tensor([audio_length] * batch).int().cpu()
                v_length = torch.tensor([visual_length] * batch).int().cpu()
                ctc_v2t_loss = self.ctc_criterion(v2t_position.transpose(0, 1).cpu(), t_position, v_length, t_length)
                ctc_a2t_loss = self.ctc_criterion(a2t_position.transpose(0, 1).cpu(), t_position, a_length, t_length)
                ctc_loss = ctc_v2t_loss + ctc_a2t_loss
                ctc_loss = ctc_loss.cuda()
            
            if self.aligned:
                all_loss = ml_loss  + 0.01 * (adv_common_loss + adv_preivate_loss) + 5e-6 * (preivate_diff_loss + common_diff_loss) + 0.5 * cml_loss    
            else:
                all_loss = ml_loss  + 0.01 * (adv_common_loss + adv_preivate_loss) + 5e-6 * (preivate_diff_loss + common_diff_loss) + 0.5 * cml_loss  + 0.5 * ctc_loss

            return  all_loss, predict_labels, groundTruth_labels, predict_scores
        else:
            return predict_labels, groundTruth_labels, predict_scores


    def get_text_visual_audio_output(self, text, text_mask, visual, visual_mask, audio, audio_mask):
        """
        Uni-modal Extractor
        """
        text_layers, text_pooled_output = self.bert(text, text_mask, output_all_encoded_layers=True)
        text_output = text_layers[-1] 

        visual_layers, visual_pooled_output = self.visual(visual, visual_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]  

        audio_layers, audio_pooled_output = self.audio(audio, audio_mask, output_all_encoded_layers=True)
        audio_output = audio_layers[-1] 

        return text_output, visual_output, audio_output


    def _get_cross_output(self, sequence_output, visual_output,  audio_output, common_feature, attention_mask, visual_mask, audio_mask, common_mask):

        # =============> visual audio fusion
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        va_concat_mask = torch.cat((audio_mask, visual_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(visual_mask)
        audio_type_ =  torch.zeros_like(audio_mask)
        va_concat_type = torch.cat((audio_type_, video_type_), dim=1)
        va_cross_layers, va_pooled_output = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)
        va_cross_output = va_cross_layers[-1]
        # <============= visual audio fusion

        # =============> VisualAudio and text fusion
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        va_type_ = torch.ones_like(va_concat_mask)
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)
        vat_cross_layers, vat_pooled_output = self.vat_cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
        # <============= VisualAudio and text fusion

        # =============> private common fusion
        pc_concate_features = torch.cat((vat_cross_output, common_feature), dim=1)
        specific_type = torch.zeros_like(vat_concat_mask)
        common_type = torch.ones_like(common_mask)
        pc_concate_type = torch.cat((specific_type, common_type), dim=1)
        pc_concat_mask = torch.cat((vat_concat_mask, common_mask), dim=1)
        pc_cross_layers, pc_pooled_output = self.pc_cross(pc_concate_features, pc_concate_type, pc_concat_mask)
        pc_cross_output = pc_cross_layers[-1]
        # <============= private common fusion
 
        return  pc_pooled_output, pc_cross_output, pc_concat_mask
    

    def inference(self, text, text_mask, visual, visual_mask, audio, audio_mask, \
                label_input, label_mask, groundTruth_labels=None):
        label_input = label_input.unsqueeze(0)
        batch = text.size(0)
        label_input = label_input.repeat(batch, 1)
        label_mask = label_mask.unsqueeze(0).repeat(batch, 1)
        text = self.text_norm(text)
        visual = self.visual_norm(visual)   
        audio = self.audio_norm(audio)
        if self.aligned == False:
            visual, _ = self.v2t_ctc(visual)
            audio, _ = self.a2t_ctc(audio)
        text_output, visual_output, audio_output = self.get_text_visual_audio_output(text, text_mask, visual, visual_mask, audio, audio_mask)

        private_text = self.private_feature_extractor[0](text_output)
        private_visual = self.private_feature_extractor[1](visual_output)
        private_audio = self.private_feature_extractor[2](audio_output)

        common_text = self.common_feature_extractor(text_output)
        common_visual = self.common_feature_extractor(visual_output)
        common_audio = self.common_feature_extractor(audio_output)

        common_feature = (common_text + common_visual + common_audio) #[B, L, D]
        preivate_feature = private_text + private_visual + private_audio 
        pooled_common = common_feature[:, 0] #[B, D]
        pooled_preivate = preivate_feature[:, 0]
        common_pred = self.common_classfier(pooled_common)
        preivate_pred = self.common_classfier(pooled_preivate)
        common_mask = torch.ones_like(text_mask)


        pooled_output, cross_output, cross_mask = self._get_cross_output(private_text, private_visual, private_audio, common_feature, text_mask, visual_mask, audio_mask, common_mask)
        decoder_output = self.decoder(label_input, cross_output, label_mask, cross_mask)
        cross_predict_scores = self.cross_classifier(decoder_output)
        cross_predict_scores  = cross_predict_scores.view(-1, self.num_classes)     
        predict_scores = cross_predict_scores
        predict_labels = getBinaryTensor(predict_scores)
        groundTruth_labels = groundTruth_labels.view(-1, self.num_classes)
        

        return predict_labels, groundTruth_labels


    def calculate_orthogonality_loss(self, first_feature, second_feature):
        diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(1, 2)), dim=(1, 2)).pow(2).mean()
        return diff_loss




    

