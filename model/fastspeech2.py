import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor
from utils.tools import get_mask_from_lengths


class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        # Phoneme Encoder
        self.encoder = Encoder(model_config)
        
        # Predict the epoch duration
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        
        # Decoder to generate mel, phase and epoch length
        self.decoder = Decoder(model_config)

        # Convert the decoder output into mel
        self.mel_postnet = PostNet()
        
        # Convert the decoder output into phase
        self.phase_postnet = PostNet()

        self.speaker_emb = None
        
        # For ljspeech, this is False
        if model_config["multi_speaker"]:
            with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json"), "r",
            ) as f:
                n_speaker = len(json.load(f))
                    
            self.speaker_emb = nn.Embedding(n_speaker, model_config["transformer"]["encoder_hidden"])

    def forward(
        self,
        speakers, 
        texts,
        text_lens,
        max_text_len,
        mels=None, 
        phases=None, 
        acoustic_lens=None, 
        max_acoustic_len=None,
        epochdurs=None, 
        epochlens=None, 
    ):
        text_masks = get_mask_from_lengths(text_lens, max_text_len)
        acoustic_masks = (
            get_mask_from_lengths(acoustic_lens, max_acoustic_len)
            if acoustic_lens is not None
            else None
        )

        output = self.encoder(texts, text_masks)    # torch.Size([2, 374, 256])

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            text_masks,
            acoustic_masks,
            max_acoustic_len,
            epochdurs
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
        )