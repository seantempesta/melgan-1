import os
import glob
import tqdm
import torch
import argparse
from scipy.io.wavfile import write
import numpy as np
from model.generator import Generator
from utils.hparams import HParam, load_hparam_str
from utils.pqmf import PQMF
from denoiser import Denoiser

MAX_WAV_VALUE = 32768.0

def init(config, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if config is not None:
        hp = HParam(config)
    else:
        hp = load_hparam_str(checkpoint['hp_str'])

    model = Generator(hp.audio.n_mel_channels, hp.model.n_residual_layers,
                        ratios=hp.model.generator_ratio, mult = hp.model.mult,
                        out_band = hp.model.out_channels).cuda()
    model.load_state_dict(checkpoint['model_g'])
    model.eval(inference=True)
    return hp, model


def predict(hp, model, mel, denoise=True):
    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        audio = model.inference(mel)
        # For multi-band inference
        if hp.model.out_channels > 1:
            pqmf = PQMF()
            audio = pqmf.synthesis(audio).view(-1)
        audio = audio.squeeze(0)  # collapse all dimension except time axis
        if denoise:
            denoiser = Denoiser(model).cuda()
            audio = denoiser(audio, 0.1)
        audio = audio.squeeze()
        audio = audio[:-(hp.audio.hop_length*10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE-1)
        audio = audio.short()
        audio = audio.cpu().detach().numpy()
        return audio
