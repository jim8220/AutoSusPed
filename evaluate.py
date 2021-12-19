from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile
import torch
import torch.nn as nn
from mido import Message, MidiFile, MidiTrack
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import hz_to_midi, midi_to_hz

import torch.nn.functional as F

from constants import HOP_SIZE, MIN_MIDI, SAMPLE_RATE
from dataset import allocate_batch


def evaluate(model, batch, device, model_type, level):



    metrics = defaultdict(list)
    batch = allocate_batch(batch, device)



    pedal_logit = model(batch['frame'])
    pedal_loss = -torch.log(pedal_logit) * batch['ccpedal']
    pedal_loss = pedal_loss.mean()


    metrics['metric/loss/pedal_loss'].append(pedal_loss.cpu().numpy())


    for batch_idx in range(batch['frame'].shape[0]):
        pedal_pred = pedal_logit[batch_idx]
        acc = pedal_eval(pedal_pred, batch['ccpedal'][batch_idx], level)
        metrics['metric/pedal/pedal_acc'].append(acc)

    return metrics

def pedal_eval(pred, label, level = 2):

    predt = pred
    for ipx in range(pred.shape[0]):
        ipred = pred[ipx,:]
        predt[ipx,:] = ipred == ipred.max()
        if (ipred == ipred.max()).sum() > 1:
            print("there are more than 2 values here!")
    labelt = label == 1

    tp_ = (predt[:, 0] == labelt[:, 0])

    for ilx in range(1,level):

        tp_ = tp_ * (predt[:,ilx] == labelt[:,ilx])

    tp = torch.sum(tp_).cpu().numpy()
    
    acc = tp / len(predt[:,0])

    return acc
