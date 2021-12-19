import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import pandas
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from model import AutoSusPed, AutoSusPed_deeper
from dataset import MAESTRO_small, allocate_batch
from evaluate import evaluate
from constants import HOP_SIZE
import matplotlib.pyplot as plt
from matplotlib import colors
import glob
import torch.backends.cudnn

set_randomseed = True
if set_randomseed:
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    # torch.cuda.manual_seed_all(1) # Activate this line when you use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(1)
    torch.backends.cudnn.enabled = False

def cycle(iterable):
    while True:
        for item in iterable:
            yield item


def train(model_type,
          model_composer,
          target_composer,
          n_test_figs,
          mnum,
          experiment_file,
          sequence_length,
          cnn_unit,
          fc_unit,
          level):


    if sequence_length % HOP_SIZE != 0:
        adj_length = sequence_length // HOP_SIZE * HOP_SIZE
        print(
            f'sequence_length {sequence_length} is not a multiple of {HOP_SIZE}.'
        )
        print(f'Adjusted to: {adj_length}')
        sequence_length = adj_length

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'baseline':
        model = AutoSusPed(cnn_unit=cnn_unit, fc_unit=fc_unit, mnum = mnum, level = level)

    elif model_type == 'deeper':
        model = AutoSusPed_deeper(cnn_unit=cnn_unit, fc_unit=fc_unit, mnum=mnum, level = level)

    else:
        assert False, "this model_type is not supported - Wonjun Yi"

    model_path = glob.glob(experiment_file + '/' + model_composer + '/' + model_type + '/bestmodel-*')[0]

    model = torch.load(model_path)

    model = model.to(device)

    test_dataset = MAESTRO_small(groups=['test'],
                                 composer=target_composer,
                                 hop_size=HOP_SIZE,
                                 random_sample=False,
                                 level = level)
    model.eval()
    with torch.no_grad():
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)

        sw = 0

        for batch in loader:



            batch_results = evaluate(model,
                                     batch,
                                     device,
                                     model_type, level)


            if model_type == 'baseline' or model_type == 'deeper':


                pred = model(batch['frame'])


                predt = pred
                for ipx in range(pred.shape[1]):
                    ipred = pred[:,ipx, :]
                    predt[:,ipx, :] = ipred == ipred.max()
                pred = predt
                pedal_ccpedal = batch['ccpedal']


            # visualize

            if sw<n_test_figs:

                test_plt = plt.figure(constrained_layout = True)

                cmap = colors.ListedColormap(['white', 'black'])



                plt.subplot(131)
                plt.title('ground truth')
                if level == 2:
                    plt.xlabel('on/off')
                else:
                    plt.xlabel('level')
                plt.ylabel('frame')
                plt.imshow(pedal_ccpedal.squeeze().cpu(), aspect='auto', origin='lower', cmap = cmap)
                plt.colorbar()
                plt.clim(0,1)
                plt.subplot(132)
                plt.title('prediction')
                if level == 2:
                    plt.xlabel('on/off')
                else:
                    plt.xlabel('level')
                plt.ylabel('frame')
                plt.imshow(pred.squeeze().cpu(), aspect='auto', origin='lower', cmap = cmap)
                plt.colorbar()
                plt.clim(0,1)
                plt.subplot(133)
                plt.title('error map')
                if level == 2:
                    plt.xlabel('on/off')
                else:
                    plt.xlabel('level')
                plt.ylabel('frame')
                plt.imshow((pedal_ccpedal.squeeze().cpu() != pred.squeeze().cpu()), aspect='auto', origin='lower', cmap = cmap)
                plt.colorbar()
                plt.clim(0,1)

                os.makedirs("./Target_"+target_composer+"/"+model_composer+"/"+model_type+"/test_pics",exist_ok=True)


                test_plt.savefig("./Target_"+target_composer+"/"+model_composer+"/"+model_type+"/test_pics/"+batch['path'][0]+'_accuracy'+str(batch_results['metric/pedal/pedal_acc'][0])+'.jpg', dpi = 5000 * (level//2))

                test_plt.savefig("./Target_"+target_composer+"/"+model_composer+"/"+model_type+"/test_pics/"+batch['path'][0]+'_accuracy'+str(batch_results['metric/pedal/pedal_acc'][0])+'.svg', dpi = 5000 * (level//2))

                plt.cla()
                sw = sw + 1

            for key, value in batch_results.items():
                metrics[key].extend(value)
            metrics['path'].extend(batch['path'])
    print('')

    metric_panda = pandas.DataFrame(metrics)
    metric_panda.to_csv("./Target_"+target_composer+"/"+model_composer+"/"+model_type+"/test.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='baseline', type=str)
    parser.add_argument('--level', default=2, type=int)
    parser.add_argument('--experiment_file', default='experiment_20210101', type=str)
    parser.add_argument('--model_composer', default='Frédéric_Chopin', type=str)
    parser.add_argument('--target_composer', default='Frédéric_Chopin', type=str)
    parser.add_argument('--n_test_figs', default=5, type=int)
    parser.add_argument('--mnum', default=0, type=int)
    parser.add_argument('-v', '--sequence_length', default=102400, type=int)
    parser.add_argument('-cnn', '--cnn_unit', default=48, type=int)
    parser.add_argument('-fc', '--fc_unit', default=256, type=int)
    args = parser.parse_args()

    train(**vars(parser.parse_args()))