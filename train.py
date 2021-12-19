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
import torch.nn.functional as F
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
          composer,
          n_test_figs,
          step_size,
          mnum,
          gamma,
          logdir,
          batch_size,
          iterations,
          validation_interval,
          sequence_length,
          learning_rate,
          weight_decay,
          cnn_unit,
          fc_unit,
          level,
          debug=False,
          save_midi=False):
    #composer = ' '.join(composer.split('_'))
    if logdir is None:
        logdir = Path(composer) / (model_type)
    Path(logdir).mkdir(parents=True, exist_ok=True)

    if sequence_length % HOP_SIZE != 0:
        adj_length = sequence_length // HOP_SIZE * HOP_SIZE
        print(
            f'sequence_length {sequence_length} is not a multiple of {HOP_SIZE}.'
        )
        print(f'Adjusted to: {adj_length}')
        sequence_length = adj_length

    if debug:
        dataset = MAESTRO_small(groups=['debug'],
                                composer = composer,
                                sequence_length=sequence_length,
                                hop_size=HOP_SIZE,
                                random_sample=True,
                                level = level)
        valid_dataset = dataset
        iterations = 100
        validation_interval = 10
    else:
        dataset = MAESTRO_small(groups=['train'],
                                composer=composer,
                                sequence_length=sequence_length,
                                hop_size=HOP_SIZE,
                                random_sample=True,
                                level = level)
        valid_dataset = MAESTRO_small(groups=['validation'],
                                      composer=composer,
                                      sequence_length=sequence_length,
                                      hop_size=HOP_SIZE,
                                      random_sample=False,
                                      level = level)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'baseline':
        model = AutoSusPed(cnn_unit=cnn_unit, fc_unit=fc_unit, mnum = mnum, level = level)

    elif model_type == 'deeper':
        model = AutoSusPed_deeper(cnn_unit=cnn_unit, fc_unit=fc_unit, mnum=mnum, level = level)

    else:
        assert False, "this model_type is not supported - Wonjun Yi"

    optimizer = torch.optim.Adam(model.parameters(),
                                 learning_rate,
                                 weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = model.to(device)

    loop = tqdm(range(1, iterations + 1))

    valid_lists = []

    valid_model_perform = []

    for step, batch in zip(loop, cycle(loader)):
        optimizer.zero_grad()
        batch = allocate_batch(batch, device)

        if model_type != 'modify5':

            pedal_logit = model(batch['frame'])
            loss = -torch.log(pedal_logit)*batch['ccpedal']
            loss = loss.mean()

        elif model_type == 'modify5':


            pedal_logit = model(batch['frame'])
            loss = -torch.log(pedal_logit)*batch['ccpedal']
            loss = loss.mean()


        loss.mean().backward()

        for parameter in model.parameters():
            clip_grad_norm_([parameter], 3.0)

        optimizer.step()
        scheduler.step()
        loop.set_postfix_str("loss: {:.3e}".format(loss.mean()))

        if step % validation_interval == 0:
            model.eval()
            with torch.no_grad():
                loader = DataLoader(valid_dataset,
                                    batch_size=batch_size,
                                    shuffle=False)
                metrics = defaultdict(list)
                for batch in loader:
                    batch_results = evaluate(model, batch, device, model_type, level)

                    for key, value in batch_results.items():
                        metrics[key].extend(value)
            print('')

            valid_lists.append(metrics.items())
            valid_model_perform.append(np.array(metrics['metric/pedal/pedal_acc']).mean())
            for key, value in metrics.items():
                #if key[-2:] == 'f1' or 'loss' in key:
                print(f'{key:27} : {np.mean(value):.4f}')
            model.train()

            state_dict = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'step': step,
                'cnn_unit': cnn_unit,
                'fc_unit': fc_unit
            }
            torch.save(state_dict, Path(logdir) / f'model-{step}.pt')

    path_info = composer + '/' + model_type + '/'
    model.load_state_dict(torch.load(path_info+'model-'+str(np.where(np.array(valid_model_perform) == np.array(valid_model_perform).max())[0][0]*validation_interval+validation_interval)+'.pt')['model_state_dict'])
    torch.save(model, Path(logdir) / f'bestmodel-{np.where(np.array(valid_model_perform) == np.array(valid_model_perform).max())[0][0]*validation_interval+validation_interval}.pt')

    del dataset, valid_dataset


    valid_acc = torch.tensor([list(valid_lists[i])[1][1] for i in range(len(valid_lists))])
    vali = [list(valid_lists[i])[1][1] for i in range(len(valid_lists))]
    epoch = torch.tensor([int(idx*validation_interval) for idx in list(range(1, len(valid_acc.mean(dim=1))+1))])

    # save raw data as .txt file

    pandas.DataFrame(vali).to_csv(composer + "/" + model_type + "/valid_acc.csv")
    # print valid results

    valid_figure_on = plt.figure(constrained_layout = True)

    plt.title('validation accuracy')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.plot(epoch,valid_acc.mean(dim=1))
    plt.ylim(0, 1)

    valid_figure_on.savefig(composer+"/"+model_type+'/validation.svg')

    plt.cla()

    test_dataset = MAESTRO_small(groups=['test'],
                                 composer=composer,
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

                os.makedirs(composer+"/"+model_type+"/test_pics",exist_ok=True)


                test_plt.savefig("./"+composer+"/"+model_type+"/test_pics/"+batch['path'][0]+'_accuracy'+str(batch_results['metric/pedal/pedal_acc'][0])+'.jpg', dpi = 5000 * (level//2))

                test_plt.savefig("./"+composer+"/"+model_type+"/test_pics/"+batch['path'][0]+'_accuracy'+str(batch_results['metric/pedal/pedal_acc'][0])+'.svg', dpi = 5000 * (level//2))

                plt.cla()
                sw = sw + 1

            for key, value in batch_results.items():
                metrics[key].extend(value)
            metrics['path'].extend(batch['path'])
    print('')

    metric_panda = pandas.DataFrame(metrics)
    metric_panda.to_csv("./"+composer+"/"+model_type+"/test.csv")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='baseline', type=str)
    parser.add_argument('--level', default=2, type=int)
    parser.add_argument('--composer', default='Frédéric_Chopin', type=str)
    parser.add_argument('--n_test_figs', default=5, type=int)
    parser.add_argument('--mnum', default=0, type=int)
    parser.add_argument('--step_size', default=1000, type=int)
    parser.add_argument('--gamma', default=0.98, type=float)
    parser.add_argument('--logdir', default=None, type=str)
    parser.add_argument('-v', '--sequence_length', default=102400, type=int)
    parser.add_argument('-lr', '--learning_rate', default=6e-4, type=float)
    parser.add_argument('-b', '--batch_size', default=16, type=int)
    parser.add_argument('-i', '--iterations', default=10000, type=int)
    parser.add_argument('-vi', '--validation_interval', default=1000, type=int)
    parser.add_argument('-wd', '--weight_decay', default=0)
    parser.add_argument('-cnn', '--cnn_unit', default=48, type=int)
    parser.add_argument('-fc', '--fc_unit', default=256, type=int)
    parser.add_argument('--save_midi', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    train(**vars(parser.parse_args()))