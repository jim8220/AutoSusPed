import json
import os
from abc import abstractmethod
import pandas
import numpy as np
import pretty_midi
import soundfile
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from constants import HOP_SIZE, MAX_MIDI, MIN_MIDI, SAMPLE_RATE


def allocate_batch(batch, device):
    for key in batch.keys():
        if key != 'path':
            batch[key] = batch[key].to(device)
    return batch


class PianoSampleDataset(Dataset):
    def __init__(self,
                 path,
                 composer,
                 groups=None,
                 sample_length=16000 * 5,
                 hop_size=HOP_SIZE,
                 seed=42,
                 random_sample=True,
                 level = 2):
        self.path = 'data_analysis/maestro-v3.0.0-midi/maestro-v3.0.0/' + composer
        self.groups = groups if groups is not None else self.available_groups()
        assert all(group in self.available_groups() for group in self.groups)
        self.sample_length = None
        if sample_length is not None:
            self.sample_length = sample_length // hop_size * hop_size
        self.random = np.random.RandomState(seed)
        self.random_sample = random_sample
        self.hop_size = hop_size
        self.composer = composer
        self.file_list = dict()
        self.data = []
        self.level = level

        print(f'Loading {len(groups)} group(s) of', self.__class__.__name__,
              'at', path)
        for group in groups:
            self.file_list[group] = self.files(group)
            for input_files in tqdm(self.file_list[group],
                                    desc=f'Loading group {group}'):
                self.data.append(self.load('data_analysis/maestro-v3.0.0-midi/maestro-v3.0.0/' + input_files))

    def __getitem__(self, index):
        data = self.data[index]


        frames = (data['frame'] >= 1)
        pedals = (data['ccpedal'] >= 1)

        frame_len = frames.shape[0]
        if self.sample_length is not None:
            n_steps = self.sample_length // self.hop_size

            if self.random_sample:
                step_begin = self.random.randint(frame_len - n_steps)
                step_end = step_begin + n_steps
            else:
                step_begin = 0
                step_end = n_steps

            begin = step_begin * self.hop_size
            end = begin + self.sample_length


            frame_seg = frames[step_begin:step_end]
            pedal_seg = pedals[step_begin:step_end]

            result = dict(path=data['path'])

            result['frame'] = frame_seg.float()
            result['ccpedal'] = pedal_seg.float()
        else:
            result = dict(path=data['path'])

            result['frame'] = frames.float()
            result['ccpedal'] = pedals.float()
        return result

    def __len__(self):
        return len(self.data)

    @classmethod
    @abstractmethod
    def available_groups(cls):
        """Returns the names of all available groups."""
        raise NotImplementedError

    @abstractmethod
    def files(self, group):
        """Returns the list of input files (audio_filename, tsv_filename) for this group."""
        raise NotImplementedError

    def load(self, midi_path):
        """Loads an audio track and the corresponding labels."""

        sr = SAMPLE_RATE
        frames_per_sec = sr / self.hop_size






        midi = pretty_midi.PrettyMIDI(midi_path)
        midi_length_sec = midi.get_end_time()
        frame_length = int(midi_length_sec * frames_per_sec)


        frame = midi.get_piano_roll(fs=frames_per_sec)
        #ccpedal = np.zeros_like(frame)
        #for inst in midi.instruments:
            #for note in inst.notes:
                #ccpedal[note.pitch, int(note.start * frames_per_sec)] = 1

        cc_raw = midi.instruments[0].control_changes

        if [ipedal.value for ipedal in cc_raw if ipedal.number == 64] != []:

            cc_pedal = [ipedal.value for ipedal in cc_raw if ipedal.number == 64]
            cc_pedal_t = [ipedal.time for ipedal in cc_raw if ipedal.number == 64]
            tick_time = midi._tick_scales[0][1]
            cc_pedal_t_ = torch.tensor(cc_pedal_t)
            cc_pedal_tint = cc_pedal_t_ / tick_time # unit : tick

        else: # In case when pedal was not used for midi file -> zero-padding

            cc_pedal = [0, 0]
            cc_pedal_t = [0, midi_length_sec]
            tick_time = midi._tick_scales[0][1]
            cc_pedal_t_ = torch.tensor(cc_pedal_t)
            cc_pedal_tint = cc_pedal_t_ / tick_time # unit : tick


        # up-sampling cc_pedal (stay-and-hold)

        #pedal_0_len = int(midi_length_sec / tick_time)
        #pedal_0 = torch.zeros(pedal_0_len)

        #for tidx in range(len(cc_pedal_tint)):
            #pedal_0[int(cc_pedal_tint[tidx-1].item()):int(cc_pedal_tint[tidx].item())] = cc_pedal[tidx]

        #pedal_0[int(cc_pedal_tint[-1].item()):] = cc_pedal[-1]

        # But now, then down sample to match with frame

        pedal_1 = torch.zeros(frame_length, self.level)
        lterval = 128 // self.level
        tick_per_frame = 1/(frames_per_sec*tick_time)


        for tidx in range(len(cc_pedal_tint)):
            for ilevel in range(int(self.level)):
                if cc_pedal[tidx] >= lterval*ilevel and cc_pedal[tidx] < lterval*(ilevel+1):
                    pedal_1[int(int(cc_pedal_tint[tidx - 1].item()) / tick_per_frame):int(int(cc_pedal_tint[tidx].item()) / tick_per_frame), self.level - 1 - ilevel] = 1
        for ilevel in range(int(self.level)):
            if cc_pedal[-1] >= lterval * ilevel and cc_pedal[-1] < lterval * (ilevel + 1):
                pedal_1[int(int(cc_pedal_tint[-1].item()) / tick_per_frame):, self.level - 1 -ilevel] = 1

        #pedal_1[int(int(cc_pedal_tint[-1].item())/tick_per_frame):,0] = cc_pedal[-1]



        # to shape (time, pitch (88))
        frame = torch.from_numpy(frame[MIN_MIDI:MAX_MIDI + 1].T)
        #ccpedal = torch.from_numpy(ccpedal[MIN_MIDI:MAX_MIDI + 1].T)
        #data = dict(path=audio_path, audio=audio, frame=frame, ccpedal=ccpedal) # part where data is defined... change here
        data = dict(path=midi_path.split('/')[-1], frame=frame, ccpedal = pedal_1)
        return data


class MAESTRO_small(PianoSampleDataset):
    def __init__(self,
                 composer,
                 path='data',
                 groups=None,
                 sequence_length=None,
                 hop_size=512,
                 seed=42,
                 random_sample=True,
                 level = 2):
        super().__init__(path, composer, groups if groups is not None else ['train'],
                         sequence_length, hop_size, seed, random_sample, level)

    @classmethod
    def available_groups(cls):
        return ['train', 'validation', 'test', 'debug']

    def files(self, group):
        metadata = json.load(open(os.path.join(self.path, 'info.json')))

        if group == 'debug':
            files = sorted([
                (os.path.join(self.path,
                              row['audio_filename'].replace('.wav', '.flac')),
                 os.path.join(self.path, row['midi_filename']))
                for row in metadata if row['split'] == 'train'
            ])
            files = files[:10]
        else:
            metadata_panda = pandas.DataFrame.from_dict(metadata)
            files = list(metadata_panda[metadata_panda['split'] == group]['midi_filename'])

        return files