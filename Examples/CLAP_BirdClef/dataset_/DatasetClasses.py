"""
Script contains the generalised dataset classes that are used for experiments,
    these include:
        -> General raw
        -> General spectrogram
    Each of these scripts is equiped with a variery of normalisation options

File also contains the basic base dataset classes that are used in the general 
    experiment framework. These datasets are also recycled 

Contains:
    -> Generic
"""

###############################################################################
# IMPORTS
###############################################################################
import os
import torch
import numpy as np
import pandas as pd
import torchaudio

from sklearn import preprocessing
from torch.utils.data import Dataset
from dataset_.dataset_stuff import per_sample_scale, nothing_func, given_stats_scale
import torchaudio.transforms as T

def safe_load_and_resample_for_clap(wav_path, target_sr=48000, max_duration_sec=40):
    """
    安全加载音频，统一处理采样率和通道数，并防御损坏文件。
    """
    try:
        # 1. 加载原始音频
        waveform, sr = torchaudio.load(wav_path)
        
        # 2. 通道对齐：CLAP 需要单声道 (Mono)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # 3. 频率对齐：强制重采样到 CLAP 的标准 48kHz
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        max_samples = target_sr * max_duration_sec
        if waveform.shape[1] > max_samples:
            # 你可以随机截取，这里为了演示直接取前 max_samples
            waveform = waveform[:, :max_samples]
            
        return waveform, target_sr

    except Exception as e:
        # 打印警告，避免进程崩溃
        print(f"\n[Warning] Corrupted audio file bypassed: {wav_path}. Error: {e}")
        # 返回一段标准的 48kHz 静音张量，防止下游维度错位报错
        # 长度默认为 max_duration_sec 秒
        dummy_silence = torch.zeros(1, target_sr * max_duration_sec)
        return dummy_silence, target_sr


###############################################################################
# NORMAL DATASET CLASS (GENERAL), WORKS FOR RAW AND SPEC
###############################################################################
class NormDataset(Dataset):
    def __init__(self, data_path, wav_path, classes, norm, stats_file_path, target_sr=48000):

        self.norm = norm
        self.classes = classes
        self.data_path = data_path
        self.wav_path = wav_path
        self.target_sr = target_sr

        self.norm_func = self.set_norm_func(norm, stats_file_path)

        self.df = pd.DataFrame(self.get_subset())
        self.df = self.df.assign(id=self.df.index.values)

        # Grabs all the class names
        self.unique_characters = sorted(self.df['class_name'].unique())

        # Creates key:pair for class_name:numeric class_id
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}

        # Creates a class_id column in df using the class_name to class_id dict
        self.df = self.df.assign(
            class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Organises the sample ids and paths into iterable arrays
        self.id_to_path = self.df.to_dict()['filepath']
        self.id_to_wav_path = self.df.to_dict()['wav_path']
        self.id_to_class_id = self.df.to_dict()['class_id']
        # [新增]: 记录对应的确切文本名称
        self.id_to_class_name = self.df.to_dict()['class_name'] 

    def set_norm_func(self, norm, stats_file):
        # ... (保持不变) ...
        if norm == 'l2':
            norm_func = preprocessing.normalize
        elif norm == 'None':
            norm_func = nothing_func
        elif norm == 'per_sample':
            norm_func = per_sample_scale
        elif norm == 'global':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale
        elif norm == 'channel':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale
        else:
            raise ValueError('Passes norm type unsupported')
        return norm_func


    def __getitem__(self, item):
        # [安全修复]: 强制将 numpy 数组或 tensor 转换为原生 int，防止 unhashable type 报错
        if isinstance(item, (np.ndarray, torch.Tensor)):
            item = item.item()
        item = int(item)

        sample = np.load(self.id_to_path[item])
        sample = torch.from_numpy(sample)
        # waveform, sr = torchaudio.load(self.id_to_wav_path[item])
        waveform, sr = safe_load_and_resample_for_clap(self.id_to_wav_path[item])

        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Final raw audio for CLAP
        raw_sample = waveform.squeeze(0) # Shape: [Time_Samples]

        # Deals with normalisation of various types
        if self.norm in ['global', 'channel']:
            sample = self.norm_func(sample, self.mu, self.sigma)
        else:
            sample = self.norm_func(sample)

        label = self.id_to_class_id[item]
        # [新增]: 提取 class_name 文本
        class_name = self.id_to_class_name[item] 

        # [修改]: 返回值加入 class_name
        return sample, raw_sample, label, class_name

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def get_subset(self):
        # ... (保持不变) ...
        audio_samples = []
        for root, folders, files in os.walk(self.data_path):
            if len(files) == 0:
                continue
            class_name = root.split('/')[-1]
            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        npy_filepath = os.path.join(root, f)
                        wav_filename = f.replace('.npy', '.mp3')
                        wav_filepath = os.path.join(self.wav_path, class_name, wav_filename)
                        if os.path.exists(wav_filepath):
                            audio_samples.append({
                                'class_name': class_name,
                                'filepath': npy_filepath,
                                'wav_path': wav_filepath
                                })
                        else:
                            print(f"Warning: WAV file not found for {npy_filepath}")
                            print(wav_filepath)
        return audio_samples


##############################################################################
# TRAINING VARIABLE LENGTH DATASET WITH SPECS OR RAW
##############################################################################
class TrainingVariableDataset(Dataset):
    def __init__(self, data_path, wav_path, classes, norm, stats_file_path, target_sr=48000):

        self.norm = norm
        self.classes = classes
        self.data_path = data_path
        self.wav_path = wav_path
        self.target_sr = target_sr
        self.norm_func = self.set_norm_func(norm, stats_file_path)

        self.df = pd.DataFrame(self.get_subset())
        self.df = self.df.assign(id=self.df.index.values)

        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(
            class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        self.id_to_path = self.df.to_dict()['filepath']
        self.id_to_wav_path = self.df.to_dict()['wav_path']
        self.id_to_class_id = self.df.to_dict()['class_id']
        # [新增]: 记录对应的确切文本名称
        self.id_to_class_name = self.df.to_dict()['class_name']

    def set_norm_func(self, norm, stats_file):
        # ... (保持不变) ...
        if norm == 'l2':
            norm_func = preprocessing.normalize
        elif norm == 'None':
            norm_func = nothing_func
        elif norm == 'per_sample':
            norm_func = per_sample_scale
        elif norm == 'global':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale
        elif norm == 'channel':
            mu, sigma = np.load(stats_file, allow_pickle=True)
            self.mu = torch.from_numpy(np.asarray(mu))
            self.sigma = torch.from_numpy(np.asarray(sigma))
            norm_func = given_stats_scale
        else:
            raise ValueError('Passes norm type unsupported')
        return norm_func


    def __getitem__(self, item):
        # [安全修复]: 强制将 numpy 数组或 tensor 转换为原生 int
        if isinstance(item, (np.ndarray, torch.Tensor)):
            item = item.item()
        item = int(item)

        sample = np.load(self.id_to_path[item])
        # waveform, sr = torchaudio.load(self.id_to_wav_path[item])
        waveform, sr = safe_load_and_resample_for_clap(self.id_to_wav_path[item])
        
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        sample = torch.from_numpy(sample)
        idx = np.random.choice(sample.shape[0])
        sample = sample[idx]

        audio_len = waveform.shape[1]
        L = 5 * self.target_sr

        if audio_len <= 0:
            raw_chunk = torch.zeros(1, L, dtype=waveform.dtype)
        else:
            # The spectrogram segment count can exceed available 5-second raw
            # chunks after audio loading/truncation. Wrap the segment index so
            # CLAP always receives a non-empty chunk aligned to the clip.
            available_chunks = max(1, int(np.ceil(audio_len / L)))
            safe_idx = int(idx) % available_chunks
            start = min(safe_idx * L, max(audio_len - 1, 0))
            raw_chunk = waveform[:, start : min(start + L, audio_len)]

            if raw_chunk.shape[1] == 0:
                raw_chunk = waveform[:, :min(audio_len, L)]

            if raw_chunk.shape[1] < L:
                multiply_up = int(np.ceil(L / max(raw_chunk.shape[1], 1)))
                raw_chunk = raw_chunk.repeat(1, multiply_up)[:, :L]

        raw_sample = raw_chunk.squeeze(0) # [Time]

        if self.norm in ['global', 'channel']:
            sample = self.norm_func(sample, self.mu, self.sigma)
        else:
            sample = self.norm_func(sample)

        label = self.id_to_class_id[item]
        # [新增]: 提取 class_name 文本
        class_name = self.id_to_class_name[item]

        # [修改]: 返回值加入 class_name
        return sample, raw_sample, label, class_name

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    def get_subset(self):
        # ... (保持不变) ...
        audio_samples = []
        for root, folders, files in os.walk(self.data_path):
            if len(files) == 0:
                continue
            class_name = root.split('/')[-1]
            if class_name in self.classes:
                for f in files:
                    if f.endswith('.npy'):
                        npy_filepath = os.path.join(root, f)
                        wav_filename = f.replace('.npy', '.mp3')
                        wav_filepath = os.path.join(self.wav_path, class_name, wav_filename)
                        if os.path.exists(wav_filepath):
                            audio_samples.append({
                                'class_name': class_name,
                                'filepath': npy_filepath,
                                'wav_path': wav_filepath
                                })  
                        else:
                            print(f"Warning: WAV file not found for {npy_filepath}")
                            print(wav_filepath)
        return audio_samples

 
###############################################################################
# STABLE NUM WORKERS DATALOADER
###############################################################################
class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class FastDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
