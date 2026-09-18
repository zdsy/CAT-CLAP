"""
Script contains all required prep batch functions for fixed and variable batch
    functions. Includes:
        -> Normal fixed length batcher with trans keywork for transpose operation
        -> Variable length train and eval prep batch functions, both with trans
            keyword for transpose operation
"""

###############################################################################
# IMPORTS
###############################################################################
import torch
import numpy as np

###############################################################################
# FIXED LENGTH BATCHING FUNCTION (TRAIN AND EVAL)
###############################################################################
###############################################################################
# FIXED LENGTH BATCHING FUNCTION (TRAIN AND EVAL)
###############################################################################
def prep_batch_fixed(n_way, k_shot, q_queries, device, trans):
    def prep_batch_fixed_child(batch, meta_batch_size):
        # [修改 1]: 接收 4 个元素
        x, wav, y, class_names_raw = batch

        x = x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                            x.shape[-2], x.shape[-1])
        wav = wav.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                            wav.shape[-1])

        if trans:
            x = torch.transpose(x, 2, 3)
            x = x.double().to(device)
        else:
            x = x.unsqueeze(2).double().to(device)

        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)
        y = y.long().to(device)

        # [修改 2]: 精准提取每个 Episode 的 N 个独特 Class Names
        batch_class_names = []
        samples_per_ep = n_way*k_shot + n_way*q_queries
        for i in range(meta_batch_size):
            ep_classes = []
            for w in range(n_way):
                # 按照 Support set 的排列规律，每隔 k_shot 取一个名字
                ep_classes.append(class_names_raw[i * samples_per_ep + w * k_shot])
            batch_class_names.append(ep_classes)

        # [修改 3]: 返回值增加 batch_class_names
        return x, wav, y, batch_class_names
    return prep_batch_fixed_child

###############################################################################
# VARIABLE LENGTH TRAIN BATCH FUNCTION
###############################################################################
###############################################################################
# VARIABLE LENGTH TRAIN BATCH FUNCTION
###############################################################################
def prep_var_train(n_way, k_shot, q_queries, device, trans):
    def prep_var_train_child(batch, meta_batch_size):
        # [修改 1]: 从 x, y 变更为接收 4 个元素
        x, wav, y, class_names_raw = batch

        new_x = torch.zeros(meta_batch_size*(n_way*k_shot + n_way*q_queries), x[0].shape[-2], x[0].shape[-1])
        for idx, samples in enumerate(x):
            ind = np.random.choice(samples.shape[0])
            new_x[idx] = samples[ind]

        x = new_x.reshape(meta_batch_size, (n_way*k_shot + n_way*q_queries),
                            x[0].shape[-2], x[0].shape[-1] )

        if trans:
            x = torch.transpose(x, 2, 3)
            x = x.double().to(device)
        else:
            x = x.unsqueeze(2).double().to(device)

        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)
        y = y.long().to(device)

        # [修改 2]: 提取 Class Names
        batch_class_names = []
        samples_per_ep = n_way*k_shot + n_way*q_queries
        for i in range(meta_batch_size):
            ep_classes = []
            for w in range(n_way):
                ep_classes.append(class_names_raw[i * samples_per_ep + w * k_shot])
            batch_class_names.append(ep_classes)

        # [修改 3]: 返回值带上 wav 和 class_names
        return x, wav, y, batch_class_names
    return prep_var_train_child

###############################################################################
# VARIABLE LENGTH EVAL BATCH FUNCTION
###############################################################################
def prep_var_eval(n_way, k_shot, q_queries, device, trans, target_sr=48000):
    def prep_var_eval_child(batch, meta_batch_size):
        # [修改 1]: 接收 4 个元素
        x, wav, y, class_names_raw = batch

        end_index = 0
        supports, queries = [], []
        support_wavs = []
        
        # [新增]: 为 CLAP 专门准备的结构
        batch_class_names = []
        batch_wavs = [] # 将包含 meta_batch_size 个子列表，每个子列表装着该 episode 所有的 raw wavs

        for i in range(meta_batch_size):
            ep_wavs = []
            ep_classes = []
            
            # --- 处理 SUPPORT ---
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*k_shot) ] ):
                supports.append(samples)
            for wav_idx, wavs in enumerate( wav[ end_index : end_index + (n_way*k_shot) ] ):
                support_wavs.append(wavs)
                ep_wavs.append(wavs) # 添加 support wav 到当前 episode
                
            # 获取当前 episode 的独特类别名
            for w in range(n_way):
                ep_classes.append(class_names_raw[end_index + w * k_shot])
            batch_class_names.append(ep_classes)
            
            end_index += n_way*k_shot
            
            # --- 处理 QUERY ---
            for idx, samples in enumerate( x[ end_index : end_index + (n_way*q_queries) ] ):
                queries.append(samples)
            for wav_idx, wavs in enumerate( wav[ end_index : end_index + (n_way*q_queries) ] ):
                ep_wavs.append(wavs) # [关键修复]: 将 Query 的 wav 也提取出来，供 Zero-Shot/CMHT 使用
                
            end_index += n_way*q_queries
            
            batch_wavs.append(ep_wavs)

        # x_support = torch.zeros(meta_batch_size*(n_way*k_shot), x[0].shape[-2], x[0].shape[-1])
        # L = 5 * target_sr
        # x_support_wavs = torch.zeros(len(supports), L)
        
        # for idx, samples in enumerate(supports):
        #     ind = np.random.choice(samples.shape[0])
        #     x_support[idx] = samples[ind]
        #     raw_full = support_wavs[idx]
        #     start = ind * L
        #     chunk = raw_full[start : start + L]
        #     if chunk.shape[0] < L:
        #         if chunk.shape[0] == 0:
        #             print(f"Warning: Empty audio slice at index {ind}. Audio length: {raw_full.shape[0]}")
        #         else:
        #             multiply_up = int(np.ceil(L / chunk.shape[0]))
        #             chunk = chunk.repeat(multiply_up)[:L]

        #     x_support_wavs[idx] = chunk
            
        # x_support = x_support.reshape(meta_batch_size, (n_way*k_shot), x[0].shape[-2], x[0].shape[-1])
        # x_support_wavs = x_support_wavs.reshape(meta_batch_size, (n_way*k_shot), L)

        x_queries = torch.zeros(1, x[0].shape[-2], x[0].shape[-1])
        query_sample_nums = []
        for idx, samples in enumerate(queries):
            query_sample_nums.append(samples.shape[0])
            for j, samp in enumerate(samples):
                if samp.ndim == 2:
                    samp = samp.unsqueeze(0)
                x_queries = torch.cat((x_queries, samp), 0)

        x_queries = x_queries[1:]

        y_tr = torch.arange(0, n_way, 1/k_shot)
        y_val = torch.arange(0, n_way, 1/q_queries)
        y = torch.cat((y_tr, y_val))
        y = y.unsqueeze(0).repeat(meta_batch_size, 1)
        y = y.long().to(device)

        # if trans:
        #     x_support = torch.transpose(x_support, 2, 3)
        #     x_queries = torch.transpose(x_queries, 1, 2)
        #     x_support = x_support.double().to(device)
        #     x_queries = x_queries.double().to(device)
        # else:
        #     x_support = x_support.unsqueeze(2).double().to(device)
        #     x_queries = x_queries.unsqueeze(1).double().to(device)

        # [修改 4]: 在最后追加 batch_wavs 和 batch_class_names
        # return x_support, x_support_wavs.to(device), x_queries, query_sample_nums, y, batch_wavs, batch_class_names
        return 0, 0, 0, query_sample_nums, y, batch_wavs, batch_class_names
    return prep_var_eval_child