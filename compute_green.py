from modules.metrics_clinical import CheXbertMetrics

from modules.metrics import compute_scores
import pandas as pd

import logging
import os
from abc import abstractmethod
import numpy as np
import time
# from radgraph import F1RadGraph
import cv2
import torch
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from concurrent.futures import ThreadPoolExecutor

import sys
import time
sys.path.append('/home/shilei/project/RadCLIQ-CXR')
# from test_metric import cal_rad_cliq
from green_score import GREEN
os.environ['RANK'] = '0'  # 当前进程的 rank
os.environ['WORLD_SIZE'] = '1'  # 总的进程数
os.environ['MASTER_ADDR'] = 'localhost'  # 主节点地址
os.environ['MASTER_PORT'] = '12356'  # 主节点端口
model_name = "StanfordAIMI/GREEN-radllama2-7b"

green_scorer = GREEN(model_name, output_dir=".", device = 0)
# BATCH_SIZE = 16
device = torch.device("cuda:1")
# chexbert_metrics = CheXbertMetrics('/ssd/shilei/dataset/checkpoints/chexbert.pth', BATCH_SIZE, device)
Bleu_scorer = Bleu(4)
Rouge_scorer = Rouge()
Meteor_scorer = Meteor()

res_cmn_path = "cmn_mimic_train.csv"
res_mrg_path = "mrg_mimic_train.csv"
report_gts = pd.read_csv(res_cmn_path)['report_gt'].values.tolist()[0:80000]
report_cmn = pd.read_csv(res_cmn_path)['report_res'].values.tolist()[0:80000]
report_mrg = pd.read_csv(res_mrg_path)['report_res'].values.tolist()[0:80000]
ids = pd.read_csv(res_mrg_path)['id'].values.tolist()[0:80000]
# print(ids)
n = len(report_gts)
print(f'len:{n}')
batch_size = 1600  # 批次大小
# print(f"gt_len : {len(report_gts)} cmn_len : {len(report_cmn)} mrg_len : {len(report_mrg)}")
batches = [
    [
        report_gts[i:min(i + batch_size, n)],  # 确保切片结束索引不会超过总长度 n
        report_cmn[i:min(i + batch_size, n)],
        report_mrg[i:min(i + batch_size, n)]
    ]
    for i in range(0, n, batch_size)
]

# print(len(batches[0]))
cmn_b1, cmn_b4, cmn_rg, cmn_f1, cmn_cliqs, cmn_green = [], [], [], [], [], []
mrg_b1, mrg_b4, mrg_rg, mrg_f1, mrg_cliqs, mrg_green = [], [], [], [], [], []
_, _, green_score_list_cmn, _, _ = green_scorer(report_gts, report_cmn)
_, _, green_score_list_mrg, _, _ = green_scorer(report_gts, report_mrg)
cmn_green.extend(green_score_list_cmn)
mrg_green.extend(green_score_list_mrg)
# print(f'type_cmn:{type(test_cmn)} \n cmn:{test_cmn}')
# gts_ = {i: [test_gt[i]] for i in range(len(test_gt))}
#
# cmn_ = {i: [test_cmn[i]] for i in range(len(test_cmn))}
#
# mrg_ = {i : [test_mrg[i]] for i in range(len(test_mrg))}
#
# t1 = time.time()
# _, cmn_bleu_scores = Bleu_scorer.compute_score(gts_, cmn_, verbose = 0)
# _, mrg_bleu_scores = Bleu_scorer.compute_score(gts_, mrg_, verbose=0)
# cmn_b1.extend(cmn_bleu_scores[0])
# cmn_b4.extend(cmn_bleu_scores[3])
# mrg_b1.extend(mrg_bleu_scores[0])
# mrg_b4.extend(mrg_bleu_scores[3])
#
# _, cmn_rouge_scores = Rouge_scorer.compute_score(gts_, cmn_)
# _, mrg_rouge_scores = Rouge_scorer.compute_score(gts_, mrg_)
# cmn_rg.extend(cmn_rouge_scores)
# mrg_rg.extend(mrg_rouge_scores)
# f1radgraph = F1RadGraph(model_type="radgraph", reward_level="all", cuda=torch.device("cuda:0"))
# # cmn_reward
# mean_reward_cmn, reward_list_cmn, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
#     hyps=test_cmn, refs=test_gt)
#
# # mrg_reward
# mean_reward_mrg, reward_list_mrg, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
#     hyps=test_mrg, refs=test_gt)
#
# reward_list_cmn = [np.mean([reward_list_cmn[0][i], reward_list_cmn[1][i], reward_list_cmn[2][i]]) for i in range(len(reward_list_cmn[0]))]
# reward_list_mrg = [np.mean([reward_list_mrg[0][i], reward_list_mrg[1][i], reward_list_mrg[2][i]]) for i in range(len(reward_list_mrg[0]))]
#
# cmn_f1.extend(reward_list_cmn)
# mrg_f1.extend(reward_list_mrg)
# cmn_df = pd.DataFrame({'study_id': range(1, len(test_cmn) + 1), 'report': test_cmn})
# gt_df = pd.DataFrame({'study_id': range(1, len(test_gt) + 1), 'report': test_gt})
# mrg_df = pd.DataFrame({'study_id': range(1, len(test_mrg) + 1), 'report': test_mrg})
#
# # 保存为 CSV 文件
# cmn_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
# gt_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/gt.csv', index=False)
# cmn_cliq = cal_rad_cliq()
# mrg_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
# mrg_cliq = cal_rad_cliq()
#
# t2 = time.time()
# cmn_cliqs.extend(cmn_cliq)
#
# mrg_cliqs.extend(mrg_cliq)
# print(f'i============================================={i}')
# print(f'time:{t2 - t1} :.4f')
df = pd.DataFrame({
    'id' : ids,
     'cmn_green' : cmn_green,
    'mrg_green' : mrg_green
}).to_csv("train_green_all_1.csv")

print('save successful!!')



