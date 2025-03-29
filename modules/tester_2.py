import logging
import os
from abc import abstractmethod
import numpy as np
import time
import pandas as pd
import cv2
import torch
from radgraph import F1RadGraph
# from green_score import GREEN
from .metrics_clinical import CheXbertMetrics
import sys
sys.path.append('/home/shilei/project/RadCLIQ-CXR')
from test_metric import cal_rad_cliq
sys.path.append("/home/shilei/project/PromptMRG/GREEN")
# from green_score import GREEN
import os
class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('/ssd/shilei/dataset/checkpoints/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():
            test_gts, test_res = [], []
            test_ids = []
            for batch_idx, (id, images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                reports, _, _ = self.model.generate(images, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)

                test_res.extend(reports)
                test_gts.extend(ground_truths)
                test_ids.extend(id)
                if batch_idx % 10 == 0:
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            # model_name = "StanfordAIMI/GREEN-radllama2-7b"
            # #
            # green_scorer = GREEN(model_name, output_dir=".")
            # _, _, green_score_list, _, _ = green_scorer(test_gts, test_res)
            # print('green_score == ', green_score_list)
            # avg_green = sum(green_score_list) * 1.0 / len(green_score_list)
            # print("green:",avg_green)
            df = pd.DataFrame({
                'id' : test_ids,
                'report_res' : test_res,
                'report_gt' : test_gts
            }).to_csv("test_result/0117_test_sft2_f1.csv")
            print('generation complete!!')
            # cmn_df = pd.DataFrame({'study_id': range(1, len(test_res) + 1), 'report': test_res})
            # gt_df = pd.DataFrame({'study_id': range(1, len(test_gts) + 1), 'report': test_gts})
            # # mrg_df = pd.DataFrame({'study_id': range(1, len(test_mrg) + 1), 'report': test_mrg})
            #
            # # 保存为 CSV 文件
            # cmn_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
            # gt_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/gt.csv', index=False)
            # cmn_cliq = cal_rad_cliq()
            # print('cliq:', sum(cmn_cliq) * 1.0 / len(cmn_cliq))
            # print('start compute F1')
            # t1 = time.time()
            # print(f'time:{t1}')
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            # f1radgraph = F1RadGraph(model_type="radgraph", reward_level="all", cuda=torch.device("cuda"))
            # mean_reward_cmn, reward_list_cmn, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
            #     hyps=test_res, refs=test_gts)
            # reward_list = [np.mean([reward_list_cmn[0][i], reward_list_cmn[1][i], reward_list_cmn[2][i]]) for i in
            #                    range(len(reward_list_cmn[0]))]

            # t2 =time.time()
            # print(f'time:{t2}')
            #
            # print(f'cost time:{t2 - t1}')
            # print('rad-f1:', 1.0 * sum(reward_list) / len(reward_list))
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
            # log.update({'test_green' : avg_green})
        return log

