import os
from abc import abstractmethod
import datetime
import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
from .metrics_clinical import CheXbertMetrics
import copy
from .optims import LinearWarmupCosineLRScheduler
from typing import Tuple
import torch.nn.functional as F
import sys
from radgraph import F1RadGraph
import wandb
sys.path.append('/home/shilei/project/RadCLIQ-CXR')
from test_metric import cal_rad_cliq

from RaTEScore import RaTEScore
import time
# sys.path.append("/home/shilei/project/PromptMRG/GREEN")
# from green_score import GREEN
class BaseTrainer(object):
    def __init__(self, model_sft, model_policy, model_peft_1, model_peft_2, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model_policy = model_policy
        self.model_sft = model_sft
        self.model_peft_1 = model_peft_1
        self.model_peft_2 = model_peft_2

        self.device = device
        self.is_main_process = is_main_process

        self.w = self.args.w

        self.chexbert_metrics = CheXbertMetrics('/ssd/shilei/dataset/checkpoints/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls
        self.index = self.args.index
        self.index2 = 0

        self.data = pd.read_csv('data_green/result.csv')
        self.b1 = self.data['BLEU_1'].values.tolist()
        self.b4 = self.data['BLEU_4'].values.tolist()
        self.F1 = self.data['F1'].values.tolist()
        self.CliQ = self.data['CliQ'].values.tolist()

        ori_dpo_vectors = [[1, 0, 0], [0.8, 0.1, 0.1], [0.1, 0.8 ,0.1], [0.1, 0.1, 0.8],[0.33, 0.33, 0.33], [0.7,0.3,0],[0.4,0.6,0],[0.1,0.9,0],[0.7, 0, 0.3], [0.4, 0, 0.6], [0.1, 0, 0.9]]
        self.dpo_vectors = torch.tensor(ori_dpo_vectors, dtype=torch.float32)
        self.w = None
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns

        self.beta = self.args.beta #modpo_loss 超参

        self.mrg_beta = self.args.mrg_beta

        self.dpo_beta = self.args.dpo_beta
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model_policy.named_parameters():
            # if not p.requires_grad:
            #     continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'test_' + args.monitor_metric

        self.mnt_best = 0
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        self.loss_type = "sigmoid"  #modpo_loss 的 type

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        today_date = datetime.datetime.now().strftime("%m%d")  # 格式为 MMDD，例如：0612
        w = self.dpo_vectors[self.index]
        # wandb.init(
        #     project="MODPO",  # 项目名称
        #     name=f"{today_date}_{self.args.mrg_beta}_{self.args.dpo_beta}_{self.args.beta}_{self.args.seed}_{w[0]:.1f}_{w[1]:.1f}_{w[2]:.1f}" ,  # 运行名称，可自定义
        # )
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)
            self.w = self.dpo_vectors[self.index]

            result = self._train_epoch_blip(epoch)

            # dist.barrier()
            result = self.eval_blip(result)

            # save logged information
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] + log['test_ce_f1'] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric] + log['test_ce_f1']
                    self.log_best = log
                    best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                    torch.save(self.model_policy.state_dict(), best_path)
                    # print('Best RadF1:', self.mnt_best)
                    print("Saving current best to {}".format(best_path))

            # print logged information
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model_sft, model_policy, model_peft_1, model_peft_2, criterion_cls, base_probs, metric_ftns, args, train_dataloader, val_dataloader,
                 test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model_sft, model_policy, model_peft_1, model_peft_2, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            self.args.epochs,
            self.args.min_lr,
            self.args.init_lr,
            decay_rate=None,
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )
    def modpo_loss(
            self,
            policy_chosen_logps: torch.FloatTensor,
            policy_rejected_logps: torch.FloatTensor,
            reference_chosen_logps: torch.FloatTensor,
            reference_rejected_logps: torch.FloatTensor,
            chosen_margin_reward: torch.FloatTensor,
            rejected_margin_reward: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:#需要检查一下这几个的size情况 明天继续测试一下
        # chosen_margin_reward = chosen_margin_reward.float()
        # rejected_margin_reward = rejected_margin_reward.float()
        policy_rejected_logps = policy_rejected_logps.to(self.device)
        policy_chosen_logps = policy_chosen_logps.to(self.device)
        reference_rejected_logps = reference_rejected_logps.to(self.device)
        reference_chosen_logps = reference_chosen_logps.to(self.device)
        chosen_margin_reward = chosen_margin_reward.to(self.device)
        rejected_margin_reward = rejected_margin_reward.to(self.device)
        self.w = self.w.to(self.device)
        chosen_rewards = (1 / self.w[0]) * (
                    self.beta * (policy_chosen_logps - reference_chosen_logps) - (policy_chosen_logps.unsqueeze(1) - chosen_margin_reward) @ self.w[1:])
        rejected_rewards = (1 / self.w[0]) * (
                    self.beta * (policy_rejected_logps - reference_rejected_logps) - (policy_rejected_logps.unsqueeze(1) - rejected_margin_reward) @ self.w[1:])

        logits = chosen_rewards - rejected_rewards
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        return losses, chosen_rewards.detach(), rejected_rewards.detach()

    def _train_epoch_blip(self, epoch):
        train_loss = 0
        train_loss_dpo = 0
        train_loss_mrg = 0

        for batch_idx, (id, images, captions, y_w, y_l, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            # print(f'y_w_margin_origin:{yw_margin_reward.shape}')
            # loss_lm, loss_cls = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls,
            #                                self.base_probs)
            # loss = loss_lm + self.args.cls_weight * loss_cls
            with torch.no_grad():
                sft_chosen_logps, sft_reject_logps = self.model_sft.generate_sft(images, y_w, y_l, captions, cls_labels, clip_memory, self.criterion_cls,
                                                                                 self.base_probs)

                yw_margin_reward_1, yl_margin_reward_1 = self.model_peft_1.generate_peft(images, y_w, y_l, captions, cls_labels, clip_memory, self.criterion_cls,
                                                                                 self.base_probs)

                yw_margin_reward_2, yl_margin_reward_2 = self.model_peft_2.generate_peft(images, y_w, y_l, captions,
                                                                                     cls_labels, clip_memory,
                                                                                     self.criterion_cls,
                                                                                     self.base_probs)

                yw_margin_reward, yl_margin_reward = torch.stack((yw_margin_reward_1, yw_margin_reward_2), dim=1), torch.stack((yl_margin_reward_1, yl_margin_reward_2), dim=1)

                # print(f'y_w_shape:{yw_margin_reward.shape}')
                # print(f'y_w_1_shape:{yw_margin_reward_1.shape}')



            self.model_policy.train()
            policy_chosen_logps, policy_rejected_logps, loss_lm, loss_cls = self.model_policy(images, y_w, y_l, captions, cls_labels, clip_memory, self.criterion_cls,
                                          self.base_probs)

            loss_dpo, chosen_reward, rejected_reward = self.modpo_loss(policy_chosen_logps, policy_rejected_logps, sft_chosen_logps, sft_reject_logps, yw_margin_reward, yl_margin_reward)
            # print(f'loss_dpo:{loss_dpo}')

            # 获取策略模型对 yw 和 yl 的预测概率的对数
            # log_policy_yw = torch.log(policy_probs.gather(dim=-1, index=yw.unsqueeze(-1)))
            # log_policy_yl = torch.log(policy_probs.gather(dim=-1, index=yl.unsqueeze(-1)))
            # wandb.log({
            #     'loss_lm' : loss_lm,
            #     'loss_cls' : loss_cls,
            #     'loss_dpo' : loss_dpo.mean()
            # })

            loss = self.mrg_beta * loss_lm + loss_cls * self.args.cls_weight + self.dpo_beta * loss_dpo.mean()
            if batch_idx % 10 == 0:
                print("{}/{} loss: {} loss_lm: {} loss_cls: {} loss_dpo: {} chosen_reward: {} rejected_reward: {}".format(batch_idx, len(self.train_dataloader),
                                                                 loss.item(), loss_lm.item(), self.args.cls_weight * loss_cls.item(), loss_dpo.mean(), chosen_reward.mean(), rejected_reward.mean()))

            train_loss += loss.item()
            train_loss_dpo += loss_dpo.mean()
            train_loss_mrg += (loss_lm.item() + loss_cls.item() * self.args.cls_weight)
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model_policy.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        # wandb.log({
        #     'train_loss' : train_loss / len(self.train_dataloader),
        #     'train_loss_dpo' : train_loss_dpo / len(self.train_dataloader),
        #     'train_loss_mrg' : train_loss_mrg / len(self.train_dataloader)
        # })

        return log

    def eval_blip(self, log):
        self.model_policy.eval()

        logits = []
        counts = []
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (id, images, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device)
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                reports, cls_preds, cls_preds_logits = self.model_policy.generate(images, clip_memory, sample=False,
                                                                                  num_beams=self.args.beam_size,
                                                                                  max_length=self.args.gen_max_len,
                                                                                  min_length=self.args.gen_min_len)
                # logit adjustment
                cls_labels = (cls_labels == 1).float()
                logit = cls_preds_logits * cls_labels
                logits.append(logit.cpu().numpy())
                counts.append(cls_labels.cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            #######
            logits = np.concatenate(logits, axis=0)
            counts = np.concatenate(counts, axis=0)
            logits = np.sum(logits, 0)
            counts = np.sum(counts, 0)
            logits = logits / counts
            logits /= np.max(logits)
            logits = np.append(logits, [1, 1, 1, 1])  # 4 auxiliary diseases
            #######
            self.base_probs = logits  # update class distribution
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            # f1radgraph = F1RadGraph(model_type="radgraph", reward_level="all", cuda=torch.device("cuda"))
            # # cmn_reward
            # mean_reward_cmn, reward_list_cmn, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
            #     hyps=val_res, refs=val_gts)
            #
            # reward_list = [np.mean([reward_list_cmn[0][i], reward_list_cmn[1][i], reward_list_cmn[2][i]]) for i in
            #                range(len(reward_list_cmn[0]))]

            # Rad_F1 = sum(reward_list) * 1.0 / len(reward_list)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_ce.items()})
            # log.update({'val_rad_f1': Rad_F1})

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (id, images, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device)
                cls_labels = cls_labels.numpy().tolist()
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                reports, _, _ = self.model_policy.generate(images, clip_memory, sample=False,
                                                           num_beams=self.args.beam_size,
                                                           max_length=self.args.gen_max_len,
                                                           min_length=self.args.gen_min_len)

                test_res.extend(reports)
                test_gts.extend(ground_truths)
            # model_name = "StanfordAIMI/GREEN-radllama2-7b"
            # #
            # green_scorer = GREEN(model_name, output_dir=".")
            # _, _, green_score_list, _, _ = green_scorer(test_gts, test_res)
            # print('green_score == ', green_score_list)
            # f1radgraph = F1RadGraph(model_type="radgraph", reward_level="all", cuda=torch.device("cuda:0"))
            # # cmn_reward
            # mean_reward_cmn, reward_list_cmn, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
            #     hyps=test_res, refs=test_gts)
            #
            # reward_list = [np.mean([reward_list_cmn[0][i], reward_list_cmn[1][i], reward_list_cmn[2][i]]) for i in
            #                range(len(reward_list_cmn[0]))]

            # Rad_F1 = sum(reward_list) * 1.0 / len(reward_list)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})

            test_ce = self.chexbert_metrics.compute(test_gts, test_res)

            ratescore = RaTEScore(batch_size=64)

            scores = ratescore.compute_score(test_res, test_gts)

            test_rate_score = 1.0 * sum(scores) / len(scores)

            log.update({'test_rate_score' : test_rate_score})

            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
            # cmn_df = pd.DataFrame({'study_id': range(1, len(test_res) + 1), 'report': test_res})
            # gt_df = pd.DataFrame({'study_id': range(1, len(test_gts) + 1), 'report': test_gts})
            # # mrg_df = pd.DataFrame({'study_id': range(1, len(test_mrg) + 1), 'report': test_mrg})
            #
            # # 保存为 CSV 文件
            # cmn_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
            # gt_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/gt.csv', index=False)
            # mrg_cliq = cal_rad_cliq()
            # avg_cliq = sum(mrg_cliq) * 1.0 / len(mrg_cliq)
            # log.update({'test_F1' : Rad_F1})
            # log.update({'test_BLEU_1' : self.b1[self.index2]})
            # log.update({'test_BLEU_4' : self.b4[self.index2]})
            # log.update({'test_CliQ' : self.CliQ[self.index2]})
            # wandb.log({
            #     'test_BLEU_1' : self.b1[self.index2],
            #     'test_BLEU_4' : self.b4[self.index2],
            #     'test_F1' : self.F1[self.index2],
            #     'test_CliQ' : self.CliQ[self.index2],
            #     'test_rate_score' : test_rate_score
            # })
            self.index = (self.index + 1) % len(self.dpo_vectors)
            self.index2 = (self.index2 + 1) % len(self.b1)
        return log


