import os
from abc import abstractmethod

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
class BaseTrainer(object):
    def __init__(self, model_sft, model_policy, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model_policy = model_policy
        self.model_sft = model_sft
        self.device = device
        self.is_main_process = is_main_process

        self.chexbert_metrics = CheXbertMetrics('/ssd/shilei/dataset/checkpoints/chexbert.pth', args.batch_size, device)

        self.criterion_cls = criterion_cls
        self.index = 0

        ori_dpo_vectors = [[0.6, 0.4], [0.7, 0.3], [0.8, 0.2]]
        self.dpo_vectors = torch.tensor(ori_dpo_vectors, dtype=torch.float32)
        self.w = None
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns

        self.beta = 0.1 #modpo_loss 超参
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model_policy.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
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

        self.mnt_metric = 'val_' + args.monitor_metric

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
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)
            self.w = self.dpo_vectors[self.index]
            self.index = (self.index + 1) % 10
            result = self._train_epoch_blip(epoch)
            print('epoch end!!!!')
            print('continue!!!')
            result = self.eval_blip(result)

            # save logged information
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = log
                    best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
                    torch.save(self.model_policy.state_dict(), best_path)
                    print("Saving current best to {}".format(best_path))

            # print logged information
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model_sft, model_policy, criterion_cls, base_probs, metric_ftns, args, train_dataloader, val_dataloader,
                 test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model_sft, model_policy, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
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
                    self.beta * (policy_chosen_logps - reference_chosen_logps) - chosen_margin_reward @ self.w[1:])
        rejected_rewards = (1 / self.w[0]) * (
                    self.beta * (policy_rejected_logps - reference_rejected_logps) - rejected_margin_reward @ self.w[1:])

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

        for batch_idx, (id, images, captions, y_w, y_l, yw_margin_reward, yl_margin_reward, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            # loss_lm, loss_cls = self.model(images, captions, cls_labels, clip_memory, self.criterion_cls,
            #                                self.base_probs)
            # loss = loss_lm + self.args.cls_weight * loss_cls

            self.model_sft.eval()
            with torch.no_grad():
                sft_chosen_logps, sft_reject_logps = self.model_sft.generate_sft(images, y_w, y_l, clip_memory)

            self.model_policy.train()
            policy_chosen_logps, policy_rejected_logps = self.model_policy(images, y_w, y_l, captions, cls_labels, clip_memory, self.criterion_cls,
                                          self.base_probs)

            loss_dpo, chosen_reward, rejected_reward = self.modpo_loss(policy_chosen_logps, policy_rejected_logps, sft_chosen_logps, sft_reject_logps, yw_margin_reward, yl_margin_reward)
            # print(f'loss_dpo:{loss_dpo}')

            # 获取策略模型对 yw 和 yl 的预测概率的对数
            # log_policy_yw = torch.log(policy_probs.gather(dim=-1, index=yw.unsqueeze(-1)))
            # log_policy_yl = torch.log(policy_probs.gather(dim=-1, index=yl.unsqueeze(-1)))


            if batch_idx % 10 == 0:
                print("{}/{} loss: {} chosen_reward: {} rejected_reward: {}".format(batch_idx, len(self.train_dataloader),
                                                                   loss_dpo.mean(), chosen_reward.mean(), rejected_reward.mean()))
            train_loss += loss_dpo.mean()
            loss_dpo.mean().backward()
            torch.nn.utils.clip_grad_value_(self.model_policy.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

        log = {'train_loss': train_loss / len(self.train_dataloader)}

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
                ## logit adjustment
            #     cls_labels = (cls_labels == 1).float()
            #     logit = cls_preds_logits * cls_labels
            #     logits.append(logit.cpu().numpy())
            #     counts.append(cls_labels.cpu().numpy())
            #
                val_res.extend(reports)
                val_gts.extend(ground_truths)
            #
            # #######
            # logits = np.concatenate(logits, axis=0)
            # counts = np.concatenate(counts, axis=0)
            # logits = np.sum(logits, 0)
            # counts = np.sum(counts, 0)
            # logits = logits / counts
            # logits /= np.max(logits)
            # logits = np.append(logits, [1, 1, 1, 1])  # 4 auxiliary diseases
            #######
            # self.base_probs = logits  # update class distribution
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_ce.items()})

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
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
        return log


