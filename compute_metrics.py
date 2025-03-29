from modules.metrics_clinical import CheXbertMetrics

from modules.metrics import compute_scores
import pandas as pd

import logging
import os
from abc import abstractmethod
import numpy as np
import time
from radgraph import F1RadGraph
import cv2
import torch
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from concurrent.futures import ThreadPoolExecutor
import sys
import time
sys.path.append('/home/shilei/project/RadCLIQ-CXR')
from test_metric import cal_rad_cliq
# from green_score import GREEN

# model_name = "StanfordAIMI/GREEN-radllama2-7b"
#
# green_scorer = GREEN(model_name, output_dir=".")
# BATCH_SIZE = 16
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = torch.device("cuda:0")
# chexbert_metrics = CheXbertMetrics('/ssd/shilei/dataset/checkpoints/chexbert.pth', BATCH_SIZE, device)
Bleu_scorer = Bleu(4)
Rouge_scorer = Rouge()
Meteor_scorer = Meteor()
#
# res_cmn_path = "data_green/res_2.csv"
# path_gt = "process_data/test_win_lose_all.csv"
# res_mrg_path = "mrg_mimic_train.csv"
# minigpt_path = "data_green/reports_minigpt_20000_final.xlsx"
# path_test = "test_result/mrg_mimic_test.csv"
# path_train = "test_result/mrg_mimic_train.csv"
path = "/home/shilei/project/PromptMRG/data_green/FINAL/combined_green_no_duplicates_final2.csv"
path2 = "/home/shilei/project/PromptMRG/test_result/mimic_test_cliq_new_0.1_0.9.csv"
path3 = "/home/shilei/project/PromptMRG/data_green/GPT-4o/green_prefer_10000.csv"
df = pd.read_csv(path)
df2 = pd.read_csv(path2)
df3 = pd.read_csv(path3)
ids = df3['id'].values

# report_gts = df['report_gt'].values
# report_yls = df['y_l'].values
# report_yws = df['y_w'].values
report_gts = df3['report_gt'].values

report_res = df3['reject_report'].values
# ids = date_test['id'].values
#
# report_gts = date_test['report_gt']

# report_gts = pd.read_csv(res_cmn_path)['report_gt'].values
# report_cmn = pd.read_csv(res_cmn_path)['report_res'].values
#
# ids = pd.read_csv(res_cmn_path)['id'].values
# # report_mrg = pd.read_csv(res_mrg_path)['report_res'].values
# report_gts = pd.read_csv(res_cmn_path)['report_gt'].values[0: len(report_cmn)]
# report_gts = pd.read_excel(minigpt_path)['report_gt'].values
#
# report_1 = pd.read_excel(minigpt_path)['report1'].values
# report_2 = pd.read_excel(minigpt_path)['report2'].values
# ids = pd.read_excel(minigpt_path)['id'].values
# print(ids)
# report_gts = ["as compared to the previous radiograph no relevant change is seen of the sternal wiring . monitoring and support devices are constant in appearance . constant low lung volumes with bilateral small pleural effusions and subsequent areas of atelectasis . moderate cardiomegaly . no new parenchymal opacities ."]
# # report_res = ["While the previous radiograph displayed stable sternal wiring, there has been no change in the device positioning. The lungs maintain their reduced volume, with bilateral pleural effusions and areas of atelectasis observed, particularly at the bases. Cardiomegaly persists in a moderate form. The parenchymal tissue shows no new opacities, remaining consistent with earlier findings."]
# report_res = ["In comparison with the prior radiograph, the sternal wiring alignment is consistent with previous images. Monitoring and support devices show no changes. Bilateral small pleural effusions are still present, along with atelectasis in both lower lung regions. The cardiomegaly remains moderate. No new parenchymal abnormalities are detected."]
# # report_res = ["The sternal wiring from the last radiograph is intact, with no shifts in position. The lung fields are smaller in size, and there is fluid accumulation in both pleural spaces. The heart continues to appear enlarged in a moderate degree. There are no signs of new lung consolidations, and the previous atelectasis remains present."]
# # report_res = ["Reviewing the current images, the sternal wires have not moved, and the lung size seems reduced. There are ongoing pleural effusions, and some regions of atelectasis are still visible, particularly at the bases. The heart appears slightly larger than normal. No new opacity was detected in the lung parenchyma."]
# # report_res = ["The position of the sternal wiring remains unchanged compared to the previous radiograph. Bilateral pleural effusions are still visible, and atelectasis continues to affect the lung bases. The cardiomegaly remains moderate. No new parenchymal opacity or consolidation is seen, and the overall lung structure appears stable."]
# #cliq: 1.944899588855694 radf1: 0.46341463414634143
# # report_res = ["Upon comparison with the prior radiograph, there are no changes in the sternal wiring position. Pleural effusions remain bilateral, and atelectasis is noted in both lower lobes. The heart retains its moderate enlargement. No new consolidation is present in the lung parenchyma, and the overall lung pattern remains consistent with earlier imaging."]
# #cliq: 2.4413666064878354 radf1: 0.34883720930232553
# # report_res = ["The sternal wires are in their original position with no evidence of movement. Bilateral pleural effusions are still present, and the lungs display decreased volume with atelectasis at the bases. The heart shows moderate enlargement, consistent with prior imaging. No new areas of consolidation or opacity are noted in the lung fields."]
# # report_res = ["Upon reviewing the radiograph, the sternal wires are correctly positioned with no movement. The bilateral pleural effusions are stable, with atelectasis still visible at the lung bases. The cardiomegaly is moderate, similar to the prior study. No new parenchymal opacity is seen, and the lungs appear stable overall."]
# #cliq: 2.1760363361348776 radf1: 0.46341463414634143
# # report_gts = ["single portable view of the chest was compared to previous exam from  . given differences in positioning and technique compared to prior there has been no significant interval change . there is no evidence of confluent consolidation or pulmonary vascular congestion . there is no large pleural effusion . cardiac silhouette is stable ."]
# report_res = ["Compared to previous studies, there is no change in sternal wiring. The monitoring and support devices have shifted slightly. Low lung volumes with small pleural effusions are still present but only on the right side. Moderate cardiomegaly is noted, and there are no new parenchymal opacities. There appears to be an old fracture of the clavicle which is notable."]
# report_res = ["sternal wiring significantly altered from prior radiograph. changes to monitoring and support devices are noted. high lung volumes with unilateral pleural effusion and significant atelectasis. severe cardiomegaly evident. multiple new parenchymal opacities visible."]
# report_res = ["sternal wiring shows no significant changes compared to previous radiograph. stable appearances of monitoring and support devices. persistent low lung volumes with bilateral small pleural effusions and resulting atelectasis. moderate cardiomegaly is present. no new areas of lung opacities detected."]
# report_res = ["As compared to the previous radiograph, there is no significant change in the chest. Mild cardiomegaly with bilateral pleural effusions and slight pulmonary edema. Areas of atelectasis are noted in the right lung base. The left internal jugular vein catheter remains unchanged. No pneumothorax detected. The sternal wires are in altered alignment compared to previous radiographs. The left pectoral port-a-cath is now slightly repositioned."]
# report_res = ["as compared to the previous radiograph there is no relevant change. moderate cardiomegaly with bilateral pleural effusions and moderate pulmonary edema. areas of atelectasis at both lung bases. the right internal jugular vein catheter is unchanged. no pneumothorax. the sternal wires are in unchanged alignment. unchanged position of the right pectoral port - a - cath."]
# report_res = ["A single portable view of the chest was compared to prior studies. Despite differences in positioning and technique, there has been no significant interval change. No evidence of confluent consolidation, pulmonary vascular congestion, or large pleural effusion is noted. The cardiac silhouette remains stable."]
# report_res = ["A single frontal view of the chest was compared to a previous exam from a different facility. There has been a noticeable increase in pulmonary vascular congestion. A large pleural effusion is present on the left. The cardiac silhouette appears enlarged. Additionally, there are signs of confluent consolidation in the right lung."]
# report_res = ["When compared to the prior radiograph, no significant changes are noted in the sternal wiring, which remains in its original position. However, the monitoring devices are slightly different than before. The lung volume is still low, but the pleural effusions seem more pronounced than previously recorded. Additionally, new parenchymal opacities were detected that were not present in the earlier radiograph, suggesting an evolution of the condition. The heart is also noted to be slightly enlarged, a change that may indicate worsening cardiomegaly."]
# report_res = ["When compared to the previous radiograph, there are new findings regarding the sternal wiring, which appears slightly altered. The monitoring devices are reported as being consistent with prior studies, but their exact positioning differs from the previous assessment. Additionally, there seems to be an issue with the lung volumes, which are described as constant, although the pleural effusions appear more extensive now. The cardiomegaly, which was previously moderate, is now noted as being more severe, suggesting a possible progression in the condition. However, no mention is made of a previous comparison that would detail any changes in the sternal wiring or lung volumes, leading to an omission of crucial reference information."]
# # report_res = ["No major differences were seen between this radiograph and the last, particularly regarding the sternal wiring. Monitoring devices and support structures remain unchanged in appearance. The lungs still show low volume, with bilateral pleural effusions and atelectasis in certain areas. Cardiomegaly is present at a moderate degree. Importantly, there are no signs of new parenchymal opacities or significant changes to the pulmonary structures."]
# # report_res = ["The radiograph comparison shows that there is no notable change in the sternal wiring, although it was suggested that a change had occurred. The monitoring and support devices are reported as unchanged, but an important device that was visible in the previous study is now omitted from this assessment. The lung volumes appear low as before, but the pleural effusions are described as absent, which contradicts the findings in the reference. Furthermore, while the report mentions no new parenchymal opacities, a new opacity was indeed visible in the current study. The assessment of cardiomegaly appears to be understated, with no mention of its moderate enlargement as previously observed."]
# report_res = ["Upon comparison with the previous radiograph, the sternal wiring appears unchanged, although it is noted to have slight adjustments that were not present before. The monitoring and support devices remain consistent with the earlier study, but one of the support devices seems to have been overlooked in the current assessment. The lung volumes are described as low, but the pleural effusions are no longer seen in the current image, contrary to the findings in the reference. Additionally, a mild cardiomegaly is mentioned, but it was previously assessed as moderate, a change in severity that was not accounted for in the report. Finally, there is no mention of a prior comparison to assess the progression of the atelectasis, a critical piece of information that is missing from the current report."]
# report_res = ["In the current radiograph, there is no observable change in the position of the sternal fixation hardware when compared with the previous study. The devices used for monitoring and providing support are identical in appearance to those in the past. The lung fields continue to display low volumes, with bilateral small pleural effusions seen along with localized areas of collapsed lung tissue. The heart shows a moderate enlargement, while no novel opacities are found in the lung tissue, suggesting stability in the condition."]
# report_res = ["When comparing this radiograph to the prior one, there is no significant change in the sternal wiring. However, the monitoring devices and support structures appear slightly different than those in the previous imaging. The lung volumes continue to be low, with bilateral pleural effusions, though these are noted to be more substantial now. Additionally, areas of atelectasis are present, which were not explicitly mentioned in the prior report. The degree of cardiomegaly appears more pronounced in this image, yet it was previously assessed as moderate. There is also no reference to a comparison that would track changes in lung volumes, which is an important omission."]
# report_res = ["Upon reviewing the current radiograph, there is no major alteration in the sternal wiring when compared to the earlier study. The monitoring and support devices are consistent in appearance, although one device is noted to be absent in this image. The lung volume remains low, with bilateral small pleural effusions, but these are slightly more noticeable in the current assessment. Additionally, some regions of the lungs show collapse, which were not referenced in the prior report. Cardiomegaly is noted to be of a moderate degree, though its presence is downplayed in comparison to the earlier assessment. Lastly, no mention is made of a previous study to assess changes in the pulmonary condition, which leaves out important historical comparison data."]
# report_res = ["In comparison to the prior radiograph, the sternal wiring appears stable, without notable changes. The monitoring and support devices remain unchanged, although one device that was present in the earlier image seems absent in this report. The lung volumes are still low, with small pleural effusions bilaterally. There is an indication of mild atelectasis in some regions, though these findings were not fully detailed in the reference. The heart continues to show mild enlargement, and no new opacities are observed in the lung fields."]

n = len(report_gts)
batch_size = 1000  # 批次大小
# print(f"gt_len : {len(report_gts)} cmn_len : {len(report_cmn)} mrg_len : {len(report_mrg)}")
batches = [
    [
        report_gts[i:min(i + batch_size, n)],  # 确保切片结束索引不会超过总长度 n
        report_res[i:min(i + batch_size, n)],
    ]
    for i in range(0, n, batch_size)
]
# print(len(batches[0]))
cmn_b1, cmn_b4, cmn_rg, cmn_f1, cmn_cliqs, cmn_green = [], [], [], [], [], []
mrg_b1, mrg_b4, mrg_rg, mrg_f1, mrg_cliqs, mrg_green = [], [], [], [], [], []
def process_batch(batch_data):
    results = []
    for i, data in enumerate(batch_data):
        # 每个数据项的处理逻辑
        print(f'i==={i}')
        log1, log2 = dict(), dict()
        # if i > 0:break
        # print(f'data:{len(data)}')
        test_gt, test_cmn= data[0].tolist(), data[1].tolist()
        # _, _, green_score_list_cmn, _, _ = green_scorer(test_gt, test_cmn)
        # _, _, green_score_list_mrg, _, _ = green_scorer(test_gt, test_mrg)
        # cmn_green.extend(green_score_list_cmn)
        # mrg_green.extend(green_score_list_mrg)
        # print(f'type_cmn:{type(test_cmn)} \n cmn:{test_cmn}')
        gts_ = {i: [test_gt[i]] for i in range(len(test_gt))}

        cmn_ = {i: [test_cmn[i]] for i in range(len(test_cmn))}

        # mrg_ = {i : [test_mrg[i]] for i in range(len(test_mrg))}

        t1 = time.time()
        _, cmn_bleu_scores = Bleu_scorer.compute_score(gts_, cmn_, verbose = 0)
        # _, mrg_bleu_scores = Bleu_scorer.compute_score(gts_, mrg_, verbose=0)
        cmn_b1.extend(cmn_bleu_scores[0])
        cmn_b4.extend(cmn_bleu_scores[3])
        # mrg_b1.extend(mrg_bleu_scores[0])
        # mrg_b4.extend(mrg_bleu_scores[3])

        _, cmn_rouge_scores = Rouge_scorer.compute_score(gts_, cmn_)
        # _, mrg_rouge_scores = Rouge_scorer.compute_score(gts_, mrg_)
        cmn_rg.extend(cmn_rouge_scores)
        # mrg_rg.extend(mrg_rouge_scores)
        f1radgraph = F1RadGraph(model_type="radgraph", reward_level="all", cuda=torch.device("cuda:0"))
        # cmn_reward
        mean_reward_cmn, reward_list_cmn, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
            hyps=test_cmn, refs=test_gt)

        # mrg_reward
        # mean_reward_mrg, reward_list_mrg, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
        #     hyps=test_mrg, refs=test_gt)

        reward_list_cmn = [np.mean([reward_list_cmn[0][i], reward_list_cmn[1][i], reward_list_cmn[2][i]]) for i in range(len(reward_list_cmn[0]))]
        # reward_list_mrg = [np.mean([reward_list_mrg[0][i], reward_list_mrg[1][i], reward_list_mrg[2][i]]) for i in range(len(reward_list_mrg[0]))]

        cmn_f1.extend(reward_list_cmn)
        # mrg_f1.extend(reward_list_mrg)
        cmn_df = pd.DataFrame({'study_id': range(1, len(test_cmn) + 1), 'report': test_cmn})
        gt_df = pd.DataFrame({'study_id': range(1, len(test_gt) + 1), 'report': test_gt})
        # mrg_df = pd.DataFrame({'study_id': range(1, len(test_mrg) + 1), 'report': test_mrg})

        # 保存为 CSV 文件
        cmn_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
        gt_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/gt.csv', index=False)
        cmn_cliq = cal_rad_cliq()
        # mrg_df.to_csv('/home/shilei/project/RadCLIQ-CXR/data/res.csv', index=False)
        # mrg_cliq = cal_rad_cliq()

        t2 = time.time()
        cmn_cliqs.extend(cmn_cliq)

        # mrg_cliqs.extend(mrg_cliq)
        print(f'i============================================={i}')
        print(f'time:{t2 - t1} :.4f')
process_batch(batch_data=batches)
print('cliq:', 1.0 * sum(cmn_cliqs) / len(cmn_cliqs))
print('radf1:', 1.0 * sum(cmn_f1) / len(cmn_f1))
df = pd.DataFrame({
    'id' : ids,
    'y_l_b1' : cmn_b1,
    'y_l_b4' : cmn_b4,
    'y_l_rg' : cmn_rg,
    'y_l_cliq' : cmn_cliqs,
    'y_l_f1' : cmn_f1,
    'y_l' : report_res,
    'report_gt' : report_gts
}).to_csv("/home/shilei/project/PromptMRG/test_result/0121_test_gpt_4o_reject_cliq.csv")



