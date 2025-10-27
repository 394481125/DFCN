#!/usr/bin/env python
"""
For evaluation
Extended from ADNet code by Hansen et al.
"""
import shutil
import SimpleITK as sitk
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from models.DFCN import FewShotSeg
from dataloaders.datasets import TestDataset
from dataloaders.dataset_specifics import *
from utils import *
from config import ex
import numpy as np
from scipy.ndimage import distance_transform_edt

# ==========================================================================================
#                         新增的评估指标计算函数
# ==========================================================================================

def surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    计算两个分割体素之间的表面距离。

    参数:
    - result: numpy array, 预测的分割掩码
    - reference: numpy array, 真实的分割掩码
    - voxelspacing: tuple, 体素间距 (e.g., (1.0, 1.0, 1.0))
    - connectivity: int, 连通性

    返回:
    - (dist_gt_to_pred, dist_pred_to_gt)
    """
    result = np.atleast_1d(result.astype(np.bool_))
    reference = np.atleast_1d(reference.astype(np.bool_))

    # 提取表面
    result_border = result ^ (result & np.roll(result, 1, axis=0) & np.roll(result, 1, axis=1) & np.roll(result, 1, axis=2)
                               & np.roll(result, -1, axis=0) & np.roll(result, -1, axis=1) & np.roll(result, -1, axis=2))
    reference_border = reference ^ (reference & np.roll(reference, 1, axis=0) & np.roll(reference, 1, axis=1) & np.roll(reference, 1, axis=2)
                                   & np.roll(reference, -1, axis=0) & np.roll(reference, -1, axis=1) & np.roll(reference, -1, axis=2))


    # 如果没有前景，返回空数组
    if not np.any(result_border) or not np.any(reference_border):
        return np.array([]), np.array([])


    # 计算欧氏距离变换
    dist_to_result_border = distance_transform_edt(~result_border, sampling=voxelspacing)
    dist_to_reference_border = distance_transform_edt(~reference_border, sampling=voxelspacing)

    # 从参考表面到结果表面的距离
    dists_ref_to_res = dist_to_result_border[reference_border]

    # 从结果表面到参考表面的距离
    dists_res_to_ref = dist_to_reference_border[result_border]

    return dists_ref_to_res, dists_res_to_ref


def hd95(result, reference, voxelspacing=None):
    """
    计算95%的豪斯多夫距离
    """
    res_to_ref, ref_to_res = surface_distances(result, reference, voxelspacing)

    if res_to_ref.size == 0 or ref_to_res.size == 0:
        return np.nan

    return max(np.percentile(res_to_ref, 95), np.percentile(ref_to_res, 95))

def asd(result, reference, voxelspacing=None):
    """
    计算平均表面距离
    """
    res_to_ref, ref_to_res = surface_distances(result, reference, voxelspacing)
    if res_to_ref.size == 0 or ref_to_res.size == 0:
        return np.nan
    return np.mean(ref_to_res)


def assd(result, reference, voxelspacing=None):
    """
    计算对称平均表面距离
    """
    res_to_ref, ref_to_res = surface_distances(result, reference, voxelspacing)
    if res_to_ref.size == 0 or ref_to_res.size == 0:
        return np.nan
    return (np.mean(res_to_ref) + np.mean(ref_to_res)) / 2.0


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model_config = {
        'dataset': _config['dataset'],
        'PREC': _config['PREC'],
        'BACKBONE_NAME': _config['BACKBONE_NAME'],
        'N_CTX': _config['N_CTX'],
        'CTX_INIT': _config['CTX_INIT'],
        'CLASS_TOKEN_POSITION': _config['CLASS_TOKEN_POSITION'],
        'INPUT_SIZE': _config['INPUT_SIZE'],
        'CSC': _config['CSC'],
        'INIT_WEIGHTS': _config['INIT_WEIGHTS'],
        'OPTIM': _config['OPTIM'],
        'PROMPT_INIT': _config['PROMPT_INIT'],
    }
    model = FewShotSeg(model_config)
    model.cuda()
    model.load_state_dict(torch.load(_config['reload_model_path'], map_location='cpu'), strict=False)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'supp_idx': _config['supp_idx'],
    }
    test_dataset = TestDataset(data_config)
    test_loader = DataLoader(test_dataset,
                             batch_size=_config['batch_size'],
                             shuffle=True,
                             num_workers=_config['num_workers'],
                             pin_memory=True,
                             drop_last=True)

    labels = get_label_names(_config['dataset'])

    class_dice = {}
    class_iou = {}
    # ==================================================================
    #                为新的评估指标创建字典
    # ==================================================================
    class_hd95 = {}
    class_asd = {}
    class_assd = {}


    _log.info(f'Starting validation...')
    for label_val, label_name in labels.items():

        if label_name == 'BG':
            continue
        elif np.intersect1d([label_val], _config['test_label']).size == 0:
            continue

        _log.info(f'Test Class: {label_name}')

        support_sample = test_dataset.getSupport(label=label_val, all_slices=False, N=_config['n_part'])

        test_dataset.label = label_val

        with torch.no_grad():
            model.eval()

            support_image = [support_sample['image'][[i]].float().cuda() for i in
                             range(support_sample['image'].shape[0])]
            support_fg_mask = [support_sample['label'][[i]].float().cuda() for i in
                               range(support_sample['image'].shape[0])]

            scores = Scores()
            # ==================================================================
            #                   为每个类别初始化指标列表
            # ==================================================================
            patient_hd95 = []
            patient_asd = []
            patient_assd = []

            for i, sample in enumerate(test_loader):

                query_image = [sample['image'][i].float().cuda() for i in
                               range(sample['image'].shape[0])]
                query_label = sample['label'].long()
                query_id = sample['id'][0].split('image_')[1][:-len('.nii.gz')]


                query_pred = torch.zeros(query_label.shape[-3:])
                C_q = sample['image'].shape[1]

                idx_ = np.linspace(0, C_q, _config['n_part'] + 1).astype('int')
                for sub_chunck in range(_config['n_part']):
                    support_image_s = [support_image[sub_chunck]]
                    support_fg_mask_s = [support_fg_mask[sub_chunck]]
                    query_image_s = query_image[0][idx_[sub_chunck]:idx_[sub_chunck + 1]]
                    query_pred_s = []
                    for i_slice in range(query_image_s.shape[0]):
                        _pred_s, _, _, _, _ = model([support_image_s], [support_fg_mask_s], [query_image_s[[i_slice]]],
                                                    _, _, train=False)
                        query_pred_s.append(_pred_s)
                    query_pred_s = torch.cat(query_pred_s, dim=0)
                    query_pred_s = query_pred_s.argmax(dim=1).cpu()
                    query_pred[idx_[sub_chunck]:idx_[sub_chunck + 1]] = query_pred_s

                scores.record(query_pred, query_label)

                # ==================================================================
                #                       计算并记录新的指标
                # ==================================================================
                # 将Tensor转换为numpy数组
                query_pred_np = query_pred.numpy().astype(bool)
                query_label_np = query_label.squeeze().numpy().astype(bool)

                # 计算表面距离指标
                patient_hd95.append(hd95(query_pred_np, query_label_np))
                patient_asd.append(asd(query_pred_np, query_label_np))
                patient_assd.append(assd(query_pred_np, query_label_np))


                _log.info(
                    f'Tested query volume: {sample["id"][0][len(_config["path"][_config["dataset"]]["data_dir"]):]}.')
                _log.info(f'Dice score: {scores.patient_dice[-1].item()}')
                _log.info(f'HD95: {patient_hd95[-1]}')
                _log.info(f'ASD: {patient_asd[-1]}')
                _log.info(f'ASSD: {patient_assd[-1]}')


                file_name = os.path.join(f'{_run.observers[0].dir}/interm_preds',
                                         f'prediction_{query_id}_{label_name}.nii.gz')
                itk_pred = sitk.GetImageFromArray(query_pred)
                sitk.WriteImage(itk_pred, file_name, True)
                _log.info(f'{query_id} has been saved. ')

            class_dice[label_name] = torch.tensor(scores.patient_dice).mean().item()
            class_iou[label_name] = torch.tensor(scores.patient_iou).mean().item()
            # ==================================================================
            #                       计算并记录类别平均指标
            # ==================================================================
            class_hd95[label_name] = np.nanmean(patient_hd95)
            class_asd[label_name] = np.nanmean(patient_asd)
            class_assd[label_name] = np.nanmean(patient_assd)


            _log.info(f'Test Class: {label_name}')
            _log.info(f'Mean class IoU: {class_iou[label_name]}')
            _log.info(f'Mean class Dice: {class_dice[label_name]}')
            _log.info(f'Mean class HD95: {class_hd95[label_name]}')
            _log.info(f'Mean class ASD: {class_asd[label_name]}')
            _log.info(f'Mean class ASSD: {class_assd[label_name]}')

    _log.info(f'Final results...')
    _log.info(f'Mean IoU: {class_iou}')
    _log.info(f'Mean Dice: {class_dice}')
    _log.info(f'Mean HD95: {class_hd95}')
    _log.info(f'Mean ASD: {class_asd}')
    _log.info(f'Mean ASSD: {class_assd}')


    def dict_Avg(Dict):
        L = len(Dict)
        if L == 0:
            return 0
        S = sum(Dict.values())
        A = S / L
        return A

    value = dict_Avg(class_dice)
    with open('results.txt', 'w') as file:
        file.write(str(value))

    _log.info(f'Whole mean Dice: {dict_Avg(class_dice)}')
    _log.info(f'Whole mean HD95: {dict_Avg(class_hd95)}')
    _log.info(f'Whole mean ASD: {dict_Avg(class_asd)}')
    _log.info(f'Whole mean ASSD: {dict_Avg(class_assd)}')

    _log.info(f'End of validation.')
    return 1