import argparse
import os
import mmcv
import torch
import cv2 as cv
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
import os.path as osp
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor, build_segmentor_local8x8

os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'


# os.environ['MASTER_ADDR']='127.0.0.2'
# os.environ['MASTER_PORT']='29502'

def parse_args():
    parser = argparse.ArgumentParser(description='mmseg test (and eval) a model')
    parser.add_argument('--globalconfig', type=str, default='../configs/bsds/EDTER_BIMLA_320x320_80k_bsds_bs_8.py',
                        help='train global config file path')
    parser.add_argument('--config', type=str, default='../configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py',
                        help='train local config file path')
    # '../configs/bsds/EDTER_BIMLA_320x320_80k_bsds_local8x8_bs_8.py'
    parser.add_argument('--checkpoint', type=str, default='../checkpoints/EDTER-BSDS-VOC-StageII.pth',
                        help='the dir of local model')
    parser.add_argument('--global-checkpoint', type=str,
                        default='../checkpoints/EDTER-BSDS-VOC-StageI.pth',
                        help='the dir of global model')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        # default=True,
        help='Format the output results without perform evaluation. It is'
             'useful when you want to format the result to a specific format and '
             'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
             ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved',
        type=str, default='')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir', type=str, default='../results/local_results',
        help='tmp directory used for collecting results from multiple '
             'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
           or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    global_cfg = mmcv.Config.fromfile(args.globalconfig)
    global_cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.globalconfig))[0])
    global_cfg.global_model_path = args.global_checkpoint
    model = build_segmentor_local8x8(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, global_cfg=global_cfg)

    print("Load Local Checkpoint from   =======>>>>   " + args.checkpoint)
    checkpoint_dict = torch.load(args.checkpoint, map_location='cpu')['state_dict']
    model_dict = model.state_dict()
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        outputs = single_gpu_test(model, data_loader, show=False, out_dir=None)
        for idx, result in enumerate(outputs):
            print(idx)
            image = result.squeeze()
            image_name = osp.join(r"D:\ywy\3.Dada&Code\Codes\1.Official_code\5.EDTER-main\results\nyud", str(idx) + ".png")
            cv.imwrite(image_name, image * 255)
    print('#' * 10 + 'finish test' + '#' * 10)


if __name__ == '__main__':
    main()
