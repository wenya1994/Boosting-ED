import argparse

from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmseg.models import build_segmentor,build_segmentor_local8x8


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--config', type=str, default='../configs/bsds2/VIT_BIMLA_320x320_80k_bsds_local8x8_bs_8.py', help='train config file path')
    parser.add_argument('--global-config', type=str, default='../configs/bsds2/VIT_BIMLA_320x320_80k_bsds_bs_8.py', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[320, 320],
        help='input image size')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    global_cfg = Config.fromfile(args.global_config)
    cfg.model.pretrained = None
    #model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()
    model = build_segmentor_local8x8(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg, global_cfg=global_cfg).cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy2
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape)
    split_line = '=' * 30
    print('{0}\nInput shape: {1}\nFlops: {2}\nParams: {3}\n{0}'.format(
        split_line, input_shape, flops, params))
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


if __name__ == '__main__':
    main()
