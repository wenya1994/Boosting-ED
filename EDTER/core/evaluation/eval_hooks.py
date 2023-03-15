import os.path as osp

from mmcv.runner import Hook
from torch.utils.data import DataLoader
import cv2 as cv


class EvalHook(Hook):
    """Evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
    """

    def __init__(self, dataloader, interval=1, **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError('dataloader must be a pytorch DataLoader, but got '
                            f'{type(dataloader)}')
        self.dataloader = dataloader
        self.interval = interval
        self.eval_kwargs = eval_kwargs

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import single_gpu_test
        runner.log_buffer.clear()
        results = single_gpu_test(runner.model, self.dataloader, show=False)
        for idx, result in enumerate(results):
            image = result.squeeze()
            image_name = osp.join(r"D:\ywy\3.Dada&Code\Codes\1.Official_code\5.EDTER-main\results\bsds",
                                  str(idx) + ".png")
            cv.imwrite(image_name, image * 255)
        # self.evaluate(runner, results)

    def evaluate(self, runner, results):
        """Call evaluate function of dataset."""
        eval_res = self.dataloader.dataset.evaluate(
            results, logger=runner.logger, **self.eval_kwargs)
        for name, val in eval_res.items():
            runner.log_buffer.output[name] = val
        runner.log_buffer.ready = True


class DistEvalHook(EvalHook):
    """Distributed evaluation hook.

    Attributes:
        dataloader (DataLoader): A PyTorch dataloader.
        interval (int): Evaluation interval (by epochs). Default: 1.
        tmpdir (str | None): Temporary directory to save the results of all
            processes. Default: None.
        gpu_collect (bool): Whether to use gpu or cpu to collect results.
            Default: False.
    """

    def __init__(self,
                 dataloader,
                 interval=1,
                 gpu_collect=False,
                 **eval_kwargs):
        if not isinstance(dataloader, DataLoader):
            raise TypeError(
                'dataloader must be a pytorch DataLoader, but got {}'.format(
                    type(dataloader)))
        self.dataloader = dataloader
        self.interval = interval
        self.gpu_collect = gpu_collect
        self.eval_kwargs = eval_kwargs
        self.interval_num = 0

    def after_train_iter(self, runner):
        """After train epoch hook."""
        if not self.every_n_iters(runner, self.interval):
            return
        from mmseg.apis import multi_gpu_test
        runner.log_buffer.clear()
        self.interval_num = self.interval_num + 1
        self.iter_num = self.interval_num * self.interval
        multi_gpu_test(
            runner.model,
            self.dataloader,
            # tmpdir=osp.join(runner.work_dir, '.eval_hook'),
            tmpdir=osp.join(runner.work_dir),
            gpu_collect=self.gpu_collect,
            iterNum=self.iter_num)
        if runner.rank == 0:
            print('\n')
            # self.evaluate(runner, results)
