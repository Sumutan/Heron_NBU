import logging as logger

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.boost import GradientAccumulation
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return ms.RowTensor(grad.indices, grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)), grad.dense_shape)


class GradAccumulationLossScaleCell(nn.TrainOneStepWithLossScaleCell):
    """construct train accu step cell"""

    def __init__(self, network, optimizer, scale_sense=None, accumulate_step=1):
        super(GradAccumulationLossScaleCell, self).__init__(network, optimizer, scale_sense)
        # 以下为梯度累积需要
        self.inner_grads = optimizer.parameters.clone(prefix="accumulate_", init='zeros')
        self.zeros = optimizer.parameters.clone(prefix="zeros_", init='zeros')
        self.counter = ms.Parameter(ms.Tensor(1, ms.int32), 'counter_')
        assert accumulate_step > 0
        self.accumulate_step = accumulate_step

    def construct(self, *inputs):
        weights = self.weights
        loss = self.network(*inputs)
        scaling_sens = self.scale_sense

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)

        self.hyper_map(ops.partial(ops.assign_add), self.inner_grads, grads)  # 梯度累积
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.counter % self.accumulate_step == 0:
                # 如果达到累积步数，进行参数优化更新
                loss = F.depend(loss, self.optimizer(grads))
                # 完成参数优化更新后，清零inner_grads
                self.hyper_map(ops.partial(ops.assign), self.inner_grads, self.zeros)
            # 计算步数加一
            ops.assign_add(self.counter, ms.Tensor(1, ms.int32))
        return loss, cond, scaling_sens


def create_train_one_step(args, net_with_loss, optimizer, log=logger):
    """get_train_one_step cell"""
    if args.use_dynamic_loss_scale:
        log.info(f"=> Using DynamicLossScaleUpdateCell")
        scale_manager = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000
        )
    else:
        log.info(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_manager = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)

    if args.accum_iter > 1:
        net_with_loss = GradAccumulationLossScaleCell(net_with_loss, optimizer, scale_sense=scale_manager, accumulate_step=args.accum_iter)
    else:
        net_with_loss = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scale_manager)
    return net_with_loss
