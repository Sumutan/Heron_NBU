import mindspore as ms
from mindspore import ops, nn

class LabelSmoothingCrossEntropy(nn.LossBase):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

        self.GatherD = ops.GatherD()
        self.ExpandDims = ops.ExpandDims()
        self.LogSoftmax = nn.LogSoftmax(-1)
    
    def construct(self, x, target):
        logprobs = self.LogSoftmax(x)
        nll_loss = self.GatherD(-logprobs, -1, self.ExpandDims(target, 1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(axis=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class SoftTargetCrossEntropy(nn.LossBase):

    def __init__(self):
        super(SoftTargetCrossEntropy, self).__init__()
        self.log_softmax = nn.LogSoftmax(axis=-1)
        self.sum = ops.ReduceSum()

    def construct(self, x, target):
        loss = self.sum(-target * self.log_softmax(x), -1)
        return loss.mean()
