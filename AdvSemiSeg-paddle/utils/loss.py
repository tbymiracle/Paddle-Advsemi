import paddle
import paddle.nn.functional as F
import paddle.nn as nn

class CrossEntropy2d(nn.Layer):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.shape_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):

        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.shape[0] == target.shape[0], "{0} vs {1} ".format(predict.shape[0], target.shape[0])
        assert predict.shape[2] == target.shape[1], "{0} vs {1} ".format(predict.shape[2], target.shape[1])
        assert predict.shape[3] == target.shape[2], "{0} vs {1} ".format(predict.shape[3], target.shape[3])
        n, c, h, w = predict.shape
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        
        if not target.dim():
            return paddle.zeros(1)
        predict = predict.transpose((0,2,3,1))
        
        
        
        predict = predict[paddle.fluid.layers.expand(target_mask.reshape([n, h, w, 1]),[1, 1, 1, c])].reshape([-1, c])
        
        loss = F.cross_entropy(predict, target, weight=weight,axis=1)

        return loss


class BCEWithLogitsLoss2d(nn.Layer):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.shape_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # assert not target.requires_grad
        
        reprod_logger.add("predict{i}", predict.cpu().detach().numpy())
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.shape[0] == target.shape[0], "{0} vs {1} ".format(predict.shape[0], target.shape[0])
        assert predict.shape[2] == target.shape[2], "{0} vs {1} ".format(predict.shape[2], target.shape[2])
        assert predict.shape[3] == target.shape[3], "{0} vs {1} ".format(predict.shape[3], target.shape[3])
        n, c, h, w = predict.shape
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.dim():
            return paddle.zeros(1)
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight)
        return loss
