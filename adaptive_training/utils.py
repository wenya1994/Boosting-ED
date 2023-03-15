import torch.nn.functional as F
import torch


def loss_fn(i, prediction, label):
    lamda = [1., 1., 1., 1., 1., 1.]
    cost = F.binary_cross_entropy_with_logits(prediction, label)  # [20, 256, 256] [20, 256, 256] type == Tensor
    return cost * lamda[i]


def Cross_entropy_loss(prediction, label):
    # lamda = [1., 1., 1., 1., 1., 1., 1.] # 给不同的通道分配权重
    # lamda = [1.3, 0.9, 0.7, 0.6, 0.5, 2., 1.]
    lamda = [0.8, 1.2, 1.0, 0.6, 0.4, 2., 1.]

    mask = label.clone()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0

    cost = F.binary_cross_entropy_with_logits(prediction, label, weight=mask, reduction='sum')
    return cost


# output-level 蒸馏损失
def Disloss_output_level(stu_pre, tec_pre):
    cost = F.binary_cross_entropy_with_logits(stu_pre, tec_pre)  # [20, 256, 256] [20, 256, 256] type == Tensor
    return cost


# feature-level 蒸馏损失
def Disloss_feature_level(stu_feature, tec_feature):
    """
        Compute the distillation loss between the previous feature maps and the current ones

        :param new_features: output of the feature maps of the current network. 4D tensor: [batch_size, 41, 41, 2048]
        :param old_features: output of the feature maps of the previous network. 4D tensor: [batch_size, 41, 41, 2048]
        :return: distillation loss on the feature space
     """
    # Compute the cross entropy loss
    loss = F.mse_loss(labels=tec_feature, predictions=stu_feature)
    # loss = tf.reduce_mean(loss)
    return loss
