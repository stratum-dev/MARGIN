import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel


# ==================== 模型定义 ====================
class MARGINModel(nn.Module):
    def __init__(self, num_classes, backbone: str, embedding_dim: int, scale):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(backbone)
        self.scale = scale
        self.num_classes = num_classes

        # Weight prototypes (已归一化)
        self.weight_prototypes = nn.Parameter(
            F.normalize(torch.randn(num_classes, embedding_dim), p=2, dim=1)
        )

        # 冻结roberta的部分层（可选优化）
        # for param in self.roberta.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_ids, attention_mask, return_features=False):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # 提取CLS token特征
        features = outputs.last_hidden_state[:, 0, :]  # [B, D]
        # L2归一化到单位超球面
        features = F.normalize(features, p=2, dim=1)  # [B, D]

        # 计算余弦相似度 (球面内积)
        cos_theta = torch.matmul(features, self.weight_prototypes.t())  # [B, C]

        if return_features:
            return cos_theta, features
        return cos_theta

    def get_weight_prototypes(self):
        """返回归一化的weight prototypes"""
        return F.normalize(self.weight_prototypes, p=2, dim=1)


# ==================== ArcFace Loss with Adaptive Margin ====================
class MARGINLossHead(nn.Module):
    def __init__(self, num_classes, scale):
        super().__init__()
        self.scale = scale
        self.num_classes = num_classes
        self.register_buffer("margins", torch.zeros(num_classes))
        self.register_buffer("kappas", torch.ones(num_classes) * 1)

    def update_margins(self, margins_dict):
        """更新各类别的margin，margins_dict: {class_idx: margin_value}"""
        for idx, margin in margins_dict.items():
            self.margins[idx] = margin

    def update_kappas(self, kappas_dict):
        """更新各类别的kappa，kappas_dict: {class_idx: kappa_value}"""
        for idx, kappa in kappas_dict.items():
            self.kappas[idx] = kappa

    def forward(self, cos_theta, labels):
        """
        cos_theta: [B, C] 余弦相似度
        labels: [B] 类别索引
        """
        batch_size = cos_theta.size(0)

        # 获取对应类别的margin
        margins_batch = self.margins[labels]  # [B]

        # 计算 theta = arccos(cos_theta)，需要数值稳定性
        cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta_clamped)

        # 应用margin: cos(theta + m)
        # 使用cos加法公式: cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
        cos_m = torch.cos(margins_batch)
        sin_m = torch.sin(margins_batch)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta_clamped, 2))

        # 只对正类应用margin
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        cos_theta_plus_m = cos_theta_clamped * cos_m.unsqueeze(
            1
        ) - sin_theta * sin_m.unsqueeze(1)

        # 混合：正类用cos(theta+m)，负类保持cos(theta)
        output = cos_theta_clamped * (1.0 - one_hot) + cos_theta_plus_m * one_hot

        # 缩放
        output = output * self.scale

        # 交叉熵
        loss = F.cross_entropy(output, labels)

        return loss
