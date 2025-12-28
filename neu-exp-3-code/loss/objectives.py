import torch
import torch.nn as nn
import torch.nn.functional as F

class Objectives(nn.Module):
    def __init__(self, lambda_logic=1.0, lambda_align=1.0):
        super(Objectives, self).__init__()
        self.lambda_logic = lambda_logic
        self.lambda_align = lambda_align
        self.ce_loss = nn.CrossEntropyLoss()

    def classification_loss(self, logits, targets):
        return self.ce_loss(logits, targets)

    def logic_consistency_loss(self, predicates, rules):
        return F.mse_loss(predicates, rules)

    def alignment_loss(self, predicates, ground_truth):
        return F.mse_loss(predicates, ground_truth)

    def forward(self, logits, targets, predicates, rules, ground_truth):
        l_cls = self.classification_loss(logits, targets)
        l_logic = self.logic_consistency_loss(predicates, rules)
        l_align = self.alignment_loss(predicates, ground_truth)
        total_loss = l_cls + self.lambda_logic * l_logic + self.lambda_align * l_align
        return total_loss, {
            "classification_loss": l_cls.item(),
            "logic_loss": l_logic.item(),
            "alignment_loss": l_align.item(),
            "total_loss": total_loss.item()
        }