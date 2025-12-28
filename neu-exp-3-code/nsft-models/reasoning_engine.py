import torch

class ReasoningEngine:
    def __init__(self, rule_set=None):
        self.rule_set = rule_set if rule_set is not None else []

    def evaluate(self, predicate_batch, predicate_names):
        batch_size = predicate_batch.shape[0]
        results = []
        for b in range(batch_size):
            pred_dict = {name: predicate_batch[b, i].item() for i, name in enumerate(predicate_names)}
            row = []
            for rule in self.rule_set:
                row.append(float(rule(pred_dict)))
            results.append(row)
        return torch.tensor(results, dtype=torch.float32)