import torch
import numpy as np

class Explainability:
    def __init__(self, model):
        self.model = model

    def attribute_features(self, inputs, target_class=None):
        inputs = inputs.clone().detach().requires_grad_(True)
        outputs = self.model(inputs)
        if target_class is None:
            target_class = outputs.argmax(dim=1)
        selected = outputs[0, target_class]
        selected.backward()
        importance = inputs.grad * inputs
        importance = importance.mean(dim=(0, 1)).detach().cpu().numpy()
        return importance / (np.linalg.norm(importance) + 1e-8)