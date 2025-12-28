import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split

from models.transformer_encoder import TransformerEncoder
from models.symbolic_bridge import SymbolicBridge
from models.reasoning_engine import ReasoningEngine
from losses.objectives import Objectives
from evaluation.metrics import compute_metrics
from evaluation.explainability import Explainability

def generate_synthetic_data(num_samples=1000, seq_len=10, input_dim=128, num_classes=3):
    X = np.random.randn(num_samples, seq_len, input_dim).astype(np.float32)
    y = np.random.randint(0, num_classes, size=num_samples)
    return X, y

def train():
    num_classes = 3
    num_predicates = 5
    predicate_names = ["red_light", "lane_change", "speeding", "safe_driving", "collision_risk"]
    X, y = generate_synthetic_data(num_samples=500)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    encoder = TransformerEncoder(input_dim=128, embed_dim=256)
    bridge = SymbolicBridge(embed_dim=256, num_predicates=num_predicates)
    classifier = nn.Linear(num_predicates, num_classes)
    rules = [
        lambda p: (p["red_light"] > 0.5) and (p["speeding"] > 0.5),
        lambda p: (p["lane_change"] > 0.5) and (p["collision_risk"] > 0.5)
    ]
    engine = ReasoningEngine(rule_set=rules)
    objectives = Objectives(lambda_logic=0.5, lambda_align=0.5)
    params = list(encoder.parameters()) + list(bridge.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params, lr=1e-3)
    for epoch in range(5):
        encoder.train(); bridge.train(); classifier.train()
        X_batch = torch.tensor(X_train, dtype=torch.float32)
        y_batch = torch.tensor(y_train, dtype=torch.long)
        embeddings = encoder(X_batch)
        predicates = bridge(embeddings)
        logits = classifier(predicates)
        rules_tensor = engine.evaluate(predicates, predicate_names)
        ground_truth_preds = torch.zeros_like(predicates)
        loss, loss_dict = objectives(logits, y_batch, predicates, rules_tensor, ground_truth_preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Components: {loss_dict}")
    encoder.eval(); bridge.eval(); classifier.eval()
    X_eval = torch.tensor(X_test, dtype=torch.float32)
    y_eval = torch.tensor(y_test, dtype=torch.long)
    with torch.no_grad():
        embeddings = encoder(X_eval)
        preds = bridge(embeddings)
        logits = classifier(preds)
        y_pred = logits.argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(y_eval.numpy(), y_pred)
    print("\nFinal Evaluation Metrics:", metrics)
    explainer = Explainability(nn.Sequential(encoder, bridge, classifier))
    importance = explainer.attribute_features(torch.tensor(X_eval[:1], dtype=torch.float32))
    print("\nFeature Importance (first sample):", importance[:10])

if __name__ == "__main__":
    train()