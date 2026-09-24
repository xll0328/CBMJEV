"""Visible-only structured Choice scoring; not an independent-risk controller."""
import torch

from .choice_features import choice_feature_width, encode_choice_features
from .choice_targets import ChoiceInputs


class StructuredChoiceController:
    objective = "choice"

    def __init__(self, schema, head):
        if head.hidden_size != choice_feature_width(schema):
            raise ValueError("Choice head width does not match structured schema")
        if head.training or any(p.requires_grad for p in head.parameters()):
            raise ValueError("inference requires an eval-mode frozen Choice head")
        self.schema, self.head = schema, head

    @torch.no_grad()
    def predict_logits(self, observed, actions, *, remaining_groups, cost, cost_weight):
        if any(len(a) > 1 for a in actions):
            raise ValueError("this Choice protocol supports singleton acquisitions only")
        inputs = ChoiceInputs(tuple(observed), tuple(actions), remaining_groups,
            tuple(cost(observed, action) for action in actions), cost_weight)
        device = next(self.head.parameters()).device
        features = encode_choice_features(inputs, self.schema, device=device).unsqueeze(0)
        valid = torch.ones(features.shape[:2], dtype=torch.bool, device=device)
        return tuple(self.head(features, valid)[0].cpu().tolist())
