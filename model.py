import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

def focal_loss(logits, targets, gamma=2.0, alpha=0.25):
    """
    Compute the focal loss between `logits` and the ground truth `targets`.
    
    Args:
        logits (Tensor): Raw output from the model (before sigmoid), shape (batch_size, num_labels)
        targets (Tensor): Ground truth labels, shape (batch_size, num_labels)
        gamma (float): Focusing parameter that reduces the relative loss for well-classified examples.
        alpha (float): Balance parameter to weight the loss for the positive class.
        
    Returns:
        Tensor: Computed focal loss.
    """
    # Compute standard binary cross-entropy loss with logits.
    BCE_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    # Convert the BCE loss to probabilities
    pt = torch.exp(-BCE_loss)
    # Compute the focal loss scaling factor
    focal_loss = alpha * (1 - pt) ** gamma * BCE_loss
    return focal_loss.mean()

class BertFRIDE(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(BertFRIDE, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]  # pooled [CLS] token output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = focal_loss(logits, labels.float(), gamma=2.0, alpha=0.25)
        return {"loss": loss, "logits": logits}
