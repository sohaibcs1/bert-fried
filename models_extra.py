import torch
import torch.nn as nn
from transformers import (
    XLNetModel, XLNetPreTrainedModel,
    RobertaModel, RobertaPreTrainedModel,
    GPT2Model, GPT2PreTrainedModel
)

#######################
# XLNet-based Model
#######################
class XLNetFRIDE(XLNetPreTrainedModel):
    def __init__(self, config, num_labels):
        super(XLNetFRIDE, self).__init__(config)
        self.num_labels = num_labels
        self.xlnet = XLNetModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.xlnet(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return {"loss": loss, "logits": logits}

#######################
# RoBERTa-based Model
#######################
class RoBERTaFRIDE(RobertaPreTrainedModel):
    def __init__(self, config, num_labels):
        super(RoBERTaFRIDE, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]  # pooled output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return {"loss": loss, "logits": logits}

#######################
# GPT-2-based Model
#######################
class GPT2FRIDE(GPT2PreTrainedModel):
    def __init__(self, config, num_labels):
        super(GPT2FRIDE, self).__init__(config)
        self.num_labels = num_labels
        self.gpt2 = GPT2Model(config)
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.classifier = nn.Linear(config.n_embd, num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
        return {"loss": loss, "logits": logits}
