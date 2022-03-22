import logging

import torch
from torch import nn
from transformers import AutoModel

from rationalizers.explainers import available_explainers
from rationalizers.lightning_models.highlights.base import BaseRationalizer
from rationalizers.modules.scalar_mix import ScalarMixWithDropout
from rationalizers.utils import get_z_stats, freeze_module, masked_average

shell_logger = logging.getLogger(__name__)


class HFRationalizer(BaseRationalizer):
    """HuggingFaces' transformer-based rationalizer."""

    def __init__(
        self,
        tokenizer: object,
        nb_classes: int,
        is_multilabel: bool,
        h_params: dict,
    ):
        """
        :param tokenizer (object): torchnlp tokenizer object
        :param nb_classes (int): number of classes used to create the last layer
        :param multilabel (bool): whether the problem is multilabel or not (it depends on the dataset)
        :param h_params (dict): hyperparams dict. See docs for more info.
        """
        super().__init__(tokenizer, nb_classes, is_multilabel, h_params)

        self.gen_arch = h_params.get("gen_arch", "bert-base-multilingual-cased")
        self.pred_arch = h_params.get("pred_arch", "bert-base-multilingual-cased")
        self.gen_emb_requires_grad = h_params.get("gen_emb_requires_grad", True)
        self.pred_emb_requires_grad = h_params.get("pred_emb_requires_grad", True)
        self.gen_encoder_requires_grad = h_params.get("gen_encoder_requires_grad", True)
        self.pred_encoder_requires_grad = h_params.get("pred_encoder_requires_grad", True)
        self.shared_gen_pred = h_params.get("shared_gen_pred", False)
        self.use_scalar_mix = h_params.get("use_scalar_mix", True)
        self.dropout = h_params.get("dropout", 0.1)
        self.temperature = h_params.get("temperature", 1.0)
        self.selection_space = h_params.get("selection_space", 'embedding')
        self.selection_vector = h_params.get("selection_vector", 'mask')
        self.selection_faithfulness = h_params.get("selection_faithfulness", True)
        self.explainer_fn = h_params.get("explainer", True)
        self.explainer_pre_mlp = h_params.get("explainer_pre_mlp", True)
        self.mask_token_id = tokenizer.mask_token_id

        # generator module
        self.gen_hf = AutoModel.from_pretrained(self.gen_arch)
        self.gen_emb_layer = self.gen_hf.embeddings
        self.gen_encoder = self.gen_hf.encoder
        self.gen_hidden_size = self.gen_hf.config.hidden_size

        # predictor module
        self.pred_hf = self.gen_hf if self.shared_gen_pred else AutoModel.from_pretrained(self.pred_arch)
        self.pred_emb_layer = self.pred_hf.embeddings
        self.pred_encoder = self.pred_hf.encoder
        self.pred_hidden_size = self.pred_hf.config.hidden_size

        # explainer
        explainer_cls = available_explainers[self.explainer_fn]
        if self.explainer_pre_mlp is True:
            self.explainer = nn.Sequential(
                nn.Linear(self.gen_hidden_size, self.gen_hidden_size),
                nn.Tanh(),
                explainer_cls(h_params, self.gen_hidden_size)
            )
        else:
            self.explainer = explainer_cls(h_params, self.gen_hidden_size)

        # freeze embedding layers
        if not self.gen_emb_requires_grad:
            freeze_module(self.gen_emb_layer)
        if not self.pred_emb_requires_grad:
            freeze_module(self.pred_emb_layer)
        # freeze models
        if not self.gen_encoder_requires_grad:
            freeze_module(self.gen_encoder)
        if not self.pred_encoder_requires_grad:
            freeze_module(self.pred_encoder)

        # define output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.pred_hidden_size, self.pred_hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.pred_hidden_size, nb_classes),
            nn.Sigmoid() if not self.is_multilabel else nn.LogSoftmax(dim=-1),
        )

        # useful for stabilizing training
        self.pred_scalar_mix = ScalarMixWithDropout(
            mixture_size=self.pred_hf.config.num_hidden_layers+1,
            dropout=self.dropout,
            do_layer_norm=False,
        )
        if self.shared_gen_pred:
            self.gen_scalar_mix = self.pred_scalar_mix
        else:
            self.gen_scalar_mix = ScalarMixWithDropout(
                mixture_size=self.gen_hf.config.num_hidden_layers+1,
                dropout=self.dropout,
                do_layer_norm=False,
            )

        # initialize params using xavier initialization for weights and zero for biases
        self.init_weights(self.explainer)
        self.init_weights(self.output_layer)

    def forward(
        self, x: torch.LongTensor, current_epoch=None, mask: torch.BoolTensor = None
    ):
        """
        Compute forward-pass.

        :param x: input ids tensor. torch.LongTensor of shape [B, T]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: the output from SentimentPredictor. Torch.Tensor of shape [B, C]
        """
        ext_mask = mask[:, None, None, :]  # add head and seq dimension
        ext_mask = ext_mask.to(dtype=self.dtype)  # fp16 compatibility
        ext_mask = (1.0 - ext_mask) * -10000.0  # will set softmax to zero
        x_mask = torch.ones_like(x) * self.mask_token_id  # create an input with full mask tokens

        gen_e = self.gen_emb_layer(x)
        if self.use_scalar_mix:
            gen_h = self.gen_encoder(gen_e, ext_mask)
            gen_h = self.gen_scalar_mix(gen_h, mask)
        else:
            # selected_layers = list(map(int, self.selected_layers.split(',')))
            # gen_h = torch.stack(self.gen_encoder(gen_e, ext_mask).hidden_states)
            # gen_h = gen_h[selected_layers].mean(dim=0)
            gen_h = self.gen_encoder(gen_e, ext_mask).last_hidden_state

        z = self.explainer(gen_h, mask)
        z_mask = (z * mask.float()).unsqueeze(-1)

        if self.selection_faithfulness is True:
            pred_e = self.pred_emb_layer(x)
        else:
            pred_e = gen_h

        if self.selection_vector == 'mask':
            pred_e_mask = self.pred_emb_layer(x_mask)
        else:
            pred_e_mask = torch.zeros_like(pred_e)

        if self.selection_space == 'token':
            z_mask_bin = (z_mask > 0).float()
            pred_e = pred_e * z_mask_bin + pred_e_mask * (1 - z_mask_bin)
            # ext_mask *= (z_mask.squeeze(-1)[:, None, None, :] > 0.0).long()
        elif self.selection_space == 'embedding':
            pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)
        else:
            pred_e = pred_e * z_mask + pred_e_mask * (1 - z_mask)
            # ext_mask *= (z_mask.squeeze(-1)[:, None, None, :] > 0.0).long()

        if self.use_scalar_mix:
            pred_h = self.pred_encoder(pred_e, ext_mask).hidden_states
            pred_h = self.pred_scalar_mix(pred_h, mask)
        else:
            pred_h = self.pred_encoder(pred_e, ext_mask).last_hidden_state

        summary = masked_average(pred_h, mask)
        y_hat = self.output_layer(summary)

        return z, y_hat

    def get_loss(self, y_hat, y, prefix, mask=None):
        """
        :param y_hat: predictions from SentimentPredictor. Torch.Tensor of shape [B, C]
        :param y: tensor with gold labels. torch.BoolTensor of shape [B]
        :param mask: mask tensor for padding positions. torch.BoolTensor of shape [B, T]
        :return: tuple containing:
            `loss cost (torch.FloatTensor)`: the result of the loss function
            `loss stats (dict): dict with loss statistics
        """
        stats = {}
        loss_vec = self.criterion(y_hat, y)  # [B] or [B,C]
        # main MSE loss for p(y | x,z)
        if not self.is_multilabel:
            loss = loss_vec.mean(0)  # [B,C] -> [B]
            stats["mse"] = loss.item()
        else:
            loss = loss_vec.mean()  # [1]
            stats["criterion"] = loss.item()  # [1]

        # latent selection stats
        num_0, num_c, num_1, total = get_z_stats(self.explainer.z, mask)
        stats[prefix + "_p0"] = num_0 / float(total)
        stats[prefix + "_pc"] = num_c / float(total)
        stats[prefix + "_p1"] = num_1 / float(total)
        stats[prefix + "_ps"] = (num_c + num_1) / float(total)

        return loss, stats
