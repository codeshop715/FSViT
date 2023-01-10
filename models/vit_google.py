import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

from .utils import get_b16_config


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    #'ViT-B_32': get_b32_config(),
    #'ViT-L_16': get_l16_config(),
    #'ViT-L_32': get_l32_config(),
    #'ViT-H_14': get_h14_config(),
    #'R50-ViT-B_16': get_r50_b16_config(),
    #'testing': configs.get_testing(),
}

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        self.patch_size = patch_size
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def interpolate_pos_encoding(self, x, h, w):
        npatch = x.shape[1] - 1
        N = self.position_embeddings.shape[1] - 1
        if npatch == N and w == h:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(h0 / math.sqrt(N), w0 / math.sqrt(N)),
            mode='bicubic',
            align_corners=False,
            recompute_scale_factor=False
        )
        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        B, nc, h, w = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)

        # Linear embedding
        x = self.patch_embeddings(x)

        # add the [CLS] token to the embed patch tokens
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        embeddings = x + self.interpolate_pos_encoding(x, h, w)
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, vis=False):
        super(VisionTransformer, self).__init__()
        #self.num_classes = num_classes
        #self.classifier = config.classifier
        self.embed_dim = config.hidden_size

        # multi-scale patches
        self.AvgPool22 = nn.AvgPool2d(2, stride=1)
        self.AvgPool33 = nn.AvgPool2d(3, stride=1)
        self.AvgPool44 = nn.AvgPool2d(4, stride=1)
        self.AvgPool55 = nn.AvgPool2d(5, stride=1)
        self.AvgPool66 = nn.AvgPool2d(6, stride=1)
        self.AvgPool77 = nn.AvgPool2d(7, stride=1)
        self.AvgPool88 = nn.AvgPool2d(8, stride=1)
        self.AvgPool99 = nn.AvgPool2d(9, stride=1)
        self.AvgPool10 = nn.AvgPool2d(10, stride=1)
        self.AvgPool11 = nn.AvgPool2d(11,stride=1)
        self.AvgPool1212 = nn.AvgPool2d(12, stride=1)
        self.AvgPool1313 = nn.AvgPool2d(13, stride=1)

        # # regular patch（stride=2）
        # self.AvgPool22 = nn.AvgPool2d(2, stride=2)
        # self.AvgPool33 = nn.AvgPool2d(3, stride=2)
        # self.AvgPool44 = nn.AvgPool2d(4, stride=2)
        # self.AvgPool55 = nn.AvgPool2d(5, stride=2)
        # self.AvgPool66 = nn.AvgPool2d(6, stride=2)
        # self.AvgPool77 = nn.AvgPool2d(7, stride=2)
        # self.AvgPool88 = nn.AvgPool2d(8, stride=2)
        # self.AvgPool99 = nn.AvgPool2d(9, stride=2)
        # self.AvgPool10 = nn.AvgPool2d(10, stride=2)
        # self.AvgPool11 = nn.AvgPool2d(11,stride=2)
        # self.AvgPool1212 = nn.AvgPool2d(12, stride=2)
        # self.AvgPool1313 = nn.AvgPool2d(13, stride=2)

        # regular patch（stride=3）
        # self.AvgPool22 = nn.AvgPool2d(2, stride=3)
        # self.AvgPool33 = nn.AvgPool2d(3, stride=3)
        # self.AvgPool44 = nn.AvgPool2d(4, stride=3)
        # self.AvgPool55 = nn.AvgPool2d(5, stride=3)
        # self.AvgPool66 = nn.AvgPool2d(6, stride=3)
        # self.AvgPool77 = nn.AvgPool2d(7, stride=3)
        # self.AvgPool88 = nn.AvgPool2d(8, stride=3)
        # self.AvgPool99 = nn.AvgPool2d(9, stride=3)
        # self.AvgPool10 = nn.AvgPool2d(10, stride=3)
        # self.AvgPool11 = nn.AvgPool2d(11,stride=3)
        # self.AvgPool1212 = nn.AvgPool2d(12, stride=3)
        # self.AvgPool1313 = nn.AvgPool2d(13, stride=3)


        self.transformer = Transformer(config, img_size, vis)
        #self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x, labels=None, use_patches=True):
        x, attn_weights = self.transformer(x)
        #logits = self.head(x[:, 0])

        if use_patches:
            # return x[:, 1:]
            patch_yuan = x[:, 1:]
            batchsize = patch_yuan.size(0)
            embedding_dim = patch_yuan.size(2)
            scale = int(patch_yuan.size(1)**0.5)
            patch_sum =  patch_yuan.view(batchsize,scale,scale,-1)
            patch_sum = patch_sum.transpose(2,3)
            patch_sum = patch_sum.transpose(1,2)

            # multi-scale
            patch22 = self.AvgPool22(patch_sum)
            patch22 = patch22.view(batchsize,embedding_dim,-1)
            patch22 = patch22.transpose(1,2)

            patch33 = self.AvgPool33(patch_sum)
            patch33 = patch33.view(batchsize,embedding_dim,-1)
            patch33 = patch33.transpose(1,2)

            patch44 = self.AvgPool44(patch_sum)
            patch44 = patch44.view(batchsize,embedding_dim,-1)
            patch44 = patch44.transpose(1,2)
       
            patch55 = self.AvgPool55(patch_sum)
            patch55 = patch55.view(batchsize,embedding_dim,-1)
            patch55 = patch55.transpose(1,2)

            patch66 = self.AvgPool66(patch_sum)
            patch66 = patch66.view(batchsize,embedding_dim,-1)
            patch66 = patch66.transpose(1,2)

            patch77 = self.AvgPool77(patch_sum)
            patch77 = patch77.view(batchsize,embedding_dim,-1)
            patch77 = patch77.transpose(1,2)

            patch88 = self.AvgPool88(patch_sum)
            patch88 = patch88.view(batchsize,embedding_dim,-1)
            patch88 = patch88.transpose(1,2)

            patch99 = self.AvgPool99(patch_sum)
            patch99 = patch99.view(batchsize,embedding_dim,-1)
            patch99 = patch99.transpose(1,2)

            patch10 = self.AvgPool10(patch_sum)
            patch10 = patch10.view(batchsize,embedding_dim,-1)
            patch10 = patch10.transpose(1,2)

            patch11 = self.AvgPool11(patch_sum)
            patch11 = patch11.view(batchsize,embedding_dim,-1)
            patch11 = patch11.transpose(1,2)

            patch1212 = self.AvgPool1212(patch_sum)
            patch1212 = patch1212.view(batchsize,embedding_dim,-1)
            patch1212 = patch1212.transpose(1,2)

            patch1313 = self.AvgPool1313(patch_sum)
            patch1313 = patch1313.view(batchsize,embedding_dim,-1)
            patch1313 = patch1313.transpose(1,2)

            patch14 = patch_yuan.mean(dim=1,keepdim = True)

            # multi-scale patches
            patch_merging = torch.cat([patch22,patch33,patch44,patch55,patch66,patch77,patch88,patch99,
            patch10,patch11,patch1212,patch1313,patch_yuan,patch14],dim=1)
            
            return patch_merging
        else:
            return x[:, 0]

    def load_from(self, weights):
        with torch.no_grad():

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                print("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
