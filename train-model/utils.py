import csv
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import CLIPTextModelWithProjection,CLIPTokenizer
from transformers.models.clip.modeling_clip import (
    CLIPTextTransformer, 
    _create_4d_causal_attention_mask, 
    _prepare_4d_attention_mask
)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def display_loss(loss, save_path=None):
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def save_toFile(path, file_name, data_saved, rows=0):
    f = open(path + file_name, 'w')
    writer = csv.writer(f)
    if rows == 0:
        writer.writerow(data_saved)
    if rows == 1:
        writer.writerows(data_saved)
    f.close()


def display_losses(loss1, loss2=None, save_path=None):
    plt.plot(loss1, label='Training Loss - Agent A')

    if loss2 is not None:
        plt.plot(loss2, label='Training Loss - Agent B')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def gumbel_softmax(logits, temperature=0.5):
    g = torch.distributions.gumbel.Gumbel(torch.zeros_like(logits), torch.ones_like(logits))
    G = g.sample()
    return F.softmax((logits + G) / temperature, dim=-1)


def straight_through_discretize(z_sampled_soft):
    z_argmax = torch.argmax(z_sampled_soft, dim=-1, keepdim=True)
    z_argmax_one_hot = torch.zeros_like(z_sampled_soft).scatter_(-1, z_argmax, 1).float()
    z_sampled_onehot_with_grad = z_sampled_soft + (z_argmax_one_hot - z_sampled_soft).detach()
    return z_sampled_onehot_with_grad, z_argmax

class MyClipTextTransformer(CLIPTextTransformer):
  def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_shape = (input_ids.shape[0],input_ids.shape[1])
        hidden_states = self.embeddings(inputs_embeds=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[:, -1, :]  
        
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

def MyCLIPTextModel(url):
    text_encoder = CLIPTextModelWithProjection.from_pretrained(url)
    config = text_encoder.text_model.config
    my_text_transformer = MyClipTextTransformer(config)
    # load_state_dict で model.text_model (CLIPTextTransformer) の重みをコピー
    my_text_transformer.load_state_dict(text_encoder.text_model.state_dict())
    # 4) model.text_model を置き換え
    text_encoder.text_model = my_text_transformer
    return text_encoder

def embed_special_token():
    text_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-large-patch14")
    tokenizer =  CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    with torch.no_grad():
        bos_embedding =  text_model(torch.tensor([bos_token_id])).last_hidden_state
        eos_embedding =  text_model(torch.tensor([eos_token_id])).last_hidden_state
    return bos_embedding,eos_embedding

class TransformerModel(nn.Module):
    def __init__(self, d_model=768, num_heads=2, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Transformer Encoder Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, 
            num_layers=num_layers
        )
        
        # Linear layer to map output back to desired shape
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # Apply Transformer
        x = self.transformer_encoder(x)
        # Output through a linear layer to map to the same shape
        x = self.fc(x)
        
        return x
    
class ClipVisionAdapter(nn.Module):
    """
    論文「CLIP-Adapter」で提案されているような
    LayerNorm + MLP + Residual のシンプルな例。
    """
    def __init__(self, embed_dim: int =768, adapter_dim: int = 512,ratio=0.1):
        super().__init__()
        self.ratio = ratio
        self.ln = nn.LayerNorm(embed_dim)  # 入力特徴の正規化
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, adapter_dim),
            nn.ReLU(),
            nn.Linear(adapter_dim, embed_dim),
        )

    def forward(self, x):
        # x: [batch_size, embed_dim]
        h = self.ln(x)
        h = self.mlp(h)
        # Residual connection (x + h)
        return (1-self.ratio)*x + self.ratio*h

