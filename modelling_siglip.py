from typing import Optional,Tuple
import torch
import torch.nn as nn

class SigLIPVisionConfig:
    def __init__(
            self,
            hidden_size=768, #size of the embedding vector.
            intermediate_size=3072,#size of linear layer used in FFN
            num_hidden_layers=12,#number of layers in vision transformer
            num_attention_heads=12,#number of heads in MHA
            num_channels=3,#how many channels each image have (RGB)
            image_size=224, # the provided image will be resized in 224by224
            patch_size=16, # Imag will be divided into patches and each patch will be 16 by 16
            layer_norm_eps=1e-6, 
            attention_dropout=0.0,
            num_image_tokens:int=None,
            **kwargs
        ):
           super().__init__()
           self.hidden_size=hidden_size
           self.intermediate_size=intermediate_size
           self.num_hidden_layers=num_hidden_layers
           self.num_attention_layers=num_attention_heads
           self.num_channels= num_channels
           self.patch_size=patch_size
           self.image_size=image_size
           self.attention_dropout=attention_dropout
           self.layer_norm_eps = layer_norm_eps
           self.num_image_tokens=num_image_tokens
           

class SigLIPVisionTransformer(nn.Module):
      def __init__(self,config:SigLIPVisionConfig):
            super().__init__()
            self.config=config
            embed_dim=config.hidden_size
            self.embeddings=SigLIPVisionEmbedding(config)
            self.encoder=SigLIPEncoder(config)
            self.post_layernorm=nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
            
class SigLIPVisionModel(nn.Module):
      def  __init__(self,config:SigLIPVisionConfig):
            super().__init__()
            self.config= config
            self.vision_model=SigLIPVisionTransformer(config)
      def forward(self,pixel_values)->Tuple:
            #[Batch_size,Channels,Height,Width] -> [Batch_size,num_Patches,Embed_dim]
            return self.vision_model(pixel_values=pixel_values)
            



            
