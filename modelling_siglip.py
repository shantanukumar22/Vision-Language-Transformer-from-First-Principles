#pre explanation
# this file will explain the vision encoder of the SigLIP
#image->tokens->transformer->contextual image embedding
#this file will build a Vision Transformer used in SigLIP

# from typing import Optional,Tuple
# import torch
# import torch.nn as nn
# class SigLIPVisionConfig:
#     def __init__(
#             self,
#             hidden_size=768, #size of the embedding vector.each image will become 768 dimensional token
#             intermediate_size=3072,#size of linear layer used in FFN
#             num_hidden_layers=12,#number of layers in vision transformer. (no of transformer block)
#             num_attention_heads=12,#number of heads in MHA
#             num_channels=3,#how many channels each image have (RGB)
#             image_size=224, # the provided image will be resized in 224X224
#             patch_size=16, # Imag will be divided into patches and each patch will be 16 by 16. 224/16= 14. 14 X 14 = 196 patches
#             layer_norm_eps=1e-6, 
#             attention_dropout=0.0,
#             num_image_tokens:int=None,
#             **kwargs
#         ):
#            super().__init__()
#            self.hidden_size=hidden_size
#            self.intermediate_size=intermediate_size
#            self.num_hidden_layers=num_hidden_layers
#            self.num_attention_heads=num_attention_heads
#            self.num_channels= num_channels
#            self.patch_size=patch_size
#            self.image_size=image_size
#            self.attention_dropout=attention_dropout
#            self.layer_norm_eps = layer_norm_eps
#            self.num_image_tokens=num_image_tokens
# #!Image to Token
# class SigLIPVisionEmbedding(nn.Module):
#       def __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.config=config
#             self.embed_dim=config.hidden_size
#             self.image_size=config.image_size
#             self.patch_size=config.patch_size
#             #patch embedding is extracting information from our image patch by patch where there is no overlapping
#             #each convolution output = one patch embedding
#             self.patch_embedding=nn.Conv2d(
#                   in_channels=config.num_channels,
#                   out_channels=config.hidden_size,
#                   kernel_size=self.patch_size,
#                   stride=self.patch_size,  # we are doing no overlap
#                   padding="valid", #this indicates no padding is added  
#                         )
#             #[Batch, 3, 224, 224] →[Batch, 768, 14, 14]
#             self.num_patches=(self.image_size // self.patch_size) ** 2
#             self.num_positions=self.num_patches # positional_encoding is equal to the number of the patches so that we can know where each patch came from
#             self.position_embedding=nn.Embedding(self.num_positions,self.embed_dim) # we have the position encoding of number num_position with each of size as the embed_dimension.These positonal encoding are learnt and not calculated like vanila transformer
#             self.register_buffer(
#                   "position_ids",
#                   torch.arange(self.num_positions).expand((1,-1)),
#                   persistent=False,
#             )

#       def forward(self,pixel_values:torch.FloatTensor)->torch.Tensor:
#             #we load the image(pixel_values) which is a batch of images with channels, height and width
#             _,_,height,width=pixel_values.shape #[Batch_size,Channels,Height,Width]
#             patch_embed=self.patch_embedding(pixel_values)
#             embeddings=patch_embed.flatten(2)
#             embeddings=embeddings.transpose(1,2) #no.of patches before embedding dimenti on
#             #[Batch, 768, 14, 14]→[Batch, 768, 196]→[Batch, 196, 768] #image is square so width and height will be resized into image_size
#             embeddings=embeddings+self.position_embedding(self.position_ids)
#             #[Batch, Num_Patches, Embed_Dim] → [B, 196, 768]
#             return embeddings

# class SigLIPEncoder(nn.Module):
#       def __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.config=config
#             self.layers=nn.ModuleList([SigLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
#       def forward(self,input_embeds:torch.Tensor)-> torch.Tensor:
#             #input_embeds :[Batch_size,Num_patches,Embed_dim]
#             hidden_states=input_embeds
#             for encoder_later in self.layers:
#                   #[Batch_size,Num_patches,Embed_dim]
#                   hidden_states=encoder_later(hidden_states)
#             return hidden_states
# class SigLIPEncoderLayer(nn.Module):
#       def __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.embed_dim=config.hidden_size
#             self.self_attn=SigLIPAttention(config)
#             self.layer_norm1=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
#             self.mlp=SigLIPMLP(config)
#             self.layer_norm2=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)
#       def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
#             #residual -> (Batch_size,Num_patches,Embed_dimension)
#             residual = hidden_states
#             hidden_states=self.layer_norm1(hidden_states) #does not change shape of input
#             hidden_states,_=self.self_attn(hidden_states=hidden_states) #does not change the shape of input but it takes an embedding and return a contextualized embedding
#             # (Batch_size,Num_patches,Embed_dimension)
#             hidden_states=hidden_states+residual
#             residual=hidden_states
#             hidden_states=self.layer_norm2(hidden_states)
#             #while in self attention there is mixing of tokens and patches for contextualizing but in the 
#             #MLP each of them in transform independetly it increases parameters so the model have more DOF to learn
#             #second would be it allows the sequence of patches for the next layer
#             #also introduces non-linearity which let model learns on complex transformations
#             hidden_states =  self.mlp(hidden_states)
#             # (Batch_size,Num_patches,Embed_dimension)
#             hidden_states=residual+hidden_states
#             # (Batch_size,Num_patches,Embed_dimension)

#             return hidden_states
      
# class SigLIPVisionTransformer(nn.Module):
#       def __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.config=config
#             embed_dim=config.hidden_size
# #! takes the patch of the image using conv then convert it into the embedding which will be added to another vector called positonal encoding
#             self.embeddings=SigLIPVisionEmbedding(config)
# #!We will feed the embedding to the encoder
#             self.encoder=SigLIPEncoder(config)
#             self.post_layernorm=nn.LayerNorm(embed_dim,eps=config.layer_norm_eps)
#             #pixel values  is the images which is the batch of images 
#             #unlike the vanilla transformer in vision transformer the normalization comes before the MHA or FFN
#       def forward(self,pixel_values:torch.Tensor) -> torch.Tensor:
#             #pixel values:[batch_size,Channels,Height,Width] -> [Batch_size,Num_Patches,Embed_dim ]
#             hidden_state=self.embeddings(pixel_values)
#             last_hidden_state = self.encoder(hidden_state)
#             last_hidden_state=self.post_layernorm(last_hidden_state)

#             return last_hidden_state
            
# class SigLIPMLP(nn.Module):
#       def __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.config=config
#             #patches are expanded into intermediate_size from hidden_size adds non-linearity then again compressed back to  the hidden_size size
#             self.fc1=nn.Linear(config.hidden_size,config.intermediate_size)
#             self.fc2=nn.Linear(config.intermediate_size,config.hidden_size)
#       def forward(self, hidden_states):
#             #[Batch_size,Num_Patches,Embed_dim]->[Batch_size,Num_patches,intermediate_size]
#             hidden_states=self.fc1(hidden_states)
#             hidden_states=nn.functional.gelu(hidden_states,approximate='tanh')
#             #[Batch_size,Num_patches,intermediate_state]->[Batch_size,Num_patches,embed_dim]
#             hidden_states=self.fc2(hidden_states)
#             return hidden_states
# ##!Attention mechanism for Vision Transformer
# # we don't have any causal mask here

# class SigLIPVisionModel(nn.Module):
#       def  __init__(self,config:SigLIPVisionConfig):
#             super().__init__()
#             self.config= config
#             self.vision_model=SigLIPVisionTransformer(config)
#       def forward(self,pixel_values)->Tuple:
#             #[Batch_size,Channels,Height,Width] -> [Batch_size,num_Patches,Embed_dim]
#             return self.vision_model(pixel_values=pixel_values)


# class SigLIPAttention(nn.Module):
#       """Multi-head attention from paper"""
#       def __init__(self,config):
#             super().__init__()
#             self.config=config
#             self.embed_dim=config.hidden_size
#             self.num_head=config.num_attention_heads 
#             self.head_dim=self.embed_dim // self.num_head
#             self.scale=self.head_dim ** -0.5 #equivalent to 1/sqrt(self.head_dim)
#             self.dropout=config.attention_dropout
#             self.k_layer = nn.Linear(self.embed_dim, self.embed_dim)
#             self.v_layer = nn.Linear(self.embed_dim, self.embed_dim)
#             self.q_layer = nn.Linear(self.embed_dim, self.embed_dim)
#             self.out_proj=nn.Linear(self.embed_dim,self.embed_dim)
#       def forward(self,
#                   hidden_states:torch.Tensor,
#                   )->Tuple[torch.Tensor,Optional[torch.Tensor]]:
#             #the output of the layer norm will work as the input for the Attention
#             #hidden statses:=[Batch_size,num_patches,embed_dim]
#             batch_size,seq_len,_=hidden_states.size() 
#             query_value=self.q_layer(hidden_states)
#             key_value=self.k_layer(hidden_states)
#             value_value=self.v_layer(hidden_states)
#             #.view splits the last dimension into the smaller part embed dim into numhead and headdim
#             #query_value:[Batch_size,num_patches,num_head,head_dim].transpose(1,2)->[Batch_size,num_head,num_patches,head_dim]
#             query_value=query_value.view(batch_size,seq_len,self.num_head,self.head_dim).transpose(1,2)
#             key_value=key_value.view(batch_size,seq_len,self.num_head,self.head_dim).transpose(1,2)
#             value_value=value_value.view(batch_size,seq_len,self.num_head,self.head_dim).transpose(1,2)
#             #calculate attention
#             #atn weights: [Batch_size,Num_heads,Num_patches,Num_patches]
#             attn_weights=(torch.matmul(query_value,key_value.transpose(2,3))* self.scale)
#             #verify the dimension of the matrix
#             if attn_weights.size() != (batch_size,self.num_head,seq_len,seq_len):
#                   raise ValueError(
#                         f"attention weight should be of size {(batch_size,self.num_head,seq_len,seq_len)}, but is {attn_weights.size()}"
#                   )
#             #then we apply the softmax to convert the attn_score into the number between zero and one such that they sum upto 1
#             attn_weights=nn.functional.softmax(attn_weights,dim=-1,dtype=torch.float32).to(query_value.dtype)
#             #multiply the attention with the value sequence
#             weighted_attn=torch.matmul(attn_weights,value_value)
#             #checking for the size
#             if weighted_attn.size() !=(batch_size,self.num_head,seq_len,self.head_dim):
#                   raise ValueError(
#                         f"attn_output  should be of size {(batch_size,self.num_head,seq_len,self.head_dim)} but it is {weighted_attn.size()}"
   
#                   )
#             #[batch_size,Num_heads,Num_patches,Head_dim]->["Batch_size","num_patches","Num_heads",Head_dim]

#             weighted_attn=weighted_attn.transpose(1,2).contiguous()
#             weighted_attn=weighted_attn.reshape(batch_size,seq_len,self.embed_dim)
#             #[Batch_size,num_patches,embed_dim]
#             weighted_attn=self.out_proj(weighted_attn)
#             return weighted_attn, attn_weights

            
              




       





            





from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid", # This indicates no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2)
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings


class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = self.fc1(hidden_states)
        # hidden_states: [Batch_Size, Num_Patches, Intermediate_Size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [Batch_Size, Num_Patches, Intermediate_Size] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # residual: [Batch_Size, Num_Patches, Embed_Dim]
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        # residual: [Batch_Size, Num_Patches, Embed_Dim] 
        residual = hidden_states
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = residual + hidden_states
        
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, pixel_values) -> Tuple:
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        return self.vision_model(pixel_values=pixel_values) 