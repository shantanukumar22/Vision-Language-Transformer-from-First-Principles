# ! this file is the fusion logic between vision and language
#Image → Vision Encoder → Image Embeddings
# Text  → Tokenizer      → Text Embeddings

# Image Embeddings + Text Embeddings
#                 ↓
#         Merge (Fusion Layer)
#                 ↓
#         Transformer (Gemma)
#                 ↓
#          Next Token Prediction 
#layer idx: positon of the layer in transformer layer index gemmma is decoder only withy many layer and each of the layer have its own kvcache to know which kvcache to use we pass it

#Image-> SigLIP Vision Encoder ->Image Patch Embeddings -> projection Layers ( resize to LM dimension )
# -> Merge with text embeddings  -> Gemma Language model -> generated text
#!causality is a choice. which is made during the arhcitecture of LLM. authors of paligemma didn't put the
#causality in the prompt for the image.but it is in the generation. however mostly LLM are build in case with masking the next tokens even the prompts cause it is considered as the generation itself
import torch
from torch import nn
from typing import Tuple,Optional,List
from torch.nn import CrossEntropyLoss
from modelling_siglip import SigLIPVisionConfig,SigLIPVisionModel
#wrtting from the bottom up approach (create architecture then we will create the 
# each part of the model) vocab_size defines how many different symbols the model can embed and predict.

class KVCache():

    def __init__(self) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # If we never added anything to the KV-Cache of this layer, let's create it.
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # ... otherwise we concatenate the new keys with the existing ones.
            # each tensor has shape: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # ... and then we return all the existing keys + the new ones.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class GemmaConfig():

    def __init__(
        self,
        vocab_size, # number of token language model understand vocab_size defines how many different symbols the model can embed and predict. example 32000 words/tokens in vocabulary
        hidden_size,# embedding dimension of each token (cat -> vector size of 2048)
        intermediate_size,#intermediate size of the FeedForward layer
        num_hidden_layers, #how many layers our transformer have in our Gemma model
        num_attention_heads, #how many heads in the self-attention
        num_key_value_heads, #Used for group Query Attention(GQA) -- important Gemma optimization (many query heads/ fewwer key/value heads reduces memory cost)
        head_dim=256, #how many dimension each head will work with. dimension per head
        max_position_embeddings=8192,  #Maximum Sequence length our model supports  .maximum number of position our model has been trained upon necessary for Rotary Positional Encoding
        rms_norm_eps=1e-6, 
        rope_theta=10000.0, # scaling constant used in RoPE frequency computation
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
#this config merges vision encoder config, text model config and multimodal projector config
#global configuration for the entier multimodal  projector configuration
    def __init__(
        self,
        vision_config=None, #configuration of vision encoder
        text_config=None, #configuration text decoder
        ignore_index=-100, # is it used in training but we are only doing inference
        image_token_index=256000, #token corresponding to placeholder <image>. special token representing <image> placeholder  inside text prompt. this model replaces <image> token with actual  image embeddings
        vocab_size=257152,
        projection_dim=2048,#final dimension image feature should be resized to before feeding to the language model
        hidden_size=2048, #embedding size of the language model
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        self.vision_config.projection_dim = projection_dim


class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    def __init__(self,config:GemmaConfig,layer_idx: Optional[int]=None):
        super().__init__()
        self.config=config
        self.layer_idx=layer_idx

        self.attention_dropout=config.attention_dropout
        self.hidden_size=config.hidden_size
        self.num_heads=config.num_attention_heads
        self.head_dim=config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True


        assert self.hidden_size % self.num_heads == 0
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size() # [Batch_Size, Seq_Len, Hidden_Size]
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(hidden_states)
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(hidden_states)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # [Batch_Size, Seq_Len, Head_Dim], [Batch_Size, Seq_Len, Head_Dim]
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim], [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        # Perform the calculation as usual, Q * K^T / sqrt(head_dim). Shape: [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # Apply the softmax
        # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        # Multiply by the values. [Batch_Size, Num_Heads_Q, Seq_Len_Q, Seq_Len_KV] x [Batch_Size, Num_Heads_KV, Seq_Len_KV, Head_Dim] -> [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Seq_Len_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Concatenate all the heads together. [Batch_Size, Seq_Len_Q, Num_Heads_Q, Head_Dim] -> [Batch_Size, Seq_Len_Q, Num_Heads_Q * Head_Dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights




class GemmaDecoderLayer(nn.Module):

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)

        self.mlp = GemmaMLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.input_layernorm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        # [Batch_Size, Seq_Len, Hidden_Size]
        residual = hidden_states
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.post_attention_layernorm(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.mlp(hidden_states)
        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = residual + hidden_states

        return hidden_states

class GemmaModel(nn.Module):
    def __init__(self,config:GemmaConfig):
        super().__init__()
        self.config=config
        self.padding_idx=config.pad_token_id
        self.vocab_size=config.vocab_size
        self.embed_tokens=nn.Embedding(config.vocab_size,config.hidden_size,self.padding_idx)
        self.layers=nn.ModuleList(
            [GemmaDecoderLayer(config,layer_idx) for layer_idx in range(config.num_hidden_layers)]

        )
        self.norm=GemmaRMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    def get_input_embeddintgs(self):
        return self.embed_tokens
    
    def forward(
            self,
            attention_mask:Optional[torch.Tensor] = None,
            positions_ids: Optional[torch.LongTensor] = None,
            inputs_embed: Optional[torch.FloatTensor] = None,
            kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        #[Batch_size,Seq_len,Hidden_size]
        hidden_states=inputs_embed
        #[Batch_size,seq_len,Hidden)size]
        normalizer=torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states=hidden_states*normalizer

        #built from bunch of layers so output of one layer becomes the input of another layer
        for decoder_layer in self.layers:
            #[Batch_size,Seq_len,hidden_size]
            hidden_states=decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                positions_ids=positions_ids,
                kv_cache=kv_cache,
            )
        hidden_states=self.norm(hidden_states)

        return hidden_states

class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        #eps added to avoid division by zero
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

# in hugging face whenever we see something something for causalLM it's just transformer model+language modelling head
class GemmaForCausalLM(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.model=GemmaModel(config)
        self.vocab_size=config.vocab_size
        self.lm_head=nn.Linear(config.hidden_size,config.vocab_size,bias=False)
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    def tie_weights(self):
        self.lm_head.weight(self.model.embed_tokens.weight)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:

        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
 
#linear layer that converts the size of the image feature extracted from the vision encoder in the size of embedding size used by the language model
#basically resizing so that they can be concatenated with the text embedding 
class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.linear=nn.linear(config.vision_config.hidden_size,config.vision_config.projection_dim,bias=True)
        def forward(self,image_features):
            hidden_states=self.linear(image_features)
            return hidden_states

class PaliGemmaForConditionalGeneration(nn.Module):
    #actual multimodal model.
    def __init__(self,config:PaliGemmaConfig):
        super().__init__()
        self.config=config
        #instance of the vision model(encoder of the image)
        self.vision_tower=SigLIPVisionModel(config.vision_config) # image encoder input [batch,3,224,224] -> [batch,196,embed_dim]
        #linear layer or linear projection as  mentioned in the paper
        self.multi_modal_projector=PaliGemmaMultiModalProjector(config) #project vision embeddings to the language model dimension ([batch,196,768] → [batch,196,2048])
        self.vocab_size=config.vocab_size
        language_model=GemmaForCausalLM(config.text_config) # text decoder, for the generation of the token autoregressively 
        self.language_model=language_model
        self.pad_token_id=self.config.pad_token_id if self.config.pad_token_id is not None else -1 # stores padding token


#weight tying is the technique of reusing the parameters of one layer into the other
    def tie_weights(self):
        return self.language_model.tie_weigths()
    #fusion
    #image_features -> embeddings from vision encoder,
    #input_embeds-> embeddings of text tokens 
    #input_ids-> token ids
    #attention mask
    #goal -> Replace <image> token position with actual image embedding
    #final sequence [image_tokens,text_tokens]
    def merge_input_ids_with_image_features(self,image_features:torch.Tensor,input_embeds:torch.Tensor,input_ids:torch.Tensor,
                                            attention_mask:torch.Tensor,kv_cache:Optional[KVCache]=None):
        _,_, embed_dim=image_features.shape
        batch_size,sequence_length=input_ids.shape
        dtype,device=input_embeds.dtype,input_embeds.device
        #shape:[Batch_size,Seq_len,Hidden_size]
        scaled_image_features=image_features/(self.config.hidden_size**0.5)
        #combine the embeddings of the image tokens, the text tokens and the mask out all the padding tokens
        final_embedding= torch.zeros(batch_size,sequence_length,embed_dim, dtype=input_embeds.dtype,device=input_embeds)
        #shape: [Batch_size, sequence_length] True for text tokens
        text_mask=(input_ids != self.config.image_token_index) & (input_ids !=self.pad_token_id)
        #shape:[Batch_size,seq len] true for image tokens
        image_mask= input_ids== self.config.image_token_index
        # shape: [Batch_size,seq_len]. True for padding tokens
        pad_mask=input_ids == self.pad_token_id
        # these mask are useful for identifying where to put the image token padding token or text token

        # we need to expand the masks to the embeddding dimension otherwise we cant use them in torch.where
        text_mask_expanded=text_mask.unsqueeze(-1).expand(-1,-1, embed_dim) #expands the 0 and 1 to embed_dimension
        pad_mask_expanded=pad_mask.unsqueeze(-1).expand(-1,-1,embed_dim)
        image_mask_expanded=image_mask.unsqueeze(-1).expand(-1,-1,embed_dim)

        #add the text embedding
        #whenever text mask is 1 we copy the embeddding from the input embeds otherwise final_embedding 
        final_embeddings=torch.where(text_mask_expanded,input_embeds,final_embedding)
        #insert image embedding, we cant use torch.where because the  sequence length of the scaled_image_feature is not equal  to the sequence length of the final emebeddding 
        #copy from the scaled image  where image_mask_expanded is true.
        final_embedding=final_embedding.masked_scatter(image_mask_expanded,scaled_image_features)

        #zero out padding tokens
        #wherever the pad_mask_expanded is true keep the zeroes in final embedding othwerwise keep the final embedding
        final_embedding=torch.where(pad_mask_expanded,torch.zeros_like(final_embedding),final_embedding)
#since the embedding and the linear in the transformer works in opposite of easch other what we do is to convert is use the parameter by using the weight tying same for teh both cases 
#also a technique to reduce the number of parameters by sharing it
        ##!# create the attention mask 
        dtype,device=input_embeds.dtype,input_embeds.device
        min_dtype=torch.finfo(dtype).min
        q_len=input_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # do not mask cause we are in the prefill phase i.e doing it for the first time 
            # this only works when we have no padding 
            
            causal_mask= torch.full(
                (batch_size,q_len,q_len), fill_value=0, dtype=dtype,device=device
            )
        else:
            ##since we are generating the token the query must be one single token
# we are not masking anything bcz we are working with kv_cache and working with kv_cache is generating only the last row which have access to all the previous tokens so we
# don't need to mask out anything however duiring training when we train a model on sometihng we need to mask out cause model will generate all the contextualized embedding in 
# parallel and we want it to have the access to only the previous tokens 
#  during inference we don't have  any causal mask but during training we have causal mask for generation   
# when we are working with the models like llamma we mask out even the pre-filling part but for the paligemma we do not mask out anytthing 

            assert q_len==1
            kv_len= kv_cache.num_items() + q_len
            # also in this case we dont need to mask anything since each query should be able to attend all previous tokens
            # this only works when we have no padding 
            causal_mask=torch.full((batch_size,q_len,kv_len),fill_value=0,dtype=type,device=device)
            
            ## add the head dimension
            
        # we have one attention computation for the each head, there will be one attention matrix for each head
            #[Batch_size,q_len,kv_len] -> [Batch_size,Num_heads_q,Q_len,KV_len]
        causal_mask=causal_mask.unsqueeze(1)
        if kv_cache is not None and kv_cache.num_items() >0:
            #prefill phase where we have image+text tokens and we will generating rope
            position_ids=attention_mask.cumsum(-1)[:,-1]
            if position_ids.dim()==1:
                position_ids=position_ids.unsqueeze(0)
        else:
            #this is the generation part so just 1 query where we will be needing the positional query
            #create the position ids  based on the size of the attention mask 
            #for masked tokens,use the number 1 as position
            position_ids= (attention_mask.cumsum(-1).masked_fill_(attention_mask==0),1).to(device)
        return final_embedding,causal_mask,position_ids
    def forward(
        self,
        input_ids:torch.LongTensor=None, # image_token contains image_seq_len bos_token  prefix_prompt \n 
#! the goal here for conditional generation will be to take pixel values and feed it to image encoder to  extract the image tokens t
        pixel_values:torch.FloatTensor=None, #image extracted from the paligemma processor which is resized rescaled and normalized
#  attention mask is provided directly by the tokenizer so whenever you tokenize the text 
#using the tokenizer    it give you two output input ids and attention mask 
        attention_mask:Optional(torch.Tensor)=None,
        kv_cache:Optional[KVCache]=None,)->Tuple:
        #we have not implemented the padding logic
        assert torch.all(attention_mask==1),"the input can not be padded"
        #Extract teh input Embeddings
        #shape:(batch_size,Seq_len,Hidden_size)
        #  convert tokens into vectors
        #image to patch  embedddings 
        input_embeds=self.language_model.get_input_embeddings()(input_ids)
         #Merge Text and Images:
         #[Batch_size,Channels,Height,Width] -> [Batch_size,num_patches,embed_dim]
        selected_image_feature=self.vision_tower(pixel_values.to(input_embeds.dtype))
        #[Batch_size,num_Patches,Embed_dim]=> [Batch_size,num_patches,hidden_size]
        #project image to the Lm dimension
        image_features=self.multi_modal_projector(selected_image_feature)
        #merge embeddings of the text token and the image tokens 
        #creates the multimodal seqeunce
        input_embeds,attention_mask,position_ids=self.merge_input_ids_with_image_features(image_features,input_embeds,input_ids,attention_mask)
        #pass into language model
        outputs= self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
         )
        return outputs
    

 #gpu  is fast in computation but very slow in copying  the real bottleneck is not the computations
 # but its actually the time it takes to copy from HBM (High bandwidth memory) to local memory(gpu)
 # so in MHA the bottleneck is not exatly the dot product computation but rather it is the transfer of the data and flash attention solve and work on this issue exactly
 # other way to fix this issues is by using the less heads leading to the less data transfer inside the GPU
 # use 1/less head for the keys while many head for the query
 #let's say we got 8 query heads and 4 keys head so two query heads will actually just attend the one key head
 # this reduces the quality of the model but it's at level we can afford to lose this much
 # read about GQA and MQA(multi Query attention)
 # MQA reduces the accurary by a lot and MHA requires a lot of computation
 # GQA settles in between it perfectly 
 # compresses this tokens also helps in handling the KV-cache. the bottleneck for high parameter is also the token storage of the kv-cacche 
 