import torch
from torch import nn
from typing import Tuple,Optional,List
from torch.nn import CrossEntropyLoss
from modelling_siglip import SigLIPVisionConfig,SigLIPVisionModel
#wrtting from the bottom up approach (create architecture then we will create the 
# each part of the model) vocab_size defines how many different symbols the model can embed and predict.
class GemmaConfig():

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,#intermediate size of the FeedForward layer
        num_hidden_layers, #how many layers our transformer have in our Gemma model
        num_attention_heads, #number of heads for the query
        num_key_value_heads, #number of heads for the key and value 
        head_dim=256, #how many dimension each head will work with
        max_position_embeddings=8192, #maximum number of position our model has been trained upon necessary for Rotary Positional Encoding
        rms_norm_eps=1e-6, 
        rope_theta=10000.0,
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

    def __init__(
        self,
        vision_config=None, #configuration of vision encoder
        text_config=None, #configuration text decoder
        ignore_index=-100, # is it used in training but we are only doing inference
        image_token_index=256000, #token corresponding to placeholder <image>
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



class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self,config:PaliGemmaCofig):
        super().__init__()
        self.config=config
        #instance of the vision model(encoder of the image)
        self.vision_tower=SigLIPVisionModel(config.vision_config)
        #linear layer or linear projection as  mentioned in the paper
        self.multi_modal_projector=PaliGemmaMultiModalProjector(config)
        self.vocab_size=config.vocab_size
        language_model=GemmaForCausalLM(config.text_config)
        self.language_model=language_model
        self.pad_token_id=self.config.pad_token_id if self.config.pad_token_id is not None else -1


#weight tying is the technique of reusing the parameters of one layer into the other
    def tie_weights(self):
        return self.language_model.tie_weigths()
    
    def merge_input_ids_with_image_features(self,image_features:torch.Tensor,input_embeds:torch.Tensor,input_ids:torch.Tensor,
                                            attention_mask:torch.Tensor,kv_cache:Optional[KVCache]=None):
        _,_, embed_dim=image_features.shape
        batch_size,sequence_length=input_ids.shape
        dtype,device=input_embeds.dtype,input_embeds.device
        

#since the embedding and the linear in the transformer works in opposite of easch other what we do is to convert is use the parameter by using the weight tying same for teh both cases 
#also a technique to reduce the number of parameters by sharing it
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
        input_embeds=self.language_model.get_input_embeddings()(input_ids)
         #Merge Text and Images:
         #[Batch_size,Channels,Height,Width] -> [Batch_size,num_patches,embed_dim]
        selected_image_feature=self.vision_tower(pixel_values.to(input_embeds.dtype))
        #[Batch_size,num_Patches,Embed_dim]=> [Batch_size,num_patches,hidden_size]
        image_features=self.multi_modal_projector(selected_image_feature)
        #merge embeddings of the text token and the image tokens 
        input_embeds,attention_mask,position_ids=self.merge_input_ids_with_image_features(image_features,input_embeds,input_ids,attention_mask)
        outputs= self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            input_embeds=input_embeds,
            kv_cache=kv_cache,
         )
        return outputs