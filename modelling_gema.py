import torch
from torch import nn
from typing import Tuple,Optional,List
from torch.nn import CrossEntropyLoss
from modelling_siglip import SigLIPVisionConfig,SigLIPVisionModel
#wrtting from the bottom up approach (create architecture then we will create the 
# each part of the model)



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