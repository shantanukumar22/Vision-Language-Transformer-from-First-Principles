from typing import Dict,List,Optional,Union,Tuple,Iterable
import numpy as np
from PIL import Image
import torch
#imagetokens=imageemebeddings
#load the image as the tensor for the input in the vision model
#used to normalize the images,convert pixel values into a range suitable for neural networks usually [-1to1]
IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD=[0.5,0.5,0.5]
class PaligemmaProcessor:
    #(num_image_tokens= how many tokens represent the image)
    #image_size=image resolution
    def __init__(self,tokenizer,num_image__tokens:int,image_size:int):
        super().__init__()
        self.image_seq_len=num_image__tokens #how long the image  embeddding sequence is
        self.image_size=image_size #what size image should be resized to 
        self.IMAGE_TOKEN="<image>"
        tokens_to_add={"additional special tokens: " [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)
        #Extra tokens for the object segmentation, these tokens are used for bounding boxed segmentation and spatial grounding
        EXTRA_TOKENS=[
            f"<loc{i:04d}>" for i in range(1024)
        ] #These  tokens are used in object segmentation
        tokenizer.add_tokens(EXTRA_TOKENS)
        # The object is at <loc0123> <loc0456> so the models can reger to image regions using the text
        #get image token id this convers "<image>" -> integer ID (the model only works with ids not strings)
        self.image_token_id=tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        #we will add the BOS and EOS token ourselves
        tokenizer.add_bos_token=False
        tokenizer.add_eos_token=False

        #saving the tokenizer so it can be used later 
        self.tokenizer=tokenizer
    #!this makes the class callbale like they are objects    
    def __call__(self,text:List[str],images:List[Image.Image],
                 padding:str="longest",
                 truncation:bool=True,
                 )->dict:   
                    assert len(images)==1 and len(text) ==1,f"Recieved {len(images)} images for {len(text)} prompts"

#processor sits before the model and prepares text and images into
#tensors that model can understand
