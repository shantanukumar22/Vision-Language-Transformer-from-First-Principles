from typing import Dict,List,Optional,Union,Tuple,Iterable
import numpy as np
from PIL import Image
import torch
#imagetokens=imageemebeddings
#load the image as the tensor for the input in the vision model
#used to normalize the images,convert pixel values into a range suitable for neural networks usually [-1to1]
IMAGENET_STANDARD_MEAN = [0.5,0.5,0.5]
IMAGENET_STANDARD_STD=[0.5,0.5,0.5]

def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally.
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
              images:List[Image.Image],
              size:Dict[str,int],
              resample:Image.Resampling=None,
              rescale_factor:float=None,
              image_mean:Optional[Union[float,List[float]]] =None,
              image_std:Optional[Union[float,List[float]]] = None,       
) -> List[np.ndarray]:
       height,width=size[0],size[1] 
       images=[
              resize(image=image,size=(height,width),resample=resample) for image in images
       ]
       #conver each image into the numpy array
       images=[np.array(image) for image in  images]
       #Rescale the pixel values to be in the range [0,1]
       images=[rescale(image,scale=rescale_factor) for image in images]
       #normalize the image to have mean =0 and standard deviation to 1
       images= [normalize(image,mean=image_mean,std=image_std) for image in images]
       #move the channel dimension to the first dimension.The model expects images in the form of [channel,height,width] instead of [height,width,channel]
       images=[image.transpose(2,0,1) for image  in images]
       return images

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
                #this model version currently supports 1 image and 1 text(no batching yet)
                assert len(images)==1 and len(text) ==1,f"Recieved {len(images)} images for {len(text)} prompts"
                # a pipeline to  make the images resize rescale pixel values normalize and conver it  into the numpy array
                pixel_values=process_images(
                       images,
                       size=(self.image_size,self.image_size), #resize image to a fix sqaure size
                       resample=Image.Resampling.BICUBIC, #this control how resizing is done. better quality than nearest neighbour
                       rescale_factor=1/255.0, #rescaled to 0-1
                       image_mean=IMAGENET_STANDARD_MEAN,
                       image_std=IMAGENET_STANDARD_STD)    
                #convert the list of numpy arrays to a single numpy array with shape [Batch_size,Channel,Height,Width]
               
                #List of images → Batched array Before: [ [3, 224, 224] ] After:  [1, 3, 224, 224]
                pixel_values=np.stack(pixel_values,axis=0) #conver the list on tensor into one big tensor
                #conversion to the torch tensor  from the numpy array with the shape of [Batch,Channels,Height,Width]
                pixel_values=torch.tensor(pixel_values)
                #creates the token for the text and placeholder for the image tokens
                input_strings = [
                       add_image_tokens_to_prompt(
                              prefix_prompt=prompt,
                              bos_token=self.tokenizer.bos_token,
                              image_seq_len=self.image_seq_len,
                              image_token=self.IMAGE_TOKEN,
                       )
                       for prompt in text 
                ]

                #Returns the input ids and attention mask as pytorch tensor
                # words->input Id's -> v
                # hey there -> [1, 4 5](input ids)->[...,...,... 1024],[....,...,.,, 1024],[...,...,...,1024]
                inputs = self.tokenizer(
                       input_strings,
                       return_tensors="pt",
                       padding=padding,
                       truncation=truncation,
                )
                return_data = {"pixel_values":pixel_values, **inputs}

                return return_data
