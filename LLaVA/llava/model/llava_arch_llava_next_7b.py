#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

## ========================================================= ZOO-Prune: only for llava-Next =================================================

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


# from llava.model.figure_utils import *


from sklearn.cluster import KMeans
import torch.nn.functional as F
import math


import os
import matplotlib.pyplot as plt
import numpy as np


MEMORY_A = []
MEMORY_B = []


class LlavaMetaModel:
   def __init__(self, config):
       super(LlavaMetaModel, self).__init__(config)


       if hasattr(config, "mm_vision_tower"):
           self.vision_tower = build_vision_tower(config, delay_load=True)
           self.mm_projector = build_vision_projector(config)


           if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
               self.image_newline = nn.Parameter(
                   torch.empty(config.hidden_size, dtype=self.dtype)
               )


   def get_vision_tower(self):
       vision_tower = getattr(self, 'vision_tower', None)
       if type(vision_tower) is list:
           vision_tower = vision_tower[0]
       return vision_tower


   def initialize_vision_modules(self, model_args, fsdp=None):
       vision_tower = model_args.vision_tower
       mm_vision_select_layer = model_args.mm_vision_select_layer
       mm_vision_select_feature = model_args.mm_vision_select_feature
       pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
       mm_patch_merge_type = model_args.mm_patch_merge_type


       self.config.mm_vision_tower = vision_tower


       if self.get_vision_tower() is None:
           vision_tower = build_vision_tower(model_args)


           if fsdp is not None and len(fsdp) > 0:
               self.vision_tower = [vision_tower]
           else:
               self.vision_tower = vision_tower
       else:
           if fsdp is not None and len(fsdp) > 0:
               vision_tower = self.vision_tower[0]
           else:
               vision_tower = self.vision_tower
           vision_tower.load_model()


       self.config.use_mm_proj = True
       self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
       self.config.mm_hidden_size = vision_tower.hidden_size
       self.config.mm_vision_select_layer = mm_vision_select_layer
       self.config.mm_vision_select_feature = mm_vision_select_feature
       self.config.mm_patch_merge_type = mm_patch_merge_type


       if getattr(self, 'mm_projector', None) is None:
           self.mm_projector = build_vision_projector(self.config)


           if 'unpad' in mm_patch_merge_type:
               embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
               self.image_newline = nn.Parameter(
                   torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
               )
       else:
           # In case it is frozen by LoRA
           for p in self.mm_projector.parameters():
               p.requires_grad = True


       if pretrain_mm_mlp_adapter is not None:
           mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
           def get_w(weights, keyword):
               return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}


           self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))




def unpad_image(tensor, original_size):
   """
   Unpads a PyTorch tensor of a padded and resized image.


   Args:
   tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
   original_size (tuple): The original size of PIL image (width, height).


   Returns:
   torch.Tensor: The unpadded image tensor.
   """
   original_width, original_height = original_size
   current_height, current_width = tensor.shape[1:]


   original_aspect_ratio = original_width / original_height
   current_aspect_ratio = current_width / current_height


   if original_aspect_ratio > current_aspect_ratio:
       scale_factor = current_width / original_width
       new_height = int(original_height * scale_factor)
       padding = (current_height - new_height) // 2
       unpadded_tensor = tensor[:, padding:current_height - padding, :]
   else:
       scale_factor = current_height / original_height
       new_width = int(original_width * scale_factor)
       padding = (current_width - new_width) // 2
       unpadded_tensor = tensor[:, :, padding:current_width - padding]

   return unpadded_tensor




class LlavaMetaForCausalLM(ABC):


   @abstractmethod
   def get_model(self):
       pass


   def get_vision_tower(self):
       return self.get_model().get_vision_tower()


   def pairwise_cosine_similarity(self, matrix):
       norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
       cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
       return cosine_similarity


   def DivPrune(self, visual_feature_vectors, image_feature_length, cosine_matrix=None, threshold_ratio=0.1):           
       threshold_terms = int(round(threshold_ratio*image_feature_length))
       if cosine_matrix is None:
           cosine_matrix = 1.0 - (self.pairwise_cosine_similarity(visual_feature_vectors))


       s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
       for i in range(threshold_terms):
           if i==0:
               m2 = cosine_matrix
           else:
               m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))


           if i==0:
               scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
           else:
               scores = torch.min(m2, dim=0).values #for distance


           phrase_to_add_idx = torch.argmax(scores)
           s[i] = phrase_to_add_idx
       return s, cosine_matrix
  




#    def select_tokens_sads(self, visual_feature_vectors, importance_scores, k):
#        """
#        Sensitivity-Aware Diverse Selection (SADS)
#        Combines DivPrune's greedy diversity with ZOO-based sensitivity.
      
#        Args:
#            visual_feature_vectors: [N_v, d] - pre_image_features[0]
#            importance_scores: [N_v] - ZOO-based sensitivity (var of similarity)
#            k: int - number of tokens to select
      
#        Returns:
#            selected_indices: [k]
#        """
#        N_v = visual_feature_vectors.shape[0]
#        device = visual_feature_vectors.device
      
#        # 1.  - cosine similarity
#        dist_matrix = 1.0 - (self.pairwise_cosine_similarity(visual_feature_vectors))


#        # print('N_v: ', N_v)
#        # print (dist_matrix.size())
#        # print (importance_scores.size())
      
#        # 2. Normalize importance_scores to [0, 1]
#        sens_weight = (importance_scores - importance_scores.min()) / (importance_scores.max() - importance_scores.min() + 1e-8)
#        # 3. Greedy selection
#        selected = torch.empty(k, dtype=torch.long, device=device)
      
#        for i in range(k):
#            if i == 0:
#                scores = sens_weight
#            else:
#                selected_dists = torch.index_select(dist_matrix, 0, selected[:i])  # [i, N_v]
#                min_dist_to_selected = selected_dists.min(dim=0).values  # [N_v]
#                scores = min_dist_to_selected * sens_weight 
  
  
#            mask = torch.ones(N_v, dtype=torch.bool, device=device)
#            if i > 0:
#                mask[selected[:i]] = False
          
#            masked_scores = scores.masked_fill(~mask, float('-inf'))
#            idx = torch.argmax(masked_scores)
#            selected[i] = idx
      
#        return selected


  

   ### ZOO-Prune for multi-frame selection
   def select_tokens_sads_multi_frames(self, visual_feature_vectors, importance_scores, frame_label, k_frame):
       """
       Sensitivity-Aware Diverse Selection (SADS) per frame
       Combines DivPrune's greedy diversity with ZOO-based sensitivity.
      
       Args:
           visual_feature_vectors: [N_v, d] - pre_image_features[0]
           importance_scores: [N_v] - ZOO-based sensitivity
           frame_label: [N_v] (labels: -9999 for image tokens, >=0 for frame index)
           k_frame: int - number of tokens to select per frame


       Returns:
           selected_indices: [num_frames * k_frame]
       """
       device = visual_feature_vectors.device
       selected_all = []


       # frames to consider (ignore -9999)
       unique_frames = torch.unique(frame_label[frame_label != -9999])


       for f in unique_frames.tolist():
           # mask tokens of this frame
           mask = (frame_label == f)
           v_sub = visual_feature_vectors[mask]   # [N_f, d]
           s_sub = importance_scores[mask]        # [N_f]


           N_f = v_sub.shape[0]               


           # cosine distance within frame
           dist_matrix = 1.0 - self.pairwise_cosine_similarity(v_sub)


           # normalize sensitivity
           sens_weight = (s_sub - s_sub.min()) / (s_sub.max() - s_sub.min() + 1e-8)


           # greedy selection
           selected = torch.empty(k_frame, dtype=torch.long, device=device)
           for i in range(k_frame):
               if i == 0:
                   scores = sens_weight
               else:
                   selected_dists = torch.index_select(dist_matrix, 0, selected[:i])
                   min_dist_to_selected = selected_dists.min(dim=0).values
                   scores = min_dist_to_selected * sens_weight


               mask_valid = torch.ones(N_f, dtype=torch.bool, device=device)
               if i > 0:
                   mask_valid[selected[:i]] = False


               masked_scores = scores.masked_fill(~mask_valid, float('-inf'))
               idx = torch.argmax(masked_scores)
               selected[i] = idx


           # map back to global indices
           global_idx = mask.nonzero(as_tuple=True)[0][selected]
           selected_all.append(global_idx)


       # concat all frames
       selected_all = torch.cat(selected_all, dim=0)

       return selected_all




   def zoo_prune_get_sensitivity(self, original_features):
       """
       Compute token-wise sensitivity scores using random gradient estimation (RGE).


       Args:
           original_features:
               Tensor of shape [N_v, d_v] or [N_i, N_v, d_v]
               - N_i: number of image patches (multi-patch case)
               - N_v: number of vision tokens per patch
               - d_v: feature dimension


       Returns:
           sensitivity_score:
               Tensor of shape [N_v, 1] if input was [N_v, d_v]
               Tensor of shape [N_i, N_v, 1] if input was [N_i, N_v, d_v]
       """


       # Normalize input shape: ensure [N_i, N_v, d_v]
       if original_features.ndim == 2:  # case [N_v, d_v]
           original_features = original_features.unsqueeze(0)  # -> [1, N_v, d_v]


       N_i, N_v, d_v = original_features.shape # N_i=5, N_v,=576 d_v=1024
       m = int(os.environ.get('NNOISERECOV_NUM', 64))   # number of random directions
       h = float(os.environ.get('NOISERECOV_INTENS', 1e-2))  # finite difference step size
       device, dtype = original_features.device, original_features.dtype


       # Flatten multi-patch tokens into a single sequence: [N_i*N_v, d_v]
       original_features_flat = original_features.reshape(N_i * N_v, d_v)


       # 1) Generate normalized random directions u: [m, d_v]
       u = torch.randn(m, d_v, device=device, dtype=dtype)
       u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)


       # Expand u to apply on all tokens: [m, N_i*N_v, d_v]
       u_expanded = u.unsqueeze(1).expand(-1, N_i * N_v, -1)


       # 2) Create perturbed features with +h and -h shifts
       perturb_plus = original_features_flat.unsqueeze(0) + h * u_expanded
       perturb_minus = original_features_flat.unsqueeze(0) - h * u_expanded


       # 3) Flatten perturbations for model input: [1, m*N_i*N_v, d_v]
       perturb_plus_flat = perturb_plus.reshape(m * N_i * N_v, d_v).unsqueeze(0)
       perturb_minus_flat = perturb_minus.reshape(m * N_i * N_v, d_v).unsqueeze(0)


       # 4) Pass perturbed features through the projector
    #    with torch.no_grad():
    #        proj_plus = self.get_model().mm_projector(perturb_plus_flat)
    #        proj_minus = self.get_model().mm_projector(perturb_minus_flat)


       ########  with low rank mm_projector
       projector = self.get_lowrank_mm_projector()
       with torch.no_grad():
           proj_plus = projector(perturb_plus_flat)
           proj_minus = projector(perturb_minus_flat)



       # 5) Reshape back to [m, N_i*N_v, d_l]
       d_l = proj_plus.shape[-1]
       proj_plus = proj_plus.view(m, N_i * N_v, d_l)
       proj_minus = proj_minus.view(m, N_i * N_v, d_l)


       # 6) Compute sensitivity as the average finite-difference gradient norm
       delta_proj = proj_plus - proj_minus            # [m, N_i*N_v, d_l]
       coeff = delta_proj.norm(dim=-1) / (2 * h)      # [m, N_i*N_v]
       sensitivity_score = coeff.mean(dim=0)          # [N_i*N_v]
       sensitivity_score = sensitivity_score.view(N_i, N_v, 1)


       return sensitivity_score
  


   def encode_images(self, images):
       pre_image_features = self.get_model().get_vision_tower()(images)
       image_features = self.get_model().mm_projector(pre_image_features)
       sensitivity_score = self.zoo_prune_get_sensitivity(pre_image_features)
       return image_features, sensitivity_score
   

  
#    def compute_sensitivity_end2end(self, images, num_refine=64, h=1e-2):
#        """
#        Compute sensitivity scores by perturbing vision tower encoder inputs (embeds).
#        Args:
#            embeds: torch.Tensor [1, N+1, D_model], pre-encoder embeddings
#            num_refine: number of perturbation directions (m)
#            h: step size
#            noise_scale: scale for Gaussian noise (optional, can be 0)
#        """
#        device = images.device
#        dtype = images.dtype
#        m = num_refine
#        h = h *10


#        # === Step 1: replicate embeds to batch m ===
#        vision_wrapper = self.get_model().get_vision_tower()
#        vision_model = vision_wrapper.vision_tower  # HuggingFace CLIPVisionModel
#        # inference mode
#        vision_wrapper.eval()
#        vision_model.eval()
  
#        # === Step 1: Patch Embedding ===
#        embeds = vision_wrapper.vision_tower.vision_model.embeddings(
#            images.to(device=self.device, dtype=self.dtype)
#        )       
#        embeds = vision_wrapper.vision_tower.vision_model.pre_layrnorm(embeds)
#        embeds_rep = embeds.repeat(m, 1, 1).contiguous()  # [m, N+1, D_model]




#        # === Step 2: create noise directions ===
#        D_model = embeds.size(-1)
#        u = torch.randn(m, D_model, device=device, dtype=dtype)
#        u = u / (u.norm(dim=-1, keepdim=True) + 1e-12)  # [m, D_model]
#        # expand to tokens: [m, N+1, D_model]
#        u_expanded = u.unsqueeze(1).expand(-1, embeds_rep.size(1), -1)


#        if (embeds_rep.size(0) > u_expanded.size(0)):
#            embeds_rep = embeds_rep[:u_expanded.size(0), ...]




#        # === Step 3: perturb embeds ===
#        perturb_plus = embeds_rep + h * u_expanded
#        perturb_minus = embeds_rep - h * u_expanded


#        vision_wrapper = self.get_model().get_vision_tower()
#        vision_model = vision_wrapper.vision_tower


#        # === Step 4: run encoder ===
#        with torch.no_grad():
#            encoder_outs_plus = vision_model.vision_model.encoder(
#                inputs_embeds=perturb_plus,
#                attention_mask=None,
#                causal_attention_mask=None,
#                output_hidden_states=True,
#                return_dict=True,
#            )
#            encoder_outs_minus = vision_model.vision_model.encoder(
#                inputs_embeds=perturb_minus,
#                attention_mask=None,
#                causal_attention_mask=None,
#                output_hidden_states=True,
#                return_dict=True,
#            )


#        # === Step 5: feature selection (remove CLS, etc.) ===
#        features_plus = vision_wrapper.feature_select(encoder_outs_plus).to(dtype)   # [m, N_v, d_v]
#        features_minus = vision_wrapper.feature_select(encoder_outs_minus).to(dtype) # [m, N_v, d_v]


#        # === Step 6: project ===
#        proj_plus = self.get_model().mm_projector(features_plus)   # [m, N_v, d_l]
#        proj_minus = self.get_model().mm_projector(features_minus) # [m, N_v, d_l]


#        # === Step 7: compute sensitivity ===
#        delta_proj = proj_plus - proj_minus          # [m, N_v, d_l]
#        coeff = delta_proj.norm(dim=-1) / (2 * h)    # [m, N_v]
#        importance_scores = coeff.mean(dim=0)        # [N_v]


#        return importance_scores


  
  
#    def encode_images_with_noise(self, images,  noise_scale=0.01, num_refine=64, h=1e-2,  debug=True):
#        import inspect


#        vision_wrapper = self.get_model().get_vision_tower()
#        vision_model = vision_wrapper.vision_tower  # HuggingFace CLIPVisionModel
#        # inference mode
#        vision_wrapper.eval()
#        vision_model.eval()
   


#        # === Step 1: Patch Embedding ===
#        embeds = vision_wrapper.vision_tower.vision_model.embeddings(images.to(device=self.device, dtype=self.dtype))       
#        embeds = vision_wrapper.vision_tower.vision_model.pre_layrnorm(embeds)


#        # --- Add noise here ---
#        noise = torch.randn_like(embeds) * noise_scale
#        embeds_noisy = embeds + noise


#        # === Step 2: Transformer Blocks ===
#        encoder_outputs = vision_model.vision_model.encoder(
#            embeds,
#            attention_mask=None,
#            causal_attention_mask=None,
#            output_hidden_states=True,   # 필요하면 True로
#            return_dict=True
#        )


#        my_implementation_final_clean =  vision_wrapper.feature_select(encoder_outputs).to(images.dtype)
#        image_features_clean = self.get_model().mm_projector(my_implementation_final_clean)




#        # === Step 2: Transformer Blocks ===
#        encoder_outputs_noisy = vision_model.vision_model.encoder(
#            embeds_noisy,
#            attention_mask=None,
#            causal_attention_mask=None,
#            output_hidden_states=True,   # 필요하면 True로
#            return_dict=True
#        )


#        my_implementation_final_nosiy =  vision_wrapper.feature_select(encoder_outputs_noisy).to(images.dtype)
#        image_features_noisy = self.get_model().mm_projector(my_implementation_final_nosiy)


#         # === Step 3: Sensitivity 계산 ===
#        importance_scores = self.compute_sensitivity(
#            my_implementation_final_clean.squeeze(0),  # [N_v, d_v]
#        )


#        # print(inspect.getsource(self.get_model().get_vision_tower().forward))
#        # print(inspect.getsource(vision_wrapper.vision_tower.vision_model.forward))


#        return image_features_clean, image_features_noisy




   def prepare_inputs_labels_for_multimodal(
       self, input_ids, position_ids, attention_mask, past_key_values, labels,
       images, image_sizes=None
   ):       
       vision_tower = self.get_vision_tower()
       if vision_tower is None or images is None or input_ids.shape[1] == 1:
           return input_ids, position_ids, attention_mask, past_key_values, None, labels


       if type(images) is list or images.ndim == 5:
           if type(images) is list:
               images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
           concat_images = torch.cat([image for image in images], dim=0)
           image_features, sensitivity_scores = self.encode_images(concat_images)
           split_sizes = [image.shape[0] for image in images]

           image_features = torch.split(image_features, split_sizes, dim=0)
           sensitivity_scores = torch.split(sensitivity_scores, split_sizes, dim=0)## 修改

 
           mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
           image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

           #### 修改
           image_frame_labels = [torch.arange(s.size(0), device=s.device).view(s.size(0), 1, 1).expand(s.size(0), s.size(1), 1) for s in sensitivity_scores]# [N, 576, 1]


           if mm_patch_merge_type == 'flat':
               image_features = [x.flatten(0, 1) for x in image_features]
           elif mm_patch_merge_type.startswith('spatial'):
               new_image_features = []
               new_sensitivity_scores = []
               new_image_frame_labels = []
               for image_idx, image_feature in enumerate(image_features):
                   if image_feature.shape[0] > 1:
                       base_image_feature = image_feature[0]

                        #### 修改
                       base_sensitivity_score = sensitivity_scores[image_idx][0]
                       base_image_frame_label = image_frame_labels[image_idx][0]

                       image_feature = image_feature[1:]
                       sensitivity_score  = sensitivity_scores[image_idx][1:]
                       image_frame_label  = image_frame_labels[image_idx][1:]


                       height = width = self.get_vision_tower().num_patches_per_side
                       assert height * width == base_image_feature.shape[0]
                       if image_aspect_ratio == 'anyres':
                           num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                           image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                           sensitivity_score = sensitivity_score.view(num_patch_height, num_patch_width, height, width, -1)
                           image_frame_label = image_frame_label.view(num_patch_height, num_patch_width, height, width, -1)
                       else:
                           raise NotImplementedError
                       if 'unpad' in mm_patch_merge_type:
                           # ========= for image
                           image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                           image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                           # print ('1', image_feature.size())
                           image_feature = unpad_image(image_feature, image_sizes[image_idx])
                           # print ('2', image_feature.size())
                           image_feature = torch.cat((
                               image_feature,
                               self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                           ), dim=-1)
                           # print ('3', image_feature.size())
                           image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                           # print ('4', image_feature.size())


                           # ========= for sensitivity
                           sensitivity_score = sensitivity_score.permute(4, 0, 2, 1, 3).contiguous()
                           sensitivity_score = sensitivity_score.flatten(1, 2).flatten(2, 3)
                           # print ('1', sensitivity_score.size())
                           sensitivity_score = unpad_image(sensitivity_score, image_sizes[image_idx])
                           # print ('2', sensitivity_score.size())
                           newline_importnace_infinite = torch.full((1, sensitivity_score.size(1), 1), float('-inf'), device=sensitivity_score.device)
                           sensitivity_score = torch.cat((
                               sensitivity_score,
                               newline_importnace_infinite
                           ), dim=-1)
                           sensitivity_score = sensitivity_score.flatten(1, 2).transpose(0, 1)


                           # important!! do not prune tokens for <image token>
                           inf_mask = torch.isinf(sensitivity_score)        
                           inf_indices_senstivitiy_score = inf_mask.nonzero(as_tuple=True)[0] 




                           # =======. for image frame label
                           image_frame_label = image_frame_label.permute(4, 0, 2, 1, 3).contiguous()
                           image_frame_label = image_frame_label.flatten(1, 2).flatten(2, 3)
                           image_frame_label = unpad_image(image_frame_label, image_sizes[image_idx])
                           new_image_frame_label = torch.full((1, image_frame_label.size(1), 1), float('-9999'), device=image_frame_label.device)
                           image_frame_label = torch.cat((
                               image_frame_label,
                               new_image_frame_label
                           ), dim=-1)
                           image_frame_label = image_frame_label.flatten(1, 2).transpose(0, 1)
                           # exit()


                  


                       else:
                           image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                           image_feature = image_feature.flatten(0, 3)
                       image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                       sensitivity_score = torch.cat((base_sensitivity_score, sensitivity_score), dim=0) 
                       image_frame_label = torch.cat((base_image_frame_label, image_frame_label), dim=0) 
                       # print ('5', image_feature.size())
                       # print ('5', sensitivity_score.size())
                   else:
                       image_feature = image_feature[0]
                       if 'unpad' in mm_patch_merge_type:
                           image_feature = torch.cat((
                               image_feature,
                               self.model.image_newline[None].to(image_feature.device)
                           ), dim=0)
                   new_image_features.append(image_feature)
                   new_sensitivity_scores.append(sensitivity_score)
                   new_image_frame_labels.append(image_frame_label)
               image_features = new_image_features
               sensitivity_scores = new_sensitivity_scores
               image_frame_labels = new_image_frame_labels
           else:
               raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
       else:
           # passing here
           image_features, pre_image_features = self.encode_images(images)


       # print("image_features:", image_features.size())




       # TODO: image start / end is not implemented here to support pretraining.
       if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
           raise NotImplementedError


       # Let's just add dummy tensors if they do not exist,
       # it is a headache to deal with None all the time.
       # But it is not ideal, and if you have a better idea,
       # please open an issue / submit a PR, thanks.
       _labels = labels
       _position_ids = position_ids
       _attention_mask = attention_mask
       if attention_mask is None:
           attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
       else:
           attention_mask = attention_mask.bool()
       if position_ids is None:
           position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
       if labels is None:
           labels = torch.full_like(input_ids, IGNORE_INDEX)


       # remove the padding using attention_mask -- FIXME
       _input_ids = input_ids
       input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
       labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]


       new_input_embeds = []
       new_labels = []
       cur_image_idx = 0
       # print ('input_ids', len(input_ids), input_ids[0].size())
       for batch_idx, cur_input_ids in enumerate(input_ids):
           num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
           # print ('num_images ======+> ', num_images)
           if num_images == 0:
               cur_image_features = image_features[cur_image_idx]
               cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
               cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
               new_input_embeds.append(cur_input_embeds)
               new_labels.append(labels[batch_idx])
               cur_image_idx += 1
               continue


           image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
           cur_input_ids_noim = []
           cur_labels = labels[batch_idx]
           cur_labels_noim = []
           for i in range(len(image_token_indices) - 1):
               cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
               cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
           split_sizes = [x.shape[0] for x in cur_labels_noim]
           cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
           cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
           cur_new_input_embeds = []
           cur_new_labels = []


           for i in range(num_images + 1):
               cur_new_input_embeds.append(cur_input_embeds_no_im[i])
               cur_new_labels.append(cur_labels_noim[i])
               if i < num_images:
                   cur_image_features = image_features[cur_image_idx]
                   # print("cur_image_features.size()", cur_image_features.size())
                   cur_image_idx += 1
                   cur_new_input_embeds.append(cur_image_features)
                   cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))


           cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]


           cur_new_input_embeds = torch.cat(cur_new_input_embeds)
           cur_new_labels = torch.cat(cur_new_labels)


           new_input_embeds.append(cur_new_input_embeds)
           new_labels.append(cur_new_labels)


           # print ("cur_new_input_embeds.size() :", cur_new_input_embeds.size())
           # print ("cur_new_labels ,", cur_new_labels.size())


       # Truncate sequences to max length as image embeddings can make the sequence longer
       tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
       if tokenizer_model_max_length is not None:
           new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
           new_labels = [x[:tokenizer_model_max_length] for x in new_labels]


       # Combine them
       max_len = max(x.shape[0] for x in new_input_embeds)
       batch_size = len(new_input_embeds)


       new_input_embeds_padded = []
       new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
       attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
       position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)


       for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
           cur_len = cur_new_embed.shape[0]
           if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
               new_input_embeds_padded.append(torch.cat((
                   torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                   cur_new_embed
               ), dim=0))
               if cur_len > 0:
                   new_labels_padded[i, -cur_len:] = cur_new_labels
                   attention_mask[i, -cur_len:] = True
                   position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
           else:
               new_input_embeds_padded.append(torch.cat((
                   cur_new_embed,
                   torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
               ), dim=0))
               if cur_len > 0:
                   new_labels_padded[i, :cur_len] = cur_new_labels
                   attention_mask[i, :cur_len] = True
                   position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)


       new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
       # print ("new_input_embeds_padded",len(new_input_embeds_padded))
       # print("new_input_embeds.size()", new_input_embeds.size())


       if _labels is None:
           new_labels = None
       else:
           new_labels = new_labels_padded


       if _attention_mask is None:
           attention_mask = None
       else:
           attention_mask = attention_mask.to(dtype=_attention_mask.dtype)


       if _position_ids is None:
           position_ids = None


       # ZOO-Prune
       if 'LAYER_INDEX' in os.environ:
           #print("I am called without layer 0")
           if type(image_features) == list: #this is for LLaVA 1.6
               img_feature_len = image_features[0].shape[0] #example is 2340x4096
           else: #for LLaVa 1.5
               img_feature_len = image_features.shape[1]


           if hasattr(self.config, 'img_feature_len'):
               self.config.img_feature_len = img_feature_len
           else:
               setattr(self.config, 'img_feature_len', img_feature_len)


       if 'LAYER_INDEX' in os.environ and os.environ['LAYER_INDEX']=='0':
           SYS_TOKEN_LEN = 35
           diverse_ratio = float(os.environ['SUBSET_RATIO']) #define the subset selection ratio
           cosine_matrix = None
           if type(image_features) == list: #this is for LLaVA 1.6
               img_feature_len = image_features[0].shape[0] #example is 2340x4096
               image_features= image_features[0]
               sensitivity_score = sensitivity_scores[0][:,0]
               image_frame_label = image_frame_labels[0][:,0]
           else: #for LLaVa 1.5
               img_feature_len = image_features.shape[1] #example is 2340x4096




           totat_frame_number=len( torch.unique(image_frame_label))-1


           # target_token_num =int(diverse_ratio * img_feature_len) #inf_indices_senstivitiy_score.size(0)


           diverse_ratio = float(os.environ['SUBSET_RATIO'])
           target_token_num = int(2880*diverse_ratio)
           
        #    target_token_num = 160 #- inf_indices_senstivitiy_score.size(0)
           frame_token_num = int(target_token_num / totat_frame_number)


           # 7) sensitivity to vision token (hybrid)


           selected_visual_tokens = self.select_tokens_sads_multi_frames(
               image_features,      # [N_v, d_v]
               sensitivity_score,          # [N_v]
               image_frame_label,
               frame_token_num)
          
           selected_visual_tokens = torch.cat([selected_visual_tokens, inf_indices_senstivitiy_score], dim=0)
           selected_visual_tokens = torch.unique(selected_visual_tokens)



           # ================================== for exp compare the ranking between mm_porjector

           # importance_scores_from_end2end_visiontower = self.compute_sensitivity_end2end(images, num_refine, h)
           # spearman_score = plot_token_importance_scatter(importance_scores, importance_scores_from_end2end_visiontower)
           # if spearman_score >0.3:
           #     MEMORY_A.append(importance_scores)
           #     MEMORY_B.append(importance_scores_from_end2end_visiontower)
          
           # if len(MEMORY_A)>0:
           #     plot_rank_correlation(MEMORY_A,MEMORY_B)

           # plot_topk_overlap(importance_scores, importance_scores_from_end2end_visiontower,
           #       ks=np.linspace(0.001, 1.0, 50),
           #       mode="fraction",
           #       )
          

           #============= [Original DivPrune] ==============================
           # print ("new_input_embeds[0].size()", new_input_embeds[0].size())
           # visual_tokens =new_input_embeds[0][SYS_TOKEN_LEN:SYS_TOKEN_LEN+img_feature_len]
           # print ("img_feature_len :", img_feature_len)
           # print ("visual_tokens.size():", visual_tokens.size())
           # print ('....................................')


           # selected_visual_tokens, cosine_matrix = self.DivPrune(visual_tokens, img_feature_len,cosine_matrix,threshold_ratio=diverse_ratio)
      
           selected_visual_tokens += SYS_TOKEN_LEN
           keep_indexs = torch.cat((torch.arange(SYS_TOKEN_LEN,device=new_input_embeds.device), selected_visual_tokens, torch.arange(SYS_TOKEN_LEN+img_feature_len,new_input_embeds.shape[1],device=new_input_embeds.device)))
           keep_indexs = keep_indexs.sort().values


           new_input_embeds = new_input_embeds[:,keep_indexs]
           if position_ids is not None:
               position_ids = position_ids[:,keep_indexs,:]
           if attention_mask is not None:
               attention_mask = attention_mask[:,keep_indexs]


  
       return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
  


   def initialize_vision_tokenizer(self, model_args, tokenizer):
       if model_args.mm_use_im_patch_token:
           tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
           self.resize_token_embeddings(len(tokenizer))


       if model_args.mm_use_im_start_end:
           num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
           self.resize_token_embeddings(len(tokenizer))


           if num_new_tokens > 0:
               input_embeddings = self.get_input_embeddings().weight.data
               output_embeddings = self.get_output_embeddings().weight.data


               input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                   dim=0, keepdim=True)
               output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                   dim=0, keepdim=True)


               input_embeddings[-num_new_tokens:] = input_embeddings_avg
               output_embeddings[-num_new_tokens:] = output_embeddings_avg


           if model_args.tune_mm_mlp_adapter:
               for p in self.get_input_embeddings().parameters():
                   p.requires_grad = True
               for p in self.get_output_embeddings().parameters():
                   p.requires_grad = False


           if model_args.pretrain_mm_mlp_adapter:
               mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
               embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
               assert num_new_tokens == 2
               if input_embeddings.shape == embed_tokens_weight.shape:
                   input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
               elif embed_tokens_weight.shape[0] == num_new_tokens:
                   input_embeddings[-num_new_tokens:] = embed_tokens_weight
               else:
                   raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
       elif model_args.mm_use_im_patch_token:
           if model_args.tune_mm_mlp_adapter:
               for p in self.get_input_embeddings().parameters():
                   p.requires_grad = False
               for p in self.get_output_embeddings().parameters():
                   p.requires_grad = False



