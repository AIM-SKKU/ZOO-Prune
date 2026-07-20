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


from typing import List, Optional, Tuple, Union
import os 
import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

if os.environ.get("BASELINE") == "OURS" and os.environ.get("ARCHIVE_MODE") == "liuhaotian/llava-v1.6-vicuna-7b":
    from ..llava_arch_llava_next_7b import LlavaMetaModel, LlavaMetaForCausalLM
else:
    from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM



class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()



       # ========= add row rank mm projctoer
       # Low-rank parameters
        rank1 = 128   # Linear1 rank <= we can control this. Ideally [64~256]
        rank2 = 128   # Linear2 rank <= we can control this. Ideally [64~256]
        orig_net = self.model.mm_projector #linear1[1024, 4096] + gelu + linear2[4.096, 4.096]
        # -----------------------------
        # 1) Linear1 SVD 
        W1 = orig_net[0].weight.data  # [4096, 1024]
        U1, S1, Vh1 = torch.linalg.svd(W1, full_matrices=False)


        U1k = U1[:, :rank1]         # [4096, rank1]
        S1k = torch.diag(S1[:rank1]) # [rank1, rank1]
        V1k = Vh1[:rank1, :]        # [rank1, 1024]


        linear1a = nn.Linear(1024, rank1, bias=False)
        linear1b = nn.Linear(rank1, 4096, bias=True)


        linear1a.weight.data = V1k
        linear1b.weight.data = U1k @ S1k
        linear1b.bias.data = orig_net[0].bias.data


        # -----------------------------
        # 2) Linear2 SVD approximation
        W2 = orig_net[2].weight.data  # [4096, 4096]
        U2, S2, Vh2 = torch.linalg.svd(W2, full_matrices=False)


        U2k = U2[:, :rank2]         # [4096, rank2]
        S2k = torch.diag(S2[:rank2]) # [rank2, rank2]
        V2k = Vh2[:rank2, :]        # [rank2, 4096]


        linear2a = nn.Linear(4096, rank2, bias=False)
        linear2b = nn.Linear(rank2, 4096, bias=True)


        linear2a.weight.data = V2k
        linear2b.weight.data = U2k @ S2k
        linear2b.bias.data = orig_net[2].bias.data


        # -----------------------------
        # Low-rank mm projectoer
        self.lowrank_mm_proejector = nn.Sequential(
            linear1a,
            linear1b,
            nn.GELU(),
            linear2a,
            linear2b
        )


    def get_lowrank_mm_projector(self):
        return self.lowrank_mm_proejector




    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # return super().generate(
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds,
        #     **kwargs
        # )
        # return super().generate(
        #     position_ids=position_ids,
        #     attention_mask=attention_mask,
        #     inputs_embeds=inputs_embeds,
        #     **kwargs
        # )
        #adjust for inference time evaluation 

        try:
            has_eval_time = os.environ['EVAL_TIME']
        except KeyError:
            has_eval_time = None
        
        #if has_eval_time and os.environ['EVAL_TIME'].lower()=='true':
        #    print("before generation memory:", (torch.cuda.max_memory_allocated(self.device)))
        generated = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if has_eval_time and os.environ['EVAL_TIME'].lower()=='true':
            print("after generation memory:", (torch.cuda.max_memory_allocated(self.device)))
            torch.cuda.reset_peak_memory_stats(self.device)
        return generated

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
