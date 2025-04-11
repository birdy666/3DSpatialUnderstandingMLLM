from typing import List
import json
import csv

import torch
from header import *
import torch.nn.functional as F
from .modeling_llama import LlamaForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList
from .common.utils import *
from .visual_encoder import VisualEncoder

def create_mapping(input_dim, output_dim, dropout_rate=0.1):
    return nn.Sequential(nn.Linear(input_dim, output_dim//4),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim//4),
                        nn.Linear(output_dim//4, output_dim//2),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim//2),
                        nn.Linear(output_dim//2, output_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout_rate),
                        nn.LayerNorm(output_dim),
                        )

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    return list(lora_module_names)


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stop_tokens: List[List[int]] = None, encounters: int = 1):
        super().__init__()
        self.stop_tokens = stop_tokens  # List of lists of token IDs
        self.ENCOUNTERS = encounters
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # Flatten the 2D list into a 1D list
        input_ids_list = [token for sublist in input_ids.tolist() for token in sublist]

        for stop_seq in self.stop_tokens:
            for i in range(len(input_ids_list) - len(stop_seq) + 1):
                if input_ids_list[i:i+len(stop_seq)] == stop_seq:
                    return True
        return False

class BaselineModel(nn.Module):
    def __init__(self, **args):
        super(BaselineModel, self).__init__()
        self.args = args
        self.max_length = args['max_length']
        self.device = torch.cuda.current_device()
        self.stage = args['stage']
        self.max_anchor = 3
        self.max_distractor = 5
        self.visual_hidden_size = 768 

        self.vicuna_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt', self.args['vicuna_version'])
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)
        if self.stage == 1:
            print("Freezing the LLaMa ...")
            for param in self.llama_model.parameters():
                param.requires_grad = False
            self.llama_model.eval()
        else:
            print("Applying LoRA ...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                bias=self.args['lora_bias'],
                target_modules=find_all_linear_names(self.llama_model)
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        print('Language decoder initialized.')
        print(f'Initializing tokenizer from {self.vicuna_ckpt_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(self.vicuna_ckpt_path)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        self._add_pc_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')
        self.special_token_proj = nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)
        
        self.llama_proj = create_mapping(self.visual_hidden_size, self.llama_model.config.hidden_size)
        self.spatial_enc = create_mapping(8, self.llama_model.config.hidden_size)
                
        self.input_embeddings = self.llama_model.get_input_embeddings()
        self.mse_loss = torch.nn.MSELoss()
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)

    def _add_pc_token(self):
        # Add an image token for loss masking (and visualization) purposes.
        self.llama_tokenizer.add_tokens(["<PC>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</PC>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Anchor>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Position>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Target>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["<Distractor>"])  # add special image token to tokenizer
    
    def special_tokens_embed(self, batch_size):
        PC_start = self.llama_tokenizer('<PC>', return_tensors="pt", add_special_tokens=False).to(self.device)
        PC_start = self.llama_model.model.model.embed_tokens(PC_start.input_ids).expand(batch_size, -1, -1)
        assert PC_start.shape[1] == 1, PC_start.shape[1]
        PC_end = self.llama_tokenizer('</PC>', return_tensors="pt", add_special_tokens=False).to(self.device)
        PC_end = self.llama_model.model.model.embed_tokens(PC_end.input_ids).expand(batch_size, -1, -1)
        Anchor = self.llama_tokenizer('<Anchor>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Anchor = self.llama_model.model.model.embed_tokens(Anchor.input_ids).expand(batch_size, -1, -1)
        Position = self.llama_tokenizer('<Position>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Position = self.llama_model.model.model.embed_tokens(Position.input_ids).expand(batch_size, -1, -1)
        
        Target = self.llama_tokenizer('<Target>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Target = self.llama_model.model.model.embed_tokens(Target.input_ids).expand(batch_size, -1, -1)
        
        Distractor = self.llama_tokenizer('<Distractor>', return_tensors="pt", add_special_tokens=False).to(self.device)
        Distractor = self.llama_model.model.model.embed_tokens(Distractor.input_ids).expand(batch_size, -1, -1)
        
        PC_start = self.special_token_proj(PC_start)
        PC_end = self.special_token_proj(PC_end)
        Anchor = self.special_token_proj(Anchor)
        Position = self.special_token_proj(Position)
        Target = self.special_token_proj(Target)
        Distractor = self.special_token_proj(Distractor)
        return PC_start, PC_end, Anchor, Position, Target, Distractor
            
    def prompt_wrap(self, pc_embeds, ids_after_prompt, target_ids, attention_mask, mm_mask=None, padding=True):
        ids_after_prompt = ids_after_prompt.to(self.device)
        target_ids = target_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        batch_size = ids_after_prompt.shape[0]
        bos = torch.ones([batch_size, 1], dtype=ids_after_prompt.dtype, device=ids_after_prompt.device) * self.llama_tokenizer.bos_token_id  # bsz x 1
        
        # Llama has a bug handling \n###, the tokenizer will parse it as \n + # + ##        
        p_before_tokens = self.llama_tokenizer('Human:', return_tensors="pt", add_special_tokens=False).to(self.device)
        start_token_tensor = torch.tensor([835], device=p_before_tokens.input_ids.device).unsqueeze(0)
        p_before_tokens = torch.cat([start_token_tensor, p_before_tokens.input_ids], dim=1)
        
        bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens).expand(batch_size, -1, -1)  # bsz x s1 x embed_dim
        p_after_embeds = self.llama_model.model.model.embed_tokens(ids_after_prompt).expand(batch_size, -1, -1)  # bsz x s2 x embed_dim
            
        # the input contains point cloud
        if pc_embeds is not None:
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, pc_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+max_pc_embeds+s2) x embed_dim
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1] + pc_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1 + 1 + max_anchor)            
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device)      
            atts_prefix = torch.cat([atts_prefix, mm_mask], dim=1) # <PC>, target, <Anchor>, anchors, <Position>, sp, </PC>
        else: # only text as input
            inputs_embeds = torch.cat([bos_embeds, p_before_embeds, p_after_embeds], dim=1).to(self.device)  # bsz x (1+s1+s2) x embed_dim
            empty_targets = (
                torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device).fill_(-100)
            )  # bsz x (1 + s1)
            atts_prefix = torch.ones([batch_size, 1 + p_before_embeds.size()[1]], dtype=torch.long).to(self.device)  # bsz x (1 + s1)
            
        targets = torch.cat([empty_targets, target_ids], dim=1).to(self.device)  # bsz x (1 + s1 + 1 + max_anchor + s2)            
        attention_mask = torch.cat([atts_prefix, attention_mask], dim=1).to(self.device)
        if padding:
            assert inputs_embeds.size()[1] == targets.size()[1]      
            assert attention_mask.size() == targets.size()
        return inputs_embeds, targets, attention_mask
    
    @torch.no_grad()
    def get_pairwise_distance(self, x):
        #torch.set_printoptions(profile="full")
        B, N, _ = x.shape
        relative_positions = x[:, None] - x[:, :, None]
        
        # Obtain the xy distances
        xy_distances = relative_positions[..., :2].norm(dim=-1, keepdim=True) + 1e-9
        r = xy_distances.squeeze(-1)
        phi = torch.atan2(relative_positions[..., 1], relative_positions[..., 0])  # Azimuth angle
        theta = torch.atan2(r, relative_positions[..., 2])  # Elevation angle
        sin_phi = torch.sin(phi)
        cos_phi = torch.cos(phi)
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        # Append the distances to the relative_positions tensor
        relative_positions = torch.cat([relative_positions, xy_distances, sin_phi.unsqueeze(-1), cos_phi.unsqueeze(-1), 
                                    sin_theta.unsqueeze(-1), cos_theta.unsqueeze(-1)], dim=-1)        
        return relative_positions.to(self.device)

    def get_mm_embeds(self, objects_pc, instance_label, box_info, target_id, distractor_ids, anchor_ids, object_ids, need_distractors):
        B, N, _ = box_info.shape
        PC_start, PC_end, Anchor, Position, Target, Distractor = self.special_tokens_embed(batch_size=1)
        xyz = box_info[...,:3] # (B,N,3)
        relative_positions = self.get_pairwise_distance(xyz)
        output_embeds = []
        output_mask = []
        yep = torch.ones([1, 1], dtype=torch.long).to(self.device)
        nope = torch.ones([1, 1], dtype=torch.long).to(self.device)
        for i in range(B):
            distractor_objs = []
            target_idx = (object_ids[i] == target_id[i]).nonzero(as_tuple=True)[0].item()
            target_obj = {
                "pc": objects_pc[i, target_idx],
                "label": instance_label[i][target_idx],
                "box": box_info[i, target_idx],
                "idx": target_idx
            }
            for distractor_id in distractor_ids[i]:
                distractor_id_tensor = torch.tensor(distractor_id, device=object_ids.device, dtype=object_ids.dtype)
                distractor_idx = (object_ids[i] == distractor_id_tensor).nonzero(as_tuple=True)[0].item()
                distractor_objs.append({
                    "pc": objects_pc[i, distractor_idx],
                    "label": instance_label[i][distractor_idx],
                    "box": box_info[i, distractor_idx],
                    "idx": distractor_idx
                    })
            anchor_objs = []
            for anchor_id in anchor_ids[i]:
                # Find the index of each anchor_id in object_ids
                anchor_id_tensor = torch.tensor(anchor_id, device=object_ids.device, dtype=object_ids.dtype)
                anchor_idx = (object_ids[i] == anchor_id_tensor).nonzero(as_tuple=True)[0].item()
                anchor_objs.append({
                    "pc": objects_pc[i, anchor_idx],
                    "label": instance_label[i][anchor_idx],
                    "box": box_info[i, anchor_idx],
                    "idx": anchor_idx
                    })
            target_embed = self.get_label_embed([[target_obj["label"]]])
            anchor_embeds = self.get_label_embed([[anchor_objs[n]["label"] for n in range(len(anchor_objs))]])
            anchor_embeds = F.pad(anchor_embeds, (0,0,0,self.max_anchor-len(anchor_objs),0,0), 'constant', 0)
            anchor_mask = torch.cat([yep]*len(anchor_objs)+[nope]*(self.max_anchor-len(anchor_objs)), dim=1)
            taret_sp_map = relative_positions[i][target_idx]
            target_sps = taret_sp_map[[anchor_obj['idx'] for anchor_obj in anchor_objs]].unsqueeze(0)
            target_sp_embed = self.spatial_enc(target_sps.to(dtype=next(self.spatial_enc.parameters()).dtype))
            target_sp_embed = F.pad(target_sp_embed, (0,0,0,self.max_anchor-len(anchor_objs),0,0), 'constant', 0)
            distractor_with_token_embeds=[]
            if not need_distractors[i] or len(distractor_objs)==0: #len(distractor_objs)==0:                
                distractor_with_token_embeds = torch.zeros(1, self.max_distractor*(3+self.max_anchor), self.llama_model.config.hidden_size).to(self.device)
                distractor_mask = torch.cat([nope]*(self.max_distractor*(3+self.max_anchor)), dim=1)
            else:
                distractor_mask = []
                for j in range(len(distractor_objs)):
                    distractor_sp_map = relative_positions[i][distractor_objs[j]['idx']] # (N, 8)
                    distractor_sps = distractor_sp_map[[anchor_obj['idx'] for anchor_obj in anchor_objs]].unsqueeze(0) # (1, len(anchor_objs), 8)
                    distractor_sp_embed = self.spatial_enc(distractor_sps.to(dtype=next(self.spatial_enc.parameters()).dtype)) # (1, len(anchor_objs), D)
                    distractor_sp_embed = F.pad(distractor_sp_embed, (0,0,0,self.max_anchor-len(anchor_objs),0,0), 'constant', 0) # (1, max_anchor, D)   
                    distractor_with_token_embed = torch.cat([Distractor, target_embed, Position, distractor_sp_embed], dim=1) # (1, 3+max_anchor, D)
                    distractor_mask.append(torch.cat([yep]*3+[anchor_mask], dim=1)) # (1, 3+max_anchor)
                    distractor_with_token_embeds.append(distractor_with_token_embed)
                distractor_with_token_embeds = torch.cat(distractor_with_token_embeds, dim=1) # (1, len(distractor_objs)*(3+max_anchor), D)
                distractor_with_token_embeds = F.pad(distractor_with_token_embeds, (0,0,0,(self.max_distractor-len(distractor_objs))*(3+self.max_anchor),0,0), 'constant', 0) # (1, max_distractor*(3+max_anchor), D)
                distractor_mask = torch.cat(distractor_mask, dim=1)
                distractor_mask = F.pad(distractor_mask, (0,(self.max_distractor-len(distractor_objs))*(3+self.max_anchor)), 'constant', 0)
            # <PC> <Anchor> anchor_embeds <Target> target_embed <Position> target_sp_embed <Distractor> target_embed <Position> sp_embed </PC>            
            single_embeds = torch.cat([PC_start, Anchor, anchor_embeds, Target, target_embed, Position, target_sp_embed]+[distractor_with_token_embeds]+[PC_end], dim=1)
            output_embeds.append(single_embeds)
            single_mask = torch.cat([yep, yep, anchor_mask, yep, yep, yep, anchor_mask, distractor_mask, yep], dim=1)
            output_mask.append(single_mask)
        output_embeds = torch.cat(output_embeds, dim=0)
        output_mask = torch.cat(output_mask, dim=0)
        return output_embeds, output_mask
    
    @torch.no_grad()
    def get_label_embed(self, label):
        B = len(label)
        N = len(label[0])
        pad_token_id = self.llama_tokenizer.pad_token_id
        flat_labels = [lbl if lbl else pad_token_id for sublist in label for lbl in sublist]
        tokenized_flat = [self.llama_tokenizer(str(lbl), add_special_tokens=False).input_ids if lbl != pad_token_id else [pad_token_id] for lbl in flat_labels]
        max_token_num = max(len(tokens) for tokens in tokenized_flat)
        
        tokenized_inputs = [tokens + [pad_token_id] * (max_token_num - len(tokens)) for tokens in tokenized_flat]
        tokenized_inputs = torch.tensor(tokenized_inputs).to(self.device).reshape(B, N, max_token_num)
        label_embeds = self.llama_model.model.model.embed_tokens(tokenized_inputs)
        
        # Averaging the embeddings, ignoring pad tokens
        pad_mask = tokenized_inputs != pad_token_id
        pad_mask = pad_mask.unsqueeze(-1).expand_as(label_embeds)
        label_embeds = label_embeds * pad_mask 
        label_embeds = label_embeds.sum(dim=-2) / (pad_mask.sum(dim=-2) + 1e-8)
        return label_embeds
    
    def _training_stage_2(self, inputs):
        mm_embeds, mm_mask = self.get_mm_embeds(inputs['objects'], inputs['instance_label'], inputs['box_info'], inputs['target_id'], inputs['distractor_ids'], inputs['anchor_ids'], inputs['object_ids'], inputs['need_distractors'])
        
        input_ids_after_prompt, target_ids, attention_mask = process_batch_stage_2(tokenizer=self.llama_tokenizer,
                                                                      batch_of_captions=inputs['utterance'],
                                                                      max_tgt_len=self.max_length,
                                                                      prompt=self.args['prompt'])
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids_after_prompt, target_ids, attention_mask, mm_mask)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )
        loss = outputs.loss
        assert not torch.isnan(loss).any(), "loss contains NaN"
        # calculate the token accuracy
        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask 
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        
        return loss, gen_acc
    
    def forward(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        if self.stage == 1:
            loss, gen_acc = self._training_stage_1(inputs)
        elif self.stage == 2:
            loss, gen_acc = self._training_stage_2(inputs)
        else:
            raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2]")

        return loss, gen_acc, mse_loss

    def get_random_ids(self, object_ids, instance_labels, need_distractors):
        """
        object_ids: (B, N)
        instance_label: list(B, N)
        need_distractors: (B)
        """
        # 1. randomly select a object in the room as target, but this object need to have at least one distractor
        target_id, target_index = select_random_target_with_label_constraint(object_ids, instance_labels)
        # 2. if the need_distractor is true then choose 1~5 distractor which belongs to the same category
        distractor_ids = gather_same_instance_indices(object_ids, target_index, need_distractors, instance_labels)
        # 3. randomly choose 1~3 anchor, and i think at least the category should be different, also if need_distractor ==True anchor should be one
        anchor_ids = select_random_anchors(object_ids, need_distractors, instance_labels)        
        return target_id, anchor_ids, distractor_ids
        
    def evaluate(self, inputs):
        target_id, anchor_ids, distractor_ids = self.get_random_ids(inputs['object_ids'], inputs['instance_label'], inputs['need_distractors'])
        mm_embeds, mm_mask = self.get_mm_embeds(inputs['objects'], inputs['instance_label'], inputs['box_info'], target_id, distractor_ids, anchor_ids, inputs['object_ids'], inputs['need_distractors'])
        
        input_ids_after_prompt, target_ids, attention_mask = process_batch_stage_2(tokenizer=self.llama_tokenizer,
                                                                      batch_of_captions=inputs['utterance'],
                                                                      max_tgt_len=self.max_length,
                                                                      prompt=self.args['prompt'],
                                                                      padding=False) # 'generate a caption'        
        # print(input_ids)
        inputs_embeds, _, _ = self.prompt_wrap(mm_embeds, input_ids_after_prompt, target_ids, attention_mask, mm_mask, padding=False)
        stops_id = [[13, 2277, 29937], [835]]
        stopping_criteria_instance = StoppingCriteriaSub(stop_tokens=stops_id, encounters=1)
        stopping_criteria = StoppingCriteriaList([stopping_criteria_instance])
        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=512,
            top_p=0.75,
            temperature=0.5,
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )
        caption = self.llama_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        if torch.cuda.current_device() == 0:
            csv_file_path = 'output_data.csv'  # Specify your CSV file path
            process_and_append_to_csv(target_id, anchor_ids, distractor_ids, caption, inputs['stimulus_id'], csv_file_path)
            print(caption)

def process_and_append_to_csv(target_ids, anchor_ids, distractor_ids, utterances, stimulus_id, csv_file_path):
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Iterate over each item in the batch
        for i in range(len(utterances)):
            target_id = target_ids[i].item()
            anchors = [aid.item() for aid in anchor_ids[i] if aid.numel() == 1 and aid.item() != -1]
            distractors = [did.item() for did in distractor_ids[i] if did.numel() == 1 and did.item() != -1]
            utterance = utterances[i].replace('\n', '').replace('###', '')
            # Write to CSV
            writer.writerow([target_id, anchors, distractors, utterance, stimulus_id[i]])