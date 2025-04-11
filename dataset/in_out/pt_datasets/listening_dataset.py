import numpy as np
from torch.utils.data import Dataset
from functools import partial
from .utils import dataset_to_dataloader, max_io_workers

# the following will be shared on other datasets too if not, they should become part of the ListeningDataset
# maybe make SegmentedScanDataset with only static functions and then inherit.
from .utils import check_segmented_object_order, pad_samples, objects_bboxes
from .utils import instance_labels_of_context, mean_rgb_unit_norm_transform
from dataset.in_out.data_generation.nr3d import decode_stimulus_string
from transformers import DistilBertTokenizer, DistilBertModel

def process_text(text):
    # Remove leading spaces
    text = text.lstrip()

    # Remove the first word if it's "choose" or "find"
    words = text.split()
    if words[0].lower() in ["choose", "find", "select"]:
        text = ' '.join(words[1:])
    
    # Capitalize the first letter of the sentence
    text = text[0].upper() + text[1:]

    # Add a period at the end if there isn't one
    if text[-1] != '.':
        text += '.'

    return text

class ListeningDataset(Dataset):
    def __init__(self, references, scans, vocab, max_seq_len, points_per_object, max_distractors,
                 class_to_idx=None, object_transformation=None,
                 visualization=False):
        self.references = references
        self.scans = scans
        self.max_seq_len = max_seq_len
        self.points_per_object = points_per_object
        self.max_distractors = max_distractors
        self.max_context_size = self.max_distractors + 1  # to account for the target.
        self.class_to_idx = class_to_idx
        self.visualization = visualization
        self.object_transformation = object_transformation
        if not check_segmented_object_order(scans):
            raise ValueError
        self.category_to_objects = self.collect_category_objects(references, scans)
        
        
    def collect_category_objects(self, references, scans):
        category_objects = {}
        for index, ref in references.iterrows():
            if ref['is_train']:  # Only process if is_train is True
                scan = scans[ref['scan_id']]
                for obj in scan.three_d_objects:
                    category = obj.instance_label
                    if category not in category_objects:
                        category_objects[category] = []
                    category_objects[category].append(obj)  # Store index and object
        return category_objects

    def __len__(self):
        return len(self.references)

    def get_reference_data(self, index):
        ref = self.references.loc[index]
        scan_id = ref['scan_id']
        scan = self.scans[ref['scan_id']]
        target = scan.three_d_objects[ref['target_id']]
        # sega_update: 使用原始的token
        #tokens = np.array(self.vocab.encode(ref['tokens'], self.max_seq_len), dtype=np.long)
        ori_tokens = ref['tokens']
        tokens = " ".join(ori_tokens)
        # tokens = self.vocab(sen).input_ids
        # print(len(tokens))
        # tokens = np.array(tokens)
        # tokens = np.array([102]*(self.max_seq_len + 2 + self.max_context_size * 2))
        # tokens[:min(self.max_seq_len + 2, len(emb))] = emb[:min(self.max_seq_len + 2, len(emb))]
        is_nr3d = ref['dataset'] == 'nr3d'

        return scan, target, ref['target_id'], tokens, is_nr3d, scan_id, ref['is_train']

    def prepare_distractors(self, scan, target):
        target_label = target.instance_label

        # First add all objects with the same instance-label as the target
        distractors = [o for o in scan.three_d_objects if
                       (o.instance_label == target_label and (o != target))]

        # Then all more objects up to max-number of distractors
        already_included = {target_label}
        clutter = [o for o in scan.three_d_objects if o.instance_label not in already_included]
        np.random.shuffle(clutter)

        distractors.extend(clutter)
        distractors = distractors[:self.max_distractors]
        np.random.shuffle(distractors)

        return distractors
    
    def sample_scan_object(self, object, n_points):
        sample = object.sample(n_samples=n_points)
        xyz = sample['xyz']
        #_, xyz = self.point_wolf(xyz)
        return np.concatenate([xyz, sample['color']], axis=1)

    def __getitem__(self, index):
        res = dict()
        scan, target, target_id, tokens, is_nr3d, scan_id, is_train = self.get_reference_data(index)
        # Make a context of distractors
        context = self.prepare_distractors(scan, target)

        # Add target object in 'context' list
        target_pos = np.random.randint(len(context) + 1)
        context.insert(target_pos, target)

        # sample point/color for them
        samples = np.array([self.sample_scan_object(o, self.points_per_object) for o in context])

        # mark their classes
        # res['ori_labels'], 
        res['class_labels'] = instance_labels_of_context(context, self.max_context_size, self.class_to_idx)
        res['scan_id'] = scan_id
        box_info = np.zeros((self.max_context_size, 4+3))
        box_info[:len(context),0] = [o.get_bbox().cx for o in context]
        box_info[:len(context),1] = [o.get_bbox().cy for o in context]
        box_info[:len(context),2] = [o.get_bbox().cz for o in context]
        box_info[:len(context),3] = [o.get_bbox().volume() for o in context]
        box_info[:len(context),4] = [o.get_bbox().lx for o in context]
        box_info[:len(context),5] = [o.get_bbox().ly for o in context]
        box_info[:len(context),6] = [o.get_bbox().lz for o in context]
        box_corners = np.zeros((self.max_context_size, 8, 3))
        box_corners[:len(context)] = [o.get_bbox().corners for o in context]
        if self.object_transformation is not None:
            samples = self.object_transformation(samples)

        res['context_size'] = len(samples)
        instance_label = [o.instance_label for o in context]
        # take care of padding, so that a batch has same number of N-objects across scans.
        res['objects'], res['instance_label'], res['objects_mask'], res['objects_color'] = pad_samples(samples, instance_label, self.max_context_size)
        # Get a mask indicating which objects have the same instance-class as the target.
        target_class_mask = np.zeros(self.max_context_size, dtype=np.bool)
        target_class_mask[:len(context)] = [target.instance_label == o.instance_label for o in context]


        res['target_class'] = self.class_to_idx[target.instance_label]
        res['target_class_mask'] = target_class_mask
        res['tokens'] = tokens
        res['is_nr3d'] = is_nr3d
        res['box_info'] = box_info
        res['box_corners'] = box_corners
        res['target_id'] = target_id

        if True:#self.visualization:
            distrators_pos = np.zeros((6))  # 6 is the maximum context size we used in dataset collection
            object_ids = np.full((self.max_context_size), -1)
            j = 0
            for k, o in enumerate(context):
                if o.instance_label == target.instance_label and o.object_id != target.object_id:
                    distrators_pos[j] = k
                    j += 1
            for k, o in enumerate(context):
                object_ids[k] = o.object_id
            res['utterance'] = process_text(self.references.loc[index]['utterance'])
            res['stimulus_id'] = self.references.loc[index]['stimulus_id']
            res['distrators_pos'] = distrators_pos
            res['object_ids'] = object_ids
            res['target_object_id'] = target.object_id
            anchor_ids_str = self.references.loc[index]['anchor_ids']            
            numbers_str = anchor_ids_str.strip("[]").split(",")
            anchor_ids = [int(number.strip()) for number in numbers_str]
            res['anchor_ids'] = anchor_ids
            res['need_distractors'] = self.references.loc[index]['need_distractors']
            
            distractor_ids_str = self.references.loc[index]['distractor_ids']            
            numbers_str = distractor_ids_str.strip("[]").split(",")
            distractor_ids = [int(number.strip()) for number in numbers_str]
            res['distractor_ids'] = distractor_ids
            """print("target_id: " + str(res['target_id']))
            print("object_ids: " + str(res['object_ids']))
            print("target_object_id: " + str(res['target_object_id']))
            print("stimulus_id: " + str(res['stimulus_id']))
            print("anchor_ids: " + str(res['anchor_ids']))"""
        """"# Identify objects of the same category
        if is_train:
            for k, o in enumerate(context):
                # Randomly select an index from the list of objects in the same category
                random_index = np.random.choice(len(self.category_to_objects[o.instance_label]))

                # Get the corresponding ThreeDObject
                selected_object = self.category_to_objects[o.instance_label][random_index]

                # Extract the point cloud from the selected object
                selected_point_cloud = self.sample_scan_object(selected_object, self.points_per_object)

                # Assign the extracted point cloud to the current object
                res['objects'][k] = selected_point_cloud"""
        return res


def make_data_loaders(args, referit_data, vocab, class_to_idx, scans, mean_rgb, mode=None):
    n_workers = 4 #config.n_workers
    if n_workers == -1:
        n_workers = max_io_workers()

    data_loaders = dict()
    sampler = dict()
    dataset = dict()
    is_train = referit_data['is_train']
    splits = ['train', 'test']
    object_transformation = partial(mean_rgb_unit_norm_transform, mean_rgb=mean_rgb,
                                    unit_norm=True) #config.unit_sphere_norm

    for split in splits:
        mask = is_train if split == 'train' else ~is_train
        d_set = referit_data[mask]
        d_set.reset_index(drop=True, inplace=True)

        max_distractors = 150 if split == 'train' else 150 - 1 # config.max_distractors config.max_test_objects
        ## this is a silly small bug -> not the minus-1.

        # if split == test remove the utterances of unique targets
        if split == 'test':
            def multiple_targets_utterance(x):
                _, _, _, _, distractors_ids = decode_stimulus_string(x.stimulus_id)
                return len(distractors_ids) > 0

            multiple_targets_mask = d_set.apply(multiple_targets_utterance, axis=1)
            d_set = d_set[multiple_targets_mask]
            d_set.reset_index(drop=True, inplace=True)
            print("length of dataset before removing non multiple test utterances {}".format(len(d_set)))
            print("removed {} utterances from the test set that don't have multiple distractors".format(
                np.sum(~multiple_targets_mask)))
            print("length of dataset after removing non multiple test utterances {}".format(len(d_set)))

            assert np.sum(~d_set.apply(multiple_targets_utterance, axis=1)) == 0
            import pickle


        dataset[split] = ListeningDataset(references=d_set,
                                   scans=scans,
                                   vocab=vocab,
                                   max_seq_len=24, #config.max_seq_len,
                                   points_per_object=1024, #config.points_per_object,
                                   max_distractors=max_distractors,
                                   class_to_idx=class_to_idx,
                                   object_transformation=object_transformation,
                                   visualization=False)
        
        seed = None
        if split == 'test':
            seed = 2077 #config.random_seed

        data_loaders[split], sampler[split] = dataset_to_dataloader(args, dataset[split], split, 12, n_workers, seed=seed)
        
    return dataset["train"], data_loaders["train"], sampler["train"]
