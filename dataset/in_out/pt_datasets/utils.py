import warnings
import numpy as np
import torch
import multiprocessing as mp
from torch.utils.data import DataLoader
from dataset.samplers import DistributedBatchSampler, DistributedDatasetBatchSampler
from torch.utils.data._utils.collate import default_collate

def custom_collate_fn(batch):
    # Initialize a dictionary to store collated data
    collated_batch = {}

    # Iterate over keys in the first item to set up the structure
    # Assuming all items in the batch have the same structure
    for key in batch[0]:
        if key == 'anchor_ids' or key == 'distractor_ids' or key == 'instance_label':
            # Keep 'anchor_ids' as a list of lists
            collated_batch[key] = [d[key] for d in batch]
        else:
            # Apply default collation for all keys except 'anchor_ids'
            collated_batch[key] = default_collate([d[key] for d in batch])

    return collated_batch

def max_io_workers():
    """ number of available cores -1."""
    n = max(mp.cpu_count() - 1, 1)
    print('Using {} cores for I/O.'.format(n))
    return n


def dataset_to_dataloader(args, dataset, split, batch_size, n_workers, pin_memory=False, seed=None):
    """
    :param dataset:
    :param split:
    :param batch_size:
    :param n_workers:
    :param pin_memory:
    :param seed:
    :return:
    """
            
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    batch_size = args['world_size'] * args['dschf'].config['train_micro_batch_size_per_gpu']
    sampler = torch.utils.data.RandomSampler(dataset)
    batch_sampler = DistributedDatasetBatchSampler(dataset=dataset,
                                                        sampler=sampler,
                                                        batch_size=batch_size,
                                                        drop_last=True,
                                                        rank=rank,
                                                        world_size=world_size)
    iter_ = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=custom_collate_fn,
        num_workers=1,
        pin_memory=True
    )

    return iter_, sampler





def pad_samples(samples, instance_label, max_context_size, padding_value=1):
    n_pad = max_context_size - len(samples)

    if n_pad > 0:
        shape = (max_context_size, samples.shape[1], samples.shape[2])
        temp = np.zeros(shape, dtype=samples.dtype) * padding_value
        temp[:samples.shape[0], :samples.shape[1]] = samples
        samples = temp
        instance_label.extend(['' for _ in range(n_pad)])
    else:
        n_pad = 0
    mask = np.ones(max_context_size, dtype=bool)
    if n_pad > 0:
        mask[-n_pad:] = False
    return samples, instance_label, mask, np.mean(samples[...,3:], axis=1)


def check_segmented_object_order(scans):
    """ check all scan objects have the three_d_objects sorted by id
    :param scans: (dict)
    """
    for scan_id, scan in scans.items():
        idx = scan.three_d_objects[0].object_id
        for o in scan.three_d_objects:
            if not (o.object_id == idx):
                print('Check failed for {}'.format(scan_id))
                return False
            idx += 1
    return True


def objects_bboxes(context):
    b_boxes = []
    for o in context:
        bbox = o.get_bbox(axis_aligned=True)

        # Get the centre
        cx, cy, cz = bbox.cx, bbox.cy, bbox.cz

        # Get the scale
        lx, ly, lz = bbox.lx, bbox.ly, bbox.lz

        b_boxes.append([cx, cy, cz, lx, ly, lz])

    return np.array(b_boxes).reshape((len(context), 6))


def instance_labels_of_context(context, max_context_size, label_to_idx=None, add_padding=True):
    """
    :param context: a list of the objects
    :return:
    """
    ori_instance_labels = [i.instance_label for i in context]

    if add_padding:
        n_pad = max_context_size - len(context)
        ori_instance_labels.extend(['pad'] * n_pad)

    if label_to_idx is not None:
        instance_labels = np.array([label_to_idx[x] for x in ori_instance_labels])

    # ori_labels=[]
    # for ori_label in ori_instance_labels:
    #     ori_labels.append('[CLS] '+ori_label+' [SEP]')
    # ori_instance_labels = ' '.join(ori_labels)

    return instance_labels


def mean_rgb_unit_norm_transform(segmented_objects, mean_rgb, unit_norm, epsilon_dist=10e-6, inplace=True):
    """
    :param segmented_objects: K x n_points x 6, K point-clouds with color.
    :param mean_rgb:
    :param unit_norm:
    :param epsilon_dist: if max-dist is less than this, we apply not scaling in unit-sphere.
    :param inplace: it False, the transformation is applied in a copy of the segmented_objects.
    :return:
    """
    if not inplace:
        segmented_objects = segmented_objects.copy()

    # adjust rgb
    segmented_objects[:, :, 3:6] -= np.expand_dims(mean_rgb, 0)

    # center xyz
    if unit_norm:
        xyz = segmented_objects[:, :, :3]
        mean_center = xyz.mean(axis=1)
        xyz -= np.expand_dims(mean_center, 1)
        max_dist = np.max(np.sqrt(np.sum(xyz ** 2, axis=-1)), -1)
        max_dist[max_dist < epsilon_dist] = 1  # take care of tiny point-clouds, i.e., padding
        xyz /= np.expand_dims(np.expand_dims(max_dist, -1), -1)
        segmented_objects[:, :, :3] = xyz

    return segmented_objects
