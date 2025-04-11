from header import *
from .in_out.neural_net_oriented import load_scan_related_data, load_referential_data
from .in_out.neural_net_oriented import compute_auxiliary_data, trim_scans_per_referit3d_data
from .in_out.pt_datasets.listening_dataset import make_data_loaders
from .samplers import DistributedDatasetBatchSampler

def load_dataset(args):
    all_scans_in_dict, scans_split, class_to_idx = load_scan_related_data(args["scannet_file"])
    # Read the linguistic data of ReferIt3D
    referit_data = load_referential_data(args["referit3D_file"], scans_split)
    # Prepare data & compute auxiliary meta-information.
    all_scans_in_dict = trim_scans_per_referit3d_data(referit_data, all_scans_in_dict)
    mean_rgb, vocab = compute_auxiliary_data(referit_data, all_scans_in_dict)
    dataset, iter_, sampler = make_data_loaders(args, referit_data, vocab, class_to_idx, all_scans_in_dict, mean_rgb)
    
    
    return dataset, iter_, sampler

