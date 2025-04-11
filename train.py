from header import *
from dataset import load_dataset
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--scannet_file', type=str, help='pkl file containing the data of Scannet')
    parser.add_argument('--referit3D_file', type=str)
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--mode', type=str, default='train', help='train or test or validation')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='../ckpt/delta_ckpt/baseline/7b_tiva_v0/')
    parser.add_argument('--log_path', type=str, default='../ckpt/delta_ckpt/baseline/7b_tiva_v0/log/')
    parser.add_argument('--assets_path', type=str, default='./assets/')
    parser.add_argument('--use_label', type=bool, default=False)

    # model configurations
    parser.add_argument('--max_length', type=int, default=512)  # the maximum input sequence length for LLMs
    parser.add_argument('--stage', type=int, default=1)  # the training stage
    return parser.parse_args()


def initialize_distributed(args):
    args['master_ip'] = os.getenv('MASTER_ADDR', 'localhost')
    args['master_port'] = 6000 #os.getenv('MASTER_PORT', '6000')
    args['world_size'] = int(os.getenv('WORLD_SIZE', '1'))
    args['local_rank'] = int(os.getenv('RANK', '0')) % torch.cuda.device_count()
    device = args['local_rank'] % torch.cuda.device_count()
    torch.cuda.set_device(device)
    init_method = f'tcp://{args["master_ip"]}:{args["master_port"]}'
    print(f"Initialized DeepSpeed with master IP {args['master_ip']}, port {args['master_port']}, rank {args['local_rank']}, and world size {args['world_size']}")
    deepspeed.init_distributed(dist_backend='nccl', init_method=init_method, rank=args['local_rank'], world_size=args['world_size'])    
    #deepspeed.init_distributed(dist_backend='nccl')


def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def config_env(args):
    args['root_dir'] = '../'
    # args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    initialize_distributed(args)
    set_random_seed(args['seed'])


def build_directory(path):
    if os.path.exists(path):
        pass
    else:  # recursively construct directory
        os.makedirs(path, exist_ok=True)


def main(**args):
    config_env(args)
    print(args)
    args['ds_config_path'] = f'dsconfig/stage_{args["stage"]}.json'
    dschf = HfDeepSpeedConfig(args['ds_config_path'])
    args['dschf'] = dschf

    build_directory(args['save_path'])
    build_directory(args['log_path'])

    if args['log_path']:
        logging.basicConfig(
            format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            level=logging.DEBUG,
            filename=f'{args["log_path"]}/train_{time.asctime()}.log',
            filemode='w'
        )
    train_data, train_iter, sampler = load_dataset(args)

    train_num = train_data.__len__()
    length = args['epochs'] * train_num // args['world_size'] // dschf.config[
        'train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * train_num // dschf.config['train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()

    # begin to train
    pbar = tqdm(total=length)  # maximum total number
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for i, batch in enumerate(train_iter):
            agent.train_model(
                batch, 
                current_step=current_step,
                pbar=pbar
            )
            current_step += 1
    # save at the end of the training
    torch.distributed.barrier()
    agent.save_model(args['save_path'], current_step)


if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
