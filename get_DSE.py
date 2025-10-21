import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import random

from dataset import load_dataset
from models import get_models
from estimator import compute_DSE

def set_seed(seed):
    random.seed(seed)             # Set seed for Python's random module
    np.random.seed(seed)          # Set seed for NumPy
    torch.manual_seed(seed)       # Set seed for PyTorch CPU
    torch.cuda.manual_seed(seed)  # Set seed for PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def process_epoch(data_loader_train, dataset_name, save_root, model_type, model_scale, model_root, num_epoch, device_id, inter_class_distance_images, normalize, n_layer, data_percent=0.001, local_clusters = 3, global_clusters = 24, batch_size = 1024):
    device = f'cuda:{device_id}'
    torch.cuda.set_device(device_id)
    print(f"Running {model_type} with checkpoint {num_epoch}.")

    tmp = f'_{local_clusters}local_clusters' 
    tmp += f'_{global_clusters}global_clusters'
    tmp += f'_{n_layer}th_layer'
    tmp += f'_datapercent_{data_percent}'
    if normalize:
        tmp += '_normalized'
    else:
        tmp += '_unnormalized'

    epoch_folder = os.path.join(save_root, f'DSE_{dataset_name}/{tmp}_bs_{batch_size}/epoch_{num_epoch}')
    os.makedirs(epoch_folder, exist_ok=True)
    
    model = get_models(model_type, model_scale, model_root, num_epoch, device)
    model.eval()

    if 'vit' in model_scale:
        L = len(model.blocks) # Number of layers in the ViT model
        H, W = 14, 14
    elif 'resnet' in model_scale:
        L = 4
        H, W = 7, 7
    elif 'swint_tiny' in model_scale:
        L = len(model.layers)
        H, W = 7, 7
    else:
        raise NotImplementedError

    n_layers = 1
    DSE_metrics = [0.0] * (n_layers)
    inter_class_distances = [0.0] * (n_layers)
    intra_class_images = [0.0] * (n_layers)
    intra_class_batchs = [0.0] * (n_layers)
    effective_dimensionalitys = [0.0] * (n_layers)
    total_samples = 0  

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(data_loader_train, ascii=True)):
            if data_percent < 1 and batch_idx / len(data_loader_train) >= data_percent:
                break
            
            torch.cuda.empty_cache()
            images, _ = data
            images = images.to(device)
            B = images.size(0)
            total_samples += 1

            # Get embeddings for all layers
            with torch.no_grad():
                if 'ijepa' in model_type or 'esvit' in model_type:
                    images_part1 = images[:B//2] 
                    images_part2 = images[B//2:] 
                    _, patch_embeddings_list_part1 = model.get_embeddings(images_part1)
                    _, patch_embeddings_list_part2 = model.get_embeddings(images_part2)
                    patch_embeddings_list = []
                    for emb1, emb2 in zip(patch_embeddings_list_part1, patch_embeddings_list_part2):
                        patch_embeddings_list.append(torch.cat((emb1, emb2), dim=0))
                else:
                    _, patch_embeddings_list = model.get_embeddings(images)

            for l in range(L - n_layers, L):
                # Patch embeddings
                p_patch = patch_embeddings_list[l]  # [B, 196, 384]
                inter_class_distance, intra_class_image, intra_class_batch, effective_dimensionality, DSE_metric = compute_DSE(p_patch, inter_class_distance_images = inter_class_distance_images, normalize = normalize,local_clusters = local_clusters, global_clusters = global_clusters)  # [B]
                
                DSE_metrics[l - L + n_layers] += DSE_metric  # Sum over batches
                inter_class_distances[l - L + n_layers] += inter_class_distance
                intra_class_images[l - L + n_layers] += intra_class_image
                effective_dimensionalitys[l - L + n_layers] += effective_dimensionality
                intra_class_batchs[l - L + n_layers] += intra_class_batch
            
            del images, p_patch
            torch.cuda.empty_cache()  # Release unused memory

    # Compute average metrics over the dataset
    avg_DSE_metrics = [metric / total_samples for metric in DSE_metrics]
    avg_inter_class_distances = [metric / total_samples for metric in inter_class_distances]
    avg_intra_class_images = [metric / total_samples for metric in intra_class_images]
    avg_effective_dimensionalitys = [metric / total_samples for metric in effective_dimensionalitys]
    avg_intra_class_batchs = [metric / total_samples for metric in intra_class_batchs]

    metrics_file = os.path.join(epoch_folder, f'metrics_epoch_{num_epoch}.txt')
    print(metrics_file)
    with open(metrics_file, 'w') as f:
        for l in range(n_layers):
            f.write(f'Layer {l + L - n_layers} DSE Metric (Original): {avg_DSE_metrics[l]}\n')
            f.write(f'Layer {l + L - n_layers} Inter-class Distance: {avg_inter_class_distances[l]}\n')
            f.write(f'Layer {l + L - n_layers} Intra-class Radius (Image): {avg_intra_class_images[l]}\n')
            f.write(f'Layer {l + L - n_layers} Effective Dimensionality: {avg_effective_dimensionalitys[l]}\n')
            f.write(f'Layer {l + L - n_layers} Intra-class Radius (Batch): {avg_intra_class_batchs[l]}\n')


def main(args):
    set_seed(args.seed)

    root = args.data_root  # Base path to data
    dataset_name = args.dataset_name
    save_root = args.save_root
    model_root = args.model_root
    epochs = list(range(args.start_epoch, args.end_epoch, args.epoch_step))
    num_gpus = args.num_gpus

    # Load dataset
    dataset_train = load_dataset(dataset_name, root, "train")
    # shuffle = False
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    processes = []
    for idx, i in enumerate(epochs):
        device_id = idx % num_gpus # available_devices[idx % num_gpus]
        p = Process(target=process_epoch, args=(data_loader_train, dataset_name, save_root, args.model_type, args.model_scale, model_root, i, device_id, args.inter_class_distance_images, args.normalize, args.n_layer, args.data_percent, args.local_clusters, args.global_clusters, args.batch_size))
        processes.append(p)
        p.start()

        if len(processes) == num_gpus:
            for p in processes:
                p.join()
            processes = []

    for p in processes:
        p.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model with DSE evaluation.")
    
    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--data_root', type=str, default='/path/to/your/dataset/')
    parser.add_argument('--save_root', type=str, required=True, help='./results')
    parser.add_argument('--model_root', type=str, required=True, help='Path to model checkpoints')
    parser.add_argument('--dataset_name', type=str, default='imagenet-1k', help='Name of the dataset')
    parser.add_argument('--model_type', type=str, default='dino', help='Model type')
    parser.add_argument('--model_scale', type=str, default='vit_small', help='Model scale (e.g., small, base, large)')
    
    # Parameters
    parser.add_argument('--normalize', action='store_true', help='Enable normalization')
    parser.add_argument('--n_layer', type=int, default=11, help='Using n-th layer')
    parser.add_argument('--inter_class_distance_images', type=int, default=8, help='Local number of clusters for DSE')

    parser.add_argument('--local_clusters', type=int, default=5, help='Number of clusters for 1 image')
    parser.add_argument('--global_clusters', type=int, default=20, help='Number of clusters for B images')
    parser.add_argument('--data_percent', type=float, default=0.001, help='Percentage of the data to use')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for data loading')
    
    # Training settings
    parser.add_argument('--start_epoch', type=int, default=10, help='Start epoch')
    parser.add_argument('--end_epoch', type=int, default=801, help='End epoch')
    parser.add_argument('--epoch_step', type=int, default=10, help='Step size for epochs')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs for parallel processing')

    args = parser.parse_args()
    
    main(args)
