# main.py

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_utils import get_dataset_iemocap, collate_fn, get_dataset_mosei
from models.model import MODEL
from trainers.trainer_definitions import EmoTrainer
from utils.loss_functions import FocalLoss, FocalLossBCE


if __name__ == "__main__":
    start = time.time()

    args = {
        'batch_size': 4,
        'learning_rate': 4.5e-6,
        'weight_decay': 0.05,
        'epochs': 30,
        'early_stop': 6,
        'cuda': '1',
        'clip': -1.0,
        'scheduler': True,
        'seed': 42,
        'loss': 'focal',
        'optim': 'adam',
        'text_lr_factor': 10,
        'model': 'mmer',
        'model_name':'model.pth',
        'text_model_size': 'base',
        'fusion': 'early',
        'feature_dim': 256,
        'sparse_threshold': 0.9,
        'hfc_sizes': [300, 144, 35],
        'trans_dim': 64,
        'trans_nlayers': 4,
        'trans_nheads': 4,
        'audio_feature_type': 0,
        'num_emotions': 4,
        'img_interval': 500,
        'hand_crafted': False,
        'text_max_len': 100,
        'datapath': '',  # Add your data path 
        'dataset': 'iemocap', # or mosei for CMU-mosei
        'modalities': 'tva',
        'valid': False,
        'test': False,
        'ckpt': '',
        'ckpt_mod': 'tva',
        'dropout': 0.1,
        'num_layers': 1,
        'hidden_size': 300,
        'feature_dim': 256,
    }

    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Start loading the data....")

    if args['dataset'] == 'iemocap':
        train_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='train',
            img_interval=args['img_interval'],
           
        )
        valid_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='valid',
            img_interval=args['img_interval'],
           
        )
        test_dataset = get_dataset_iemocap(
            data_folder=args['datapath'],
            phase='test',
            img_interval=args['img_interval'],
           
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args['batch_size'],
            shuffle=True,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args['batch_size'],
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
            drop_last=True
        )
    elif args['dataset'] == 'mosei':
        train_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='train', img_interval=args['img_interval'], drop_last=True)
        valid_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='valid', img_interval=args['img_interval'], drop_last=True)
        test_dataset = get_dataset_mosei(data_folder=args['datapath'], phase='test', img_interval=args['img_interval'], drop_last=True)

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0,  collate_fn=collate_fn,drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn,drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0, collate_fn=collate_fn,drop_last=True)  

    print(f'# Train samples = {len(train_loader.dataset)}')
    print(f'# Valid samples = {len(valid_loader.dataset)}')
    print(f'# Test samples = {len(test_loader.dataset)}')

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader,
        'test': test_loader
    }

    lr = args['learning_rate']
    if args['model'] == 'mmer':
        model = MODEL(args=args, device=device)
        model = model.to(device=device)

        # When using a pre-trained text modal, you can use text_lr_factor to give a smaller learning rate to the textual model parts
        if args['text_lr_factor'] == 1:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args['learning_rate'],
                weight_decay=args['weight_decay']
            )
        else:
            optimizer = torch.optim.Adam([
                {'params': model.T.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.t_out.parameters(), 'lr': lr / args['text_lr_factor']},
                {'params': model.V.parameters()},
                {'params': model.v_flatten.parameters()},
                {'params': model.v_transformer.parameters()},
                {'params': model.v_out.parameters()},
                {'params': model.A.parameters()},
                {'params': model.v_cross_modal_attention.parameters()},
                {'params': model.a_cross_modal_attention.parameters()},
                {'params': model.a_flatten.parameters()},
                {'params': model.a_out.parameters()},
                {'params': model.weighted_fusion.parameters()},
            ], lr=lr, weight_decay=args['weight_decay'])

    else:
        raise ValueError('Incorrect model name!')

    if args['scheduler']:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args['epochs'] * len(train_loader.dataset) // args['batch_size']
        )
    else:
        scheduler = None

  
    if args['loss'] == 'bce':
        pos_weight = train_dataset.getPosWeight()
        pos_weight = torch.tensor(pos_weight).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif args['loss'] == 'focal':
        class_weights = train_dataset.getClassWeight()
        
        if args['dataqset']=='iemocap':
            criterion = FocalLoss(alpha=torch.FloatTensor(class_weights), gamma=2)
        else:
            criterion = FocalLossBCE(alpha=torch.FloatTensor(class_weights), gamma=2)

    if args['dataset'] in ['iemocap', 'mosei']:
        dataloader = dataloaders['train']
        pbar = tqdm(dataloader, desc='Train')

        # with torch.autograd.set_detect_anomaly(True):
        trainer = EmoTrainer(
            args,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            dataloaders
        )

    if args['test']:
        trainer.test()
    elif args['valid']:
        trainer.valid()
    else:
        trainer.train()

    end = time.time()

    print(f'Total time usage = {(end - start) / 3600:.2f} hours.')  # train
    # print(f'Total time usage = {(end - start) :.2f} s.')  # test
