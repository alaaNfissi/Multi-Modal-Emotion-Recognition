import os
import copy
import torch
import pickle
import copy
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from tabulate import tabulate
from transformers import AlbertTokenizer
from utils.evaluations import eval_iemocap,eval_mosei


def save(toBeSaved, filename, mode='wb'):
    """
    Save an object to a file using pickle.

    Args:
        toBeSaved (object): The object to be saved.
        filename (str): The path to the file where the object will be saved.
        mode (str): The mode in which to open the file. Defaults to 'wb' (write binary).

    Returns:
        None
    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = open(filename, mode)
    pickle.dump(toBeSaved, file, protocol=4)
    file.close()

class TrainerBase():
    """
    Base class for training models.

    Attributes:
        args (dict): Configuration arguments for training.
        model (torch.nn.Module): The model to be trained.
        best_model (dict): State dictionary of the best model.
        device (str): The device to run the model on (CPU or GPU).
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        dataloaders (dict): Dictionary containing data loaders for training and validation.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        earlyStop (int): Counter for early stopping.
        saving_path (str): Directory path for saving model and statistics.
    """

    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        self.args = args
        self.model = model
        self.best_model = copy.deepcopy(model.state_dict())
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.scheduler = scheduler
        self.earlyStop = args['early_stop']

        self.saving_path = f"./savings/"

    def make_stat(self, prev, curr): 
        """
        Generate a string representation of statistics.

        Args:
            prev (list): Previous statistics.
            curr (list): Current statistics.

        Returns:
            list: A list of strings indicating the change in statistics (↑, ↓, or unchanged).
        """
        new_stats = []
        for i in range(len(prev)):
            if curr[i] > prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2191')
            elif curr[i] < prev[i]:
                new_stats.append(f'{curr[i]:.4f} \u2193')
            else:
                new_stats.append(f'{curr[i]:.4f} -')
        return new_stats



    def save_stats(self, file_name:str):
        """
        Save training statistics and configuration to a file.

        Returns:
            None
        """
        stats = {
            'args': self.args,
            'train_stats': self.all_train_stats,
            'valid_stats': self.all_valid_stats,
            'test_stats': self.all_test_stats,
            'best_valid_stats': self.best_valid_stats,
            'best_epoch': self.best_epoch
        }

        save(stats, os.path.join(self.saving_path, 'stats', file_name))



    def save_model(self): 
        """
        Save the best-performing model's state dictionary to a file.

        Returns:
            None
        """
        torch.save(self.best_model, os.path.join(self.saving_path, 'models', self.get_saving_file_name()))



class EmoTrainer(TrainerBase):
    """
    Trainer class for the Iemocap dataset.

    Inherits from TrainerBase and adds functionality specific to the Iemocap training process.
    """

    def __init__(self, args, model, criterion, optimizer, scheduler, device, dataloaders):
        super(EmoTrainer, self).__init__(args, model, criterion, optimizer, scheduler, device, dataloaders)
        self.args = args
        self.model=model
        self.text_max_len = args['text_max_len']
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
         
        self.all_train_stats = []
        self.all_valid_stats = []
        self.all_test_stats = []
        annotations = dataloaders['train'].dataset.get_annotations()

        if self.args['dataset']=='iemocap':
            self.eval_func = eval_iemocap 
        elif  self.args['dataset']=='mosei':  
            self.eval_func = eval_mosei

        if self.args['loss'] == 'bce' or 'focal':
            self.headers = [
                ['phase (acc)', *annotations, 'average'],
                ['phase (recall)', *annotations, 'average'],
                ['phase (precision)', *annotations, 'average'],
                ['phase (f1)', *annotations, 'average'],
                ['phase (auc)', *annotations, 'average']
            ]
            n = len(annotations) + 1
            self.prev_train_stats = [[-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n, [-float('inf')] * n]
            self.prev_valid_stats = copy.deepcopy(self.prev_train_stats)
            self.prev_test_stats = copy.deepcopy(self.prev_train_stats)
            self.best_valid_stats = copy.deepcopy(self.prev_train_stats)
        else:
            self.header = ['Phase', 'Acc', 'Recall', 'Precision', 'F1']
            self.best_valid_stats = [0, 0, 0, 0]
        self.best_epoch = -1

    def train(self):
        """
        Execute the training process for a specified number of epochs.

        Returns:
            None
        """
        for epoch in range(1, self.args['epochs'] + 1):
            print(f'=== Epoch {epoch} ===')
            train_stats, train_thresholds = self.train_one_epoch()
            valid_stats, valid_thresholds = self.eval_one_epoch()
            
            test_stats, _ = self.eval_one_epoch('test', valid_thresholds)

            print('Train thresholds: ', train_thresholds)
            print('Valid thresholds: ', valid_thresholds)

            self.all_train_stats.append(train_stats)
            self.all_valid_stats.append(valid_stats)
            self.all_test_stats.append(test_stats)

            if self.args['loss'] == 'ce' or self.args['loss'] == 'focalloss':
                train_stats_str = [f'{s:.4f}' for s in train_stats]
                valid_stats_str = [f'{s:.4f}' for s in valid_stats]
                test_stats_str = [f'{s:.4f}' for s in test_stats]
                print(tabulate([
                    ['Train', *train_stats_str],
                    ['Valid', *valid_stats_str],
                    ['Test', *test_stats_str]
                ], headers=self.header))
                if valid_stats[-1] > self.best_valid_stats[-1]:
                    self.best_valid_stats = valid_stats
                    self.best_epoch = epoch
                    self.earlyStop = self.args['early_stop']
                else:
                    self.earlyStop -= 1
            else:
                for i in range(len(self.headers)):
                    for j in range(len(valid_stats[i])):
                        is_pivot = (i == 3 and j == (len(valid_stats[i]) - 1))
                        if valid_stats[i][j] > self.best_valid_stats[i][j]:
                            self.best_valid_stats[i][j] = valid_stats[i][j]
                            if is_pivot:
                                self.earlyStop = self.args['early_stop']
                                self.best_epoch = epoch
                                self.best_model = copy.deepcopy(self.model.state_dict())
                        elif is_pivot:
                            self.earlyStop -= 1

                    train_stats_str = self.make_stat(self.prev_train_stats[i], train_stats[i])
                    valid_stats_str = self.make_stat(self.prev_valid_stats[i], valid_stats[i])
                    test_stats_str = self.make_stat(self.prev_test_stats[i], test_stats[i])

                    self.prev_train_stats[i] = train_stats[i]
                    self.prev_valid_stats[i] = valid_stats[i]
                    self.prev_test_stats[i] = test_stats[i]

                    print(tabulate([
                        ['Train', *train_stats_str],
                        ['Valid', *valid_stats_str],
                        ['Test', *test_stats_str]
                    ], headers=self.headers[i]))

        print('=== Best performance ===')
        if self.args['loss'] == 'ce':
            print(tabulate([
                [f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1]]
            ], headers=self.header))
        else:
            for i in range(len(self.headers)):
                print(tabulate([[f'Test ({self.best_epoch})', *self.all_test_stats[self.best_epoch - 1][i]]], headers=self.headers[i]))

        self.save_stats(self.args['model_name'])
        torch.save(self.model.state_dict(), self.args['model_name'])
        print('Results and model are saved!')

    def valid(self):
        """
        Validate the model on the validation set.

        Returns:
            None
        """
        valid_stats = self.eval_one_epoch()
        for i in range(len(self.headers)):
            print(tabulate([['Valid', *valid_stats[0][i]]], headers=self.headers[i]))
            print()

    def test(self):
    
        """
        Test the model on the test set.

        Returns:
            None
        """
        test_stats = self.eval_one_epoch('test')
        for i in range(len(self.headers)):
            print(tabulate([['Test', *test_stats[0][i]]], headers=self.headers[i]))
            print()
        for stat in test_stats[0]:
            for n in stat:
                print(f'{n:.4f},', end='')
        print()

    def train_one_epoch(self):
        """
        Train the model for one epoch.

        Returns:
            tuple: Training statistics and thresholds for evaluation.
        """
        self.model.train()
        if self.args['model'] == 'mme2e' :
            self.model.mtcnn.eval()
        dataloader = self.dataloaders['train']
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []

        pbar = tqdm(dataloader, desc='Train')
        #for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in dataloader:
        # with torch.autograd.set_detect_anomaly(True):
        for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in pbar:  
        
            
            
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            waveforms = waveforms.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            
            self.optimizer.zero_grad()
        
            with torch.set_grad_enabled(True):
                logits = self.model(imgs, imgLens, waveforms, waveformLens, text)
                loss = self.criterion(logits.squeeze(), Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
                
                if self.args['clip'] > 0:
                    clip_grad_norm_(self.model.parameters(), self.args['clip'])
                
                loss.backward()
                self.optimizer.step()
            
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())
            print("train loss:{:.4f}".format(epoch_loss / data_size))

            if self.scheduler is not None:
                self.scheduler.step()

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        epoch_loss /= len(dataloader.dataset)
        return self.eval_func(total_logits, total_Y)

    def eval_one_epoch(self, phase='valid', thresholds=None):
        """
        Evaluate the model for one epoch.

        Args:
            phase (str): The phase to evaluate ('valid' or 'test').
            thresholds (list): Thresholds for evaluation metrics.

        Returns:
            tuple: Evaluation statistics and thresholds.
        """
        for m in self.model.modules():
            if hasattr(m, 'switch_to_deploy'):
                m.switch_to_deploy()
        self.model.eval()

        dataloader = self.dataloaders[phase]
        epoch_loss = 0.0
        data_size = 0
        total_logits = []
        total_Y = []

        #for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in dataloader:
        pbar = tqdm(dataloader, desc=phase)

        for uttranceId, imgs, imgLens, waveforms, waveformLens, text, Y in pbar:
        
            if 'lf_' not in self.args['model']:
                text = self.tokenizer(text, return_tensors='pt', max_length=self.text_max_len, padding='max_length', truncation=True)
            else:
                imgs = imgs.to(device=self.device)

            if self.args['loss'] == 'ce':
                Y = Y.argmax(-1)

            waveforms = waveforms.to(device=self.device)
            text = text.to(device=self.device)
            Y = Y.to(device=self.device)
            
            with torch.set_grad_enabled(False):
                logits = self.model(imgs, imgLens, waveforms, waveformLens, text)
                loss = self.criterion(logits.squeeze(), Y)
                epoch_loss += loss.item() * Y.size(0)
                data_size += Y.size(0)
            
            total_logits.append(logits.cpu())
            total_Y.append(Y.cpu())

        total_logits = torch.cat(total_logits, dim=0)
        total_Y = torch.cat(total_Y, dim=0)
        epoch_loss /= len(dataloader.dataset)
        
        
        
        return self.eval_func(total_logits, total_Y, thresholds)

