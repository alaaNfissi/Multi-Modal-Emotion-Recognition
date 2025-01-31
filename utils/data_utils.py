import os
import glob
import pickle
import torch
import torch.nn.functional as F
import torchaudio
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

def load(filename, mode='rb'):
    file = open(filename, mode)
    loaded = pickle.load(file)
    file.close()
    return loaded

def get_max_len(sents):
    max_len = max([len(sent) for sent in sents])
    return max_len    

def padTensor(t: torch.tensor, targetLen: int) -> torch.tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)    


def getEmotionDict() -> Dict[str, int]:
    """
    Returns a dictionary mapping emotion labels to integer indices.
    """
    #return {'ang': 0, 'exc': 1, 'fru': 2, 'hap': 3, 'neu': 4, 'sad': 5} if you want 6emotions

    return {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}

def get_dataset_iemocap(data_folder: str, phase: str, img_interval: int) -> Dataset:
    """
    Loads the IEMOCAP dataset for a given phase (train, valid, test).

    Args:
        data_folder (str): Path to the main data folder.
        phase (str): The dataset split to load ('train', 'valid', 'test').
        img_interval (int): Interval for sampling images.
        

    Returns:
        Dataset: An instance of the IEMOCAP dataset.
    """
    main_folder = os.path.join(data_folder, 'IEMOCAP_RAW_PROCESSED')
    meta_path = os.path.join(main_folder, 'meta.pkl')
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    
    with open(meta_path, 'rb') as file:
        meta = pickle.load(file)

    emoDict = getEmotionDict()
    
    split_file = os.path.join(data_folder, 'IEMOCAP_SPLIT', f'{phase}_split7.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found at {split_file}")
    
    with open(split_file, 'r') as f:
        uttr_ids = f.read().splitlines()
    
    texts = [meta[uttr_id]['text'] for uttr_id in uttr_ids]
    labels = [emoDict.get(meta[uttr_id]['label'], -1) for uttr_id in uttr_ids]
    
    # Filter out any labels that were not found in emoDict
    filtered_data = [(uttr_id, text, label) for uttr_id, text, label in zip(uttr_ids, texts, labels) if label != -1]
    if not filtered_data:
        raise ValueError("No valid data found. Please check your labels and emotion dictionary.")
    
    filtered_uttr_ids, filtered_texts, filtered_labels = zip(*filtered_data)
    
    this_dataset = IEMOCAP(
        main_folder=main_folder,
        utterance_ids=filtered_uttr_ids,
        texts=filtered_texts,
        labels=filtered_labels,
        label_annotations=list(emoDict.keys()),
        img_interval=img_interval
    )

    return this_dataset


def get_dataset_mosei(data_folder: str, phase: str, img_interval: int):
    """
    Loads the CMU-MOSEI dataset for a given phase (train, valid, test).

    Args:
        data_folder (str): Path to the main data folder.
        phase (str): The dataset split to load ('train', 'valid', 'test').
        img_interval (int): Interval for sampling images.

    Returns:
        Dataset: An instance of the CMU-MOSEI dataset.
    """
    main_folder = os.path.join(data_folder, 'MOSEI_RAW_PROCESSED')
    meta = load(os.path.join(main_folder, 'meta.pkl'))

    ids = open(os.path.join(data_folder, 'MOSEI_SPLIT', f'reduced_{phase}_split.txt'), 'r').read().splitlines()
    texts = [meta[id]['text'] for id in ids]
    labels = [meta[id]['label'] for id in ids]


    return MOSEI(
        main_folder=main_folder,
        ids=ids,
        texts=texts,
        labels=labels,
        img_interval=img_interval
    )





class MOSEI(Dataset):
    """
    PyTorch Dataset class for the CMU-MOSEI dataset.
    """

    def __init__(self, main_folder: str, ids: List[str], texts: List[List[int]], labels: List[int], img_interval: int):
        super(MOSEI, self).__init__()
        self.ids = ids
        self.texts = texts
        self.labels = np.array(labels)
        self.main_folder = main_folder
        self.img_interval = img_interval
        self.crop = transforms.CenterCrop(360)

    def get_annotations(self) -> List[str]:
        return ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    

    def getPosWeight(self):
        """
        Computes positive weights for each class to handle class imbalance.

        Returns:
            np.ndarray: Array of positive weights.
        """
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight
    
    def getClassWeight(self):
        """
          Computes class weights using sklearn's compute_class_weight.

          Returns:
            np.ndarray: Array of class weights.
        """
        labels=np.argmax(self.labels, axis=1)
        classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        return  class_weights
     
    
    def sample_imgs_by_interval(self, folder: str, fps: Optional[int] = 30) -> List[str]:
        """
        Samples image paths based on the specified interval.

        Args:
            folder (str): Utterance folder path.
            fps (Optional[int], optional): Frames per second of the original video. Defaults to 30.

        Returns:
            List[str]: List of sampled image file paths.
        """
        files = glob.glob(f'{folder}/*')
        nums = len(files) 
        step = int(self.img_interval / 1000 * fps)
        sampled = [os.path.join(folder, f) for i in range(0, nums, step) for f in files[i:i + step]]
        return sampled
    
    def sample_imgs(self, folder: str) -> List[str]:
        """
            Retrieves all image paths in the folder that start with the specified prefix.

          Args:
            folder (str): The path to the folder containing images.

         Returns:
            List[str]: A list of full paths of images matching the prefix.
        """
        files = glob.glob(f'{folder}/*')
        sampled = [os.path.join(folder, f) for f in files]
        return sampled

    def __len__(self):
        """
        Returns the total number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.array, List[torch.tensor], List[int], np.array]:
        """
        Retrieves the sample at the specified index.

        Args:
            ind (int): Index of the sample.

        Returns:
            Tuple[str, np.ndarray, torch.Tensor, List[int], np.ndarray]:
                - utterance ID
                - sampled images as a NumPy array
                - resampled waveform tensor
                - text sequence
                - label
        """
        this_id = self.ids[ind]
        sample_folder = os.path.join(self.main_folder, this_id)

        img_folder = os.path.join(self.main_folder, this_id, 'faces')  # Adjust to include 'faces'

        sampledImgs = []
        for imgPath in self.sample_imgs(img_folder):
            this_img = Image.open(imgPath)
            H = np.float32(this_img).shape[0]
            W = np.float32(this_img).shape[1]
        
            # Resize if height is greater than 360
            if H!=48 or W!=48:
                resize = transforms.Resize([48, 48])
                this_img = resize(this_img)

            #this_img = self.crop(this_img)  # Ensure this function crops appropriately
            sampledImgs.append(np.float32(this_img))
    
        sampledImgs = np.array(sampledImgs)
        
        
        waveform, sr = torchaudio.load(os.path.join(sample_folder, f'audio.wav'))
        resampler = torchaudio.transforms.Resample(sr, 8000)
        resampled_waveform = resampler(waveform)
        
        max_len=83000 
        if resampled_waveform.size(1) > max_len:
            resampled_waveform = resampled_waveform[:, :max_len]
        elif resampled_waveform.size(1) < max_len:
            # Padding if the length is less than 16000
            resampled_waveform = torch.nn.functional.pad(resampled_waveform, (0, max_len - resampled_waveform.size(1)))

        
        
        

        return this_id, sampledImgs, resampled_waveform, self.texts[ind], self.labels[ind]
    

class IEMOCAP(Dataset):
    """
    PyTorch Dataset class for the IEMOCAP dataset.
    """

    def __init__(
        self, 
        main_folder: str, 
        utterance_ids: List[str], 
        texts: List[List[int]], 
        labels: List[int],
        label_annotations: List[str], 
        img_interval: int
    ):
        super(IEMOCAP, self).__init__()
        self.utterance_ids = utterance_ids
        self.texts = texts
        self.labels = F.one_hot(torch.tensor(labels)).numpy()
        self.label_annotations = label_annotations

        self.utteranceFolders = {
            folder.split('\\')[-1]: folder
            for folder in glob.glob(os.path.join(main_folder, '**/*'), recursive=True)
        }
        self.img_interval = img_interval

    def get_annotations(self) -> List[str]:
        """
        Returns the list of label annotations.

        Returns:
            List[str]: List of emotion labels.
        """
        return self.label_annotations

    def use_left(self, utteranceFolder: str) -> bool:
        """
        Determines whether to use the left or right side based on folder naming.

        Args:
            utteranceFolder (str): The utterance folder name.

        Returns:
            bool: True if left side is to be used, False otherwise.
        """
        entries = utteranceFolder.split('_')
        return entries[0][-1] == entries[-1][0]

    def sample_imgs_by_interval(self, folder: str, imgNamePrefix: str, fps: Optional[int] = 30) -> List[str]:
        """
        Samples image paths based on the specified interval.

        Args:
            folder (str): Utterance folder path.
            imgNamePrefix (str): Prefix of the image filenames.
            fps (Optional[int], optional): Frames per second of the original video. Defaults to 30.

        Returns:
            List[str]: List of sampled image file paths.
        """
        files = glob.glob(os.path.join(folder, '*.jpg'))
        nums = (len(files) - 5) // 2  # Adjust based on actual file naming convention
        step = int(self.img_interval / 1000 * fps)
        sampled_indices = list(range(0, nums, step))
        sampled = [os.path.join(folder, f'{imgNamePrefix}{i}.jpg') for i in sampled_indices]
        return sampled
    
    def sample_imgs(self, folder: str, imgNamePrefix: str) -> List[str]:
        """
            Retrieves all image paths in the folder that start with the specified prefix.

          Args:
            folder (str): The path to the folder containing images.
            imgNamePrefix (str): The prefix to match image filenames.

         Returns:
            List[str]: A list of full paths of images matching the prefix.
        """
        
        files = glob.glob(os.path.join(folder, '*.jpg'))
        sampled = [f for f in files if os.path.basename(f).startswith(imgNamePrefix)]
        return sampled



    def getPosWeight(self) -> np.ndarray:
        """
        Computes positive weights for each class to handle class imbalance.

        Returns:
            np.ndarray: Array of positive weights.
        """
        pos_nums = self.labels.sum(axis=0)
        neg_nums = self.__len__() - pos_nums
        pos_weight = neg_nums / pos_nums
        return pos_weight

    def getClassWeight(self) -> np.ndarray:
        """
        Computes class weights using sklearn's compute_class_weight.

        Returns:
            np.ndarray: Array of class weights.
        """
        labels = np.argmax(self.labels, axis=1)
        classes = np.unique(labels)
        class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
        return class_weights

    def __len__(self) -> int:
        """
        Returns the total number of samples.

        Returns:
            int: Number of samples.
        """
        return len(self.utterance_ids)

    def __getitem__(self, ind: int) -> Tuple[str, np.ndarray, torch.Tensor, List[int], np.ndarray]:
        """
        Retrieves the sample at the specified index.

        Args:
            ind (int): Index of the sample.

        Returns:
            Tuple[str, np.ndarray, torch.Tensor, List[int], np.ndarray]:
                - utterance ID
                - sampled images as a NumPy array
                - resampled waveform tensor
                - text sequence
                - label
        """
        uttrId = self.utterance_ids[ind]
        uttrFolder = self.utteranceFolders.get(uttrId, None)
        if uttrFolder is None:
            raise FileNotFoundError(f"Utterance folder for ID {uttrId} not found.")

        use_left = self.use_left(uttrId)
        suffix = 'L' if use_left else 'R'
        audio_suffix = 'L' if use_left else 'R'
        imgNamePrefix = f'image_{suffix}_'

        sampledImgs = np.array([
            np.float32(Image.open(imgPath).resize((128, 128)))
            for imgPath in self.sample_imgs(uttrFolder, imgNamePrefix)
            if os.path.exists(imgPath)
        ])

        # Load audio waveform
        audio_path = os.path.join(uttrFolder, f'audio_{audio_suffix}.wav')
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at {audio_path}")
        
        waveform, sr = torchaudio.load(audio_path)

        # Resample the waveform to the target sampling rate
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=8000)
        resampled_waveform = resampler(waveform)
        
        max_len = 83426
        if resampled_waveform.size(1) > max_len:
            resampled_waveform = resampled_waveform[:, :max_len]
        elif resampled_waveform.size(1) < max_len:
            # Padding if the length is less than max_len
            resampled_waveform = F.pad(resampled_waveform, (0, max_len - resampled_waveform.size(1)))

        return uttrId, sampledImgs, resampled_waveform, self.texts[ind], self.labels[ind]

def collate_fn(batch: List[Tuple[str, np.ndarray, torch.Tensor, List[int], np.ndarray]]) -> Tuple:
    """
    Custom collate function to handle batching of IEMOCAP data.

    Args:
        batch (List[Tuple[str, np.ndarray, torch.Tensor, List[int], np.ndarray]]): List of samples.

    Returns:
        Tuple: Batched data containing utterance IDs, images, image sequence lengths,
               waveforms, waveform sequence lengths, texts, and labels.
    """
    utterance_ids = []  # To store the IDs of the utterances.
    texts = []          # To store the text associated with each utterance.
    labels = []         # To store the labels for each utterance.

    newSampledImgs = None  # Variable to store concatenated image data.
    imgSeqLens = []        # To store the lengths of image sequences.

    waveforms = []
    waveformSeqLens = []
    
    for dp in batch:
        utteranceId, sampledImgs, waveform, text, label = dp
        if sampledImgs.shape[0] == 0:
            continue  # Skip samples with no images
        utterance_ids.append(utteranceId)
        texts.append(text)
        labels.append(label)

        imgSeqLens.append(sampledImgs.shape[0])
        if newSampledImgs is None:
            newSampledImgs = sampledImgs
        else:
            newSampledImgs = np.concatenate((newSampledImgs, sampledImgs), axis=0)

        waveformSeqLens.append(waveform.size(1))
        waveforms.append(waveform)

    if newSampledImgs is None:
        raise ValueError("All samples have no images. Check your dataset.")

    imgs = newSampledImgs

    return (
        utterance_ids,
        imgs,
        imgSeqLens,
        torch.cat(waveforms, dim=0),
        waveformSeqLens,
        texts,
        torch.tensor(labels, dtype=torch.float32)
    )
