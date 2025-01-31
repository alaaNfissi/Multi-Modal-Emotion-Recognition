from copy import deepcopy #creates a deep copy of an object, including all nested objects.
import numpy as np 
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap




def weighted_acc(preds, truths, verbose):
    """
    Computes a weighted accuracy based on true positives and true negatives, optionally printing detailed metrics.

    Args:
        preds (torch.Tensor): The predicted labels, expected to be a flattened tensor.
        truths (torch.Tensor): The ground truth labels, expected to be a flattened tensor.
        verbose (bool): If True, prints detailed metrics including TP, TN, FP, FN, Recall, and F1 score.

    Returns:
        float: The weighted accuracy.
    """
    preds = preds.view(-1)
    truths = truths.view(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / p + tn) / (2 * n)

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N', n, 'Recall', recall, "f1", f1)

    return w_acc




#Evaluates predictions and ground truth labels for the IEMOCAP dataset using various metrics including accuracy, recall, precision, F1 score, and area under the ROC curve (AUC). It also selects the best threshold for each emotion category based on F1 score if best_thresholds are not provided.
def eval_iemocap(preds, truths, best_thresholds=None):
    # emos = ["Happy", "Sad", "Angry", "Neutral"]
    
    """
    Evaluates predictions and ground truth labels for the IEMOCAP dataset using various metrics including accuracy, recall,
    precision, F1 score, and area under the ROC curve (AUC). It also selects the best threshold for each emotion category
    based on F1 score if best_thresholds are not provided.

    Args:
        preds (torch.Tensor): The predicted logits or probabilities with shape (bs, num_emotions).
        truths (torch.Tensor): The ground truth labels with shape (bs, num_emotions).
        best_thresholds (Optional[np.ndarray], optional): Predefined thresholds for each emotion category. If None,
            the function will determine the best thresholds based on F1 scores. Defaults to None.

    Returns:
        Tuple[Tuple[List[float], List[float], List[float], List[float], List[float]], Optional[np.ndarray]]:
            - A tuple containing lists of accuracies, recalls, precisions, F1 scores, and AUCs for each emotion class,
              including their average values.
            - The best thresholds used for each emotion category.
    """
   

    num_emo = preds.size(1)
    
    
    #Detach and move predictions and truths to CPU
    preds = preds.cpu().detach()
    truths = truths.cpu().detach()
    
    
    # Apply sigmoid activation to predictions
    preds = torch.sigmoid(preds)
    print(preds)
    # Calculate area under the ROC curve (AUC) for each emotion class
    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    #aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None, multi_class='ovr').tolist()

    # Append the average AUC across all classes
    aucs.append(np.average(aucs))
    
    #If best_thresholds is not provided, select the best threshold for each emotion category based on F1 score
    if best_thresholds is None:
        # select the best threshold for each emotion category, based on F1 score
        thresholds = np.arange(0.05, 1, 0.05)
        _f1s = []
        for t in thresholds:
            
            _preds = deepcopy(preds)
        
            _preds[_preds > t] = 1
            _preds[_preds <=t] = 0
            
            
           

            this_f1s = []
            #Compute F1 score for each emotion class using the current threshold
            for i in range(num_emo):
                pred_i = _preds[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i))

            _f1s.append(this_f1s)
        _f1s = np.array(_f1s)
        #Select the threshold that maximizes F1 score for each class
        best_thresholds = (np.argmax(_f1s, axis=0) + 1) * 0.05
    
    #apply thresholding using the best thresholds for each class
    # th = [0.5] * truths.size(1)
  
    preds2 = deepcopy(preds)
    for i in range(num_emo):
        
        pred = preds2[:, i]
        
        
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds2[:, i] = pred
        
        
    for row_idx in range(preds.size(0)) :
        row = preds2[row_idx, :]
        if row.sum().item() >1 or row.sum().item()==0:
            max_value, max_index = torch.max(preds[row_idx, :], dim=0)
            new_row = torch.zeros_like(row)
            new_row[max_index] = 1
            preds[row_idx] = new_row
        else:
            preds[row_idx]=preds2[row_idx]
            
        
    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds[:, i]
        truth_i = truths[:, i]
        
        # Compute accuracy, recall, precision, and F1 score for each class
        #acc = weighted_acc(pred_i, truth_i, verbose=False)
        acc = accuracy_score(truth_i, pred_i)
        recall = recall_score(truth_i, pred_i)
        precision = precision_score(truth_i, pred_i)
        f1 = f1_score(truth_i, pred_i)

        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    
    #Append average metrics across all classes
    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))


    return (accs, recalls, precisions, f1s, aucs), best_thresholds

def eval_mosei(preds, truths, best_thresholds=None):
    """
    Evaluate the CMU-MOSEI model performance based on predictions and ground truths.
    
    Args:
        preds: (bs, num_emotions) - predicted outputs
        truths: (bs, num_emotions) - true labels
        best_thresholds: (optional) - thresholds for each emotion
    Returns:
        Tuple[Tuple[List[float], List[float], List[float], List[float], List[float]], Optional[np.ndarray]]:
            - A tuple containing lists of accuracies, recalls, precisions, F1 scores, and AUCs for each emotion class,
              including their average values.
            - The best thresholds used for each emotion category.
    """

    num_emo = preds.size(1)

    # Detach and move predictions and truths to CPU
    preds = preds.cpu().detach()
    truths = truths.cpu().detach()
    
    # Apply sigmoid activation to predictions
    preds = torch.sigmoid(preds)
    # Calculate area under the ROC curve (AUC) for each emotion class
    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(np.average(aucs))  # Append the average AUC across all classes

    # If best_thresholds is not provided, select the best threshold for each emotion category based on F1 score
    if best_thresholds is None:
        thresholds = np.arange(0.4, 1, 0.01)  # More granular thresholds
        _f1s = []

        for t in thresholds:
            _preds = preds.clone()  # Use clone instead of deepcopy
            _preds[_preds > t] = 1
            _preds[_preds <= t] = 0

            this_f1s = []

            for i in range(num_emo):
                pred_i = _preds[:, i].numpy()  # Convert to numpy for sklearn
                truth_i = truths[:, i].numpy()  # Convert to numpy for sklearn
                precision = precision_score(truth_i, pred_i, zero_division=0)
                recall = recall_score(truth_i, pred_i, zero_division=0)
                this_f1s.append(2 * (precision * recall) / (precision + recall + 1e-6))  #  F1 calculation

            _f1s.append(this_f1s)

        _f1s = np.array(_f1s)
        best_thresholds = thresholds[np.argmax(_f1s, axis=0)]

    # Apply best thresholds
    for i in range(num_emo):
        pred = preds[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds[:, i] = pred

    accs, recalls, precisions, f1s = [], [], [], []

    
    
   
    for i in range(num_emo):
        pred_i = preds[:, i].numpy()  # Convert to numpy for sklearn
        truth_i = truths[:, i].numpy()  # Convert to numpy for sklearn

        # Compute accuracy, recall, precision, and F1 score for each class
        #acc = accuracy_score(truth_i, pred_i)
        acc = weighted_acc(pred_i, truth_i, verbose=False)
        recall = recall_score(truth_i, pred_i, zero_division=0)
        precision = precision_score(truth_i, pred_i, zero_division=0)
        f1 = f1_score(truth_i, pred_i, zero_division=0)


        accs.append(acc)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    # Append average metrics across all classes
    accs.append(np.average(accs))
    recalls.append(np.average(recalls))
    precisions.append(np.average(precisions))
    f1s.append(np.average(f1s))
    
   

    return (accs, recalls, precisions, f1s, aucs), best_thresholds

