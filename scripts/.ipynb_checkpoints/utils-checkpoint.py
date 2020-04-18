import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as F

def get_next_run_name(log_dir, continue_training=False):
    if not os.path.isdir(log_dir):
        return os.path.join(log_dir, 'run_0')
    else:
        runs = [int(d[4:]) for d in next(os.walk(log_dir))[1] if d[:3] == 'run']
        
        if continue_training:
            return os.path.join(log_dir, 'run_' + str(max(runs)))
        else:
            return os.path.join(log_dir, 'run_' + str(max(runs) + 1))
    
    
def save_cm(cm, filename='', classes=[]):
    fig = plt.figure(figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(cm, cmap='Oranges')
    
    if len(classes) == 0:
        classes = [str(c) for c in range(len(cm))]
    tick_marks = np.arange(len(classes))
    
    ax.set_xlabel('Predicted', fontsize=7)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=7, rotation=-90,  ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label', fontsize=7)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=7, va ='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i,j]!=0 else '.', horizontalalignment="center", fontsize=6, verticalalignment='center', color="black")
    fig.set_tight_layout(True)
    plt.savefig(os.path.join(filename, 'confusion_matrix.jpg'))
    plt.close()
    

def activated_output_transform(output):
    y_pred, y = output
    y_pred = torch.sigmoid(y_pred)
    return y_pred, y


def plot_stats(stats):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    for i, k in enumerate(stats):
        if k == 'conf_matrix':
            continue
        
        axes[i // 3, i % 3].plot(range(len(stats[k])), stats[k], label=k)
        axes[i // 3, i % 3].legend()
        
    plt.tight_layout()
    plt.show()
    
    print(stats['conf_matrix'])
    

class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m/s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1/s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = F.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)
    

def show_batch(dataloader):
    fig, ax = plt.subplots(1, dataloader.batch_size, sharex=True, sharey=True, figsize=(20, 10))
    img_batch, labels = iter(dataloader).next()
    
    denormalize = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))  
    
    for i in range(dataloader.batch_size):                                
        ax[i].imshow(transforms.ToPILImage()(denormalize(img_batch[i, :, :, :])))
        
    print(labels)
