import os

import numpy as np
import pandas as pd

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import torchvision

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage, ConfusionMatrix, Precision, Recall
from ignite.contrib.metrics import ROC_AUC
from ignite.handlers import ModelCheckpoint, EarlyStopping, Timer

from .utils import *


def train(params, dataloaders_gen, continue_training=False, stats=None, scheduler=None):
    train_loader, val_loader, holdout_loader, dataset_name, class_count = dataloaders_gen(params['train_batch_size'], params['val_batch_size'])
    model_dir = os.path.join(params['log_dir'], dataset_name)
    run_dir = get_next_run_name(model_dir, continue_training)
    
    if not stats:
        stats = {
            'batch_loss': [],
            'train_loss': [], 
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'precision': [],
            'recall': [],
            'lr': [],
            'conf_matrix': None
        }
    
    model = params['model']
    
    if params['tensorboard']:
        writer = create_summary_writer(model, train_loader, run_dir)
    device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
    print('Device: ', device)

    optimizer = params['optimizer']
    
    if not continue_training:
        optimizer = optimizer(model.parameters(), lr=params['lr'])
    
    criterion = params['criterion']
    
    if not scheduler:
        scheduler = ReduceLROnPlateau(optimizer, patience=params['lr_patience'], factor=params['lr_factor'], min_lr=params['min_lr'])
    
    model_name = (
        params['model_name'] + 
        '_lr:' + str(params['lr']) +
        '_' + str(optimizer).split(' ')[0] +
        '_' + str(criterion).split('.')[-1].split("'")[0][:-2]
    )
    
    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    train_timer = Timer(average=True)
    train_timer.attach(trainer, start=Events.EPOCH_STARTED, 
                       resume=Events.EPOCH_STARTED, 
                       pause=Events.EPOCH_COMPLETED, 
                       step=Events.EPOCH_STARTED)
    
    train_evaluator = create_supervised_evaluator(
        model, 
        metrics=
        {
            "accuracy": Accuracy(), 
            "nll": Loss(F.nll_loss),
        },
        device=device
    )
    
    val_evaluator = create_supervised_evaluator(
        model, 
        metrics=
        {
            "accuracy": Accuracy(), 
            "nll": Loss(F.nll_loss),
            "precision": Precision(),
            "recall": Recall(),
            "conf_matrix": ConfusionMatrix(class_count),
        },
        device=device
    )
    
    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss
    
    if params['early_stopping']:
        handler = EarlyStopping(patience=params['early_stopping_patience'], score_function=score_function, trainer=trainer)
        val_evaluator.add_event_handler(Events.COMPLETED, handler)
    
    checkpointer = ModelCheckpoint(os.path.join(model_dir,'saved_models'), dataset_name, n_saved=3, create_dir=True, save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {model_name: model})
    
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    @trainer.on(Events.ITERATION_COMPLETED(every=params['log_interval']))
    def log_training_loss(engine):   
        stats['batch_loss'].append(engine.state.output)
        
        if params['tensorboard']:
            writer.add_scalar("training/batch_loss", engine.state.output, engine.state.iteration)
        
        if params['verbose']:
            print(
                "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
                "".format(engine.state.epoch, engine.state.iteration, len(train_loader), engine.state.output)
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        avg_accuracy = metrics["accuracy"]
        avg_nll = metrics["nll"]
        
        stats['train_accuracy'].append(avg_accuracy)
        stats['train_loss'].append(avg_nll)
        
        if params['tensorboard']:
            writer.add_scalar("training/epoch_loss", avg_nll, engine.state.epoch)
            writer.add_scalar("training/epoch_accuracy", avg_accuracy, engine.state.epoch)
        
        if params['verbose']:
            print(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    engine.state.epoch, avg_accuracy, avg_nll
                )
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        val_evaluator.run(val_loader)        
        metrics = val_evaluator.state.metrics
        accuracy = metrics["accuracy"]       
        precision = metrics["precision"]
        recall = metrics["recall"]
        
        nll = metrics["nll"] 
        if scheduler:
            scheduler.step(nll)
              
        stats['val_loss'].append(nll)
        stats['val_accuracy'].append(accuracy)
        stats['precision'].append((precision.sum() / len(precision)).item())
        stats['recall'].append((recall.sum() / len(recall)).item())
        stats['lr'].append(get_lr(optimizer))

        if params['tensorboard']:
            writer.add_scalar("valdation/epoch_loss", nll, engine.state.epoch)
            writer.add_scalar("valdation/epoch_accuracy", accuracy, engine.state.epoch)            
            writer.add_scalar("valdation/epoch_precision", precision.sum() / len(precision), engine.state.epoch)
            writer.add_scalar("valdation/epoch_recall", recall.sum() / len(recall), engine.state.epoch)            
            writer.add_scalar("valdation/learning_rate", get_lr(optimizer), engine.state.epoch)
            save_cm(metrics["conf_matrix"] , filename=run_dir, classes=[])
        
        if params['verbose']:
            print(
                "Validation Results - Epoch: {}  Accuracy: {:.2f}  loss: {:.2f}".format(
                    engine.state.epoch, accuracy, nll
                )
            )
    
    @trainer.on(Events.COMPLETED)
    def log_holdout(engine):
        val_evaluator.run(holdout_loader)
        metrics = val_evaluator.state.metrics
        accuracy = metrics["accuracy"]       
        precision = metrics["precision"]
        recall = metrics["recall"]
        stats['conf_matrix'] = metrics["conf_matrix"]
        
        print('Final accuracy: ', accuracy)
        print('Confusion matrix: ', metrics["conf_matrix"])
        print('Precision: ', precision)
        print('Recall: ', recall)
        
        total_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        runs_df = pd.DataFrame({
            'model_name': [params['model_name']],
            'dataset_name': [dataset_name],
            'optimizer': [str(optimizer).split(' ')[0]],
            'criterion': [str(criterion).split('.')[-1].split("'")[0][:-2]],
            'init_lr': [params['lr']],
            'lr_patience': [params['lr_patience']],
            'min_lr': [params['min_lr']],
            'accuracy': [accuracy],
            'precision': [(sum(precision) / len(precision)).item()],
            'recall': [(sum(recall) / len(recall)).item()],
            'train_time(epoch)': [train_timer.value()],
            'train_images': [len(train_loader) * params['train_batch_size']],
            'total_model_params': [total_model_params],
            'holdout_images': [len(holdout_loader) * params['val_batch_size']],
            'validation_images': [len(val_loader) * params['val_batch_size']],
        })
        
        if os.path.exists('runs.csv'):
            old_runs_df = pd.read_csv('runs.csv', index_col=0)
            runs_df = pd.concat([old_runs_df, runs_df], ignore_index=True)
            
        runs_df.to_csv('runs.csv')
        
    trainer.run(train_loader, max_epochs=params['epochs'])
    
    if params['tensorboard']:
        writer.close()    
        
    return model, optimizer, scheduler, stats


def continue_training(model, params, dataloaders_gen, stats, scheduler, optimizer, new_params=None):
    params['model'] = model
    params['optimizer'] = optimizer
    
    if new_params:
        for i in new_params:
            params[i] = new_params[i]
    
    train(params, dataloaders_gen, continue_training=True, stats=stats, scheduler=scheduler)