import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeavePGroupsOut
import matplotlib.pyplot as plt


from data import *
from models import *

### TRAIN EXPERIMENTS

class TrainParams():
    k = 5
    n_epochs = 100
    batch_size = 128
    early_stopping = None

def train_kfold_model(dataset, trainparams):
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    device = dataset.__getitem__(0)[0].device

    splits = KFold(n_splits=trainparams.k, shuffle=False)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=test_sampler)
        
        model = SingleLeadModel(lstm_hidden_size=16).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)

        weight_tensor = torch.Tensor(trainparams.weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []

        best_test_loss = float('inf')  # Initialize best test loss to infinity
        patience_counter = 0  # Initialize patience counter

        for epoch in range(trainparams.n_epochs):
            train_loss, train_acc=train_epoch(model,train_loader,criterion,optimizer,trainparams.labelmap,device)
            test_loss, test_acc=val_epoch(model,test_loader,criterion,trainparams.labelmap,device)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            # Early Stopping Check
            if test_loss < best_test_loss:
                best_test_loss = test_loss  # Update best test loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            if trainparams.early_stopping and patience_counter >= trainparams.early_stopping:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

            train_loss = train_loss
            train_acc = train_acc * 100
            test_loss = test_loss
            test_acc = test_acc * 100
            
            if epoch % 10 == 9:
                print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_acc,test_acc))

        history['train_loss'].append(train_loss_list)
        history['train_acc'].append(train_acc_list)
        history['test_loss'].append(test_loss_list)
        history['test_acc'].append(test_acc_list)

        # break

    return history

def train_entire_model(dataset, trainparams):
    history = {'train_loss': [], 'test_loss': [],'train_acc':[],'test_acc':[]}
    device = dataset.__getitem__(0)[0].device

    dataloader = DataLoader(dataset, batch_size=trainparams.batch_size)
    
    model = SingleLeadModel(lstm_hidden_size=16).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.001)

    weight_tensor = torch.Tensor(trainparams.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(trainparams.n_epochs):
        train_loss, train_acc=train_epoch(model,dataloader,criterion,optimizer,trainparams.labelmap,device)
        test_loss, test_acc=val_epoch(model,dataloader,criterion,trainparams.labelmap,device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        train_loss = train_loss
        train_acc = train_acc * 100
        test_loss = test_loss
        test_acc = test_acc * 100
        
        if epoch % 10 == 9:
            print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_acc,test_acc))

    return model, history

def train_kfold_transfer_model(dataset, buffer, trainparams, verbose=True, test=False):
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    device = dataset.__getitem__(0)[0].device

    # Assume the dataset is divided into k groups evenly
    num_samples = len(dataset)
    k = trainparams.k  # Total number of groups
    m = trainparams.m  # Number of groups to leave out in each split
    groups = np.array([i % k for i in range(num_samples)])

    # Initialize LeavePGroupsOut with n_groups=m
    leave_m_out = LeavePGroupsOut(n_groups=m)

    for fold, (train_idx, val_idx) in enumerate(leave_m_out.split(X=np.arange(num_samples), groups=groups)):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=test_sampler)
        
        transfer_def = TransferDef()
        transfer_def.return_request = [4]

        model = TransferModel(buffer, transfer_def=transfer_def).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)

        criterion = nn.CrossEntropyLoss()

        train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []

        best_test_loss = float('inf')  # Initialize best test loss to infinity
        patience_counter = 0  # Initialize patience counter

        for epoch in range(trainparams.n_epochs):
            train_loss, train_acc=train_epoch(model,train_loader,criterion,optimizer,trainparams.labelmap,device,base_decay=0.0005)
            test_loss, test_acc=val_epoch(model,test_loader,criterion,trainparams.labelmap,device)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            # Early Stopping Check
            if test_loss < best_test_loss:
                best_test_loss = test_loss  # Update best test loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            if trainparams.early_stopping and patience_counter >= trainparams.early_stopping:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

            train_loss = train_loss
            train_acc = train_acc * 100
            test_loss = test_loss
            test_acc = test_acc * 100

            # print(f'bd loss {bd_loss}, misclassification loss {single_loss}')
            
            if epoch % 10 == 9 and verbose:
                print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_acc,test_acc))

        history['train_loss'].append(train_loss_list)
        history['train_acc'].append(train_acc_list)
        history['test_loss'].append(test_loss_list)
        history['test_acc'].append(test_acc_list)

        if test:
            break

    return history

def train_control_12lead_model(dataset, trainparams, verbose=True, test=False):
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    device = dataset.__getitem__(0)[0].device

    # Assume the dataset is divided into k groups evenly
    num_samples = len(dataset)
    k = trainparams.k  # Total number of groups
    m = trainparams.m  # Number of groups to leave out in each split
    groups = np.array([i % k for i in range(num_samples)])

    # Initialize LeavePGroupsOut with n_groups=m
    leave_m_out = LeavePGroupsOut(n_groups=m)

    for fold, (train_idx, val_idx) in enumerate(leave_m_out.split(X=np.arange(num_samples), groups=groups)):
        print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=test_sampler)

        transfer_def = TransferDef()
        transfer_def.return_request = [4]

        model = TransferModel(transfer_def=transfer_def).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loss_list, test_loss_list, train_acc_list, test_acc_list = [], [], [], []

        best_test_loss = float('inf')  # Initialize best test loss to infinity
        patience_counter = 0  # Initialize patience counter

        for epoch in range(trainparams.n_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, trainparams.labelmap, device)
            test_loss, test_acc = val_epoch(model, test_loader, criterion, trainparams.labelmap, device)

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            # Early Stopping Check
            if test_loss < best_test_loss:
                best_test_loss = test_loss  # Update best test loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            if trainparams.early_stopping and patience_counter >= trainparams.early_stopping:
                print(f"Early stopping triggered after epoch {epoch + 1}")
                break

            # Verbose Logging
            if verbose and epoch % 10 == 9:
                print(f"Epoch:{epoch + 1}/{trainparams.n_epochs} AVG Training Loss:{train_loss:.5f} AVG Test Loss:{test_loss:.5f} AVG Training Acc {train_acc * 100:.2f} % AVG Test Acc {test_acc * 100:.2f} %")

        history['train_loss'].append(train_loss_list)
        history['train_acc'].append(train_acc_list)
        history['test_loss'].append(test_loss_list)
        history['test_acc'].append(test_acc_list)

        if test:
            break

    return history



### TRAIN STEP

def val_epoch(model,dataloader,criterion,labelmap,device,confusion=False):
    totalloss = .0
    correct = 0
    total = 0

    model.eval()

    allp = []
    alll = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            x, label = batch

            y = id(label, device, n=len(labelmap))

            yhat = model.forward(x)[0]

            loss = criterion(yhat, y)

            _, predicted = torch.max(yhat.data, 1)

            total += yhat.shape[0]
            totalloss += loss.item()
            correct += (predicted == label).sum().item()

            if confusion:
                allp += [x.item() for x in predicted]
                alll += [x.item() for x in label]

    if confusion:
        plot_confusion_matrix(allp, alll, labelmap)

    return totalloss / total, correct / total

def train_epoch(model,dataloader,criterion,optimizer,labelmap,device,max_norm=1,base_decay=0):
    totalloss = .0
    correct = 0
    total = 0
    single_loss = 0
    bd_loss = 0



    model.train()
    for i,batch in enumerate(dataloader, 0):
        optimizer.zero_grad()

        x, label = batch

        y = id(label, device=device, n=len(labelmap))

        yhat = model.forward(x)[0]

        loss = criterion(yhat, y)
        if base_decay != 0:
            loss += base_decay * model.get_l1_weightdiff()
            # single_loss += criterion(yhat, y).item()
            # bd_loss += (base_decay * model.get_l1_weightdiff()).item()
            
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)

        optimizer.step()

        _, predicted = torch.max(yhat.data, 1)

        total += yhat.shape[0]
        totalloss += loss.item()
        correct += (predicted == label).sum().item()

    return totalloss / total, correct / total#, bd_loss / total, single_loss / total

### UTILITY

def id(labels, device, n):
    return torch.eye(n, device=device)[labels]

def test_forwards(model,data,labelmap,device,batch_size=128,max_norm=1):
    device = next(model.parameters()).device

    dataset = WindowDataset(data, labelmap, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    for i,batch in enumerate(dataloader, 0):
        x, label = batch

        y = id(label, device, n=len(labelmap))

        print(x.shape)

        out = model.forward(x)

        for z in out:
            print(z.shape)

        break

def plot_confusion_matrix(predictions, actuals, labelmap):
    """
    Plots a confusion matrix heatmap given the predictions and actual values.

    :param predictions: List of predicted categories.
    :param actuals: List of actual categories.
    """
    # Calculating the confusion matrix
    cm = confusion_matrix(actuals, predictions)

    labelmap = {x:y for y,x in labelmap.items()}

    # Converting the confusion matrix to fraction
    cm_fraction = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Creating a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_fraction, annot=True, fmt=".2f", cmap='Blues', xticklabels=[labelmap[i] for i in range(4)], yticklabels=[labelmap[i] for i in range(4)])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (as fraction)')
    plt.show()
