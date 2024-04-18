import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, LeavePGroupsOut
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support


from data import *
from models import *

### TRAIN EXPERIMENTS

class TrainParams():
    k = 5
    n_epochs = 100
    batch_size = 128
    early_stopping = None
    base_decay = 0.0005
    lr = 0.001
    m = None
    transfer_l1_map = [True] * 12
    n_categories = 3

def train_kfold_model(dataset, trainparams, test=False):
    history = {'train_loss': [], 'test_loss': [],'train_F':[],'test_F':[]}
    device = dataset.__getitem__(0)[0].device

    confusion_list = []

    splits = KFold(n_splits=trainparams.k, shuffle=False)

    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(dataset)))):

        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=test_sampler)
        
        model = SingleLeadModel(lstm_hidden_size=16, output_size=trainparams.n_categories).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001, lr=trainparams.lr)

        weight_tensor = torch.Tensor(trainparams.weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        train_loss_list, test_loss_list, train_F_list, test_F_list = [], [], [], []

        best_test_loss = float('inf')  # Initialize best test loss to infinity
        patience_counter = 0  # Initialize patience counter

        for epoch in range(trainparams.n_epochs):
            train_loss, train_F=train_epoch(model,train_loader,criterion,optimizer,trainparams.n_categories,device)
            # train_loss, train_F,_=val_epoch(model,train_loader,criterion,trainparams.labelmap,device)
            test_loss, test_F, confusion=val_epoch(model,test_loader,criterion,trainparams.n_categories,device)
            confusion_list.append(confusion)

            train_loss_list.append(train_loss)
            train_F_list.append(train_F)
            test_loss_list.append(test_loss)
            test_F_list.append(test_F)

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
            train_F = train_F * 100
            test_loss = test_loss
            test_F = test_F * 100
            
            if epoch % 10 == 9:
                print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training F1 {:.2f} % AVG Test F1 {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_F,test_F))

        history['train_loss'].append(train_loss_list)
        history['train_F'].append(train_F_list)
        history['test_loss'].append(test_loss_list)
        history['test_F'].append(test_F_list)

        if test:
            break

    return history, confusion_list

def train_entire_model(dataset, trainparams):
    history = {'train_loss': [], 'test_loss': [],'train_F':[],'test_F':[]}
    device = dataset.__getitem__(0)[0].device

    dataloader = DataLoader(dataset, batch_size=trainparams.batch_size)
    
    model = SingleLeadModel(lstm_hidden_size=16).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.001)

    weight_tensor = torch.Tensor(trainparams.weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    for epoch in range(trainparams.n_epochs):
        train_loss, train_F=train_epoch(model,dataloader,criterion,optimizer,trainparams.n_categories,device)
        test_loss, test_F, _=val_epoch(model,dataloader,criterion,trainparams.n_categories,device)

        history['train_loss'].append(train_loss)
        history['train_F'].append(train_F)
        history['test_loss'].append(test_loss)
        history['test_F'].append(test_F)

        train_loss = train_loss
        train_F = train_F * 100
        test_loss = test_loss
        test_F = test_F * 100
        
        if epoch % 10 == 9:
            print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training F1 {:.2f} % AVG Test F1 {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_F,test_F))

    return model, history

def train_kfold_transfer_model(dataset, trainparams, model_class, buffer=None, verbose=True, test=False):
    history = {'train_loss': [], 'test_loss': [], 'train_F': [], 'test_F': []}
    device = dataset.__getitem__(0)[0].device

    # Assume the dataset is divided into k groups evenly
    num_samples = len(dataset)
    k = trainparams.k  # Total number of groups
    m = trainparams.m  # Number of groups to leave out in each split
    groups = np.array([i % k for i in range(num_samples)])

    # Initialize LeavePGroupsOut with n_groups=m
    leave_m_out = LeavePGroupsOut(n_groups=m)

    for fold, (train_idx, val_idx) in enumerate(leave_m_out.split(X=np.arange(num_samples), groups=groups)):
        if verbose:
            print(f'Fold {fold + 1}')

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=trainparams.batch_size, sampler=test_sampler)

        model = model_class(buffer, output_size=trainparams.n_categories).to(device)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        train_loss_list, test_loss_list, train_F_list, test_F_list = [], [], [], []

        best_test_loss = float('inf')  # Initialize best test loss to infinity
        patience_counter = 0  # Initialize patience counter

        # print(val_epoch(model, test_loader, criterion, trainparams.labelmap, device))

        for epoch in range(trainparams.n_epochs):
            base_decay = trainparams.base_decay if buffer else 0
            train_loss, train_F=train_epoch(model,train_loader,criterion,optimizer,trainparams.n_categories,device,base_decay=base_decay)
            test_loss, test_F, _=val_epoch(model,test_loader,criterion,trainparams.n_categories,device)

            train_loss_list.append(train_loss)
            train_F_list.append(train_F)
            test_loss_list.append(test_loss)
            test_F_list.append(test_F)

            # Early Stopping Check
            if test_loss < best_test_loss:
                best_test_loss = test_loss  # Update best test loss
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1  # Increment patience counter

            if trainparams.early_stopping and patience_counter >= trainparams.early_stopping:
                if verbose:
                    print(f"Early stopping triggered after epoch {epoch + 1}")
                break

            train_loss = train_loss
            train_F = train_F * 100
            test_loss = test_loss
            test_F = test_F * 100

            # print(f'bd loss {bd_loss}, misclassification loss {single_loss}')
            
            if epoch % 10 == 9 and verbose:
                print("Epoch:{}/{} AVG Training Loss:{:.5f} AVG Test Loss:{:.5f} AVG Training F1 {:.2f} % AVG Test F1 {:.2f} %".format(epoch + 1, trainparams.n_epochs,train_loss,test_loss,train_F,test_F))

        history['train_loss'].append(train_loss_list)
        history['train_F'].append(train_F_list)
        history['test_loss'].append(test_loss_list)
        history['test_F'].append(test_F_list)

        if test:
            break

    return history



### TRAIN STEP

def val_epoch(model,dataloader,criterion,n_categories,device,confusion=False):
    totalloss = .0

    model.eval()

    all_labels = []
    all_predictions = []

    allp = []
    alll = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 0):
            x, labels = batch

            y = id(labels, device, n=n_categories)

            yhat = model.forward(x)

            loss = criterion(yhat, y)

            _, predicted = torch.max(yhat.data, 1)

            totalloss += loss.item()

            if confusion or True:
                allp += [x.item() for x in predicted]
                alll += [x.item() for x in labels]

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # if confusion:
    #     plot_confusion_matrix(allp, alll, labelmap)

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=1)

    return totalloss / len(dataloader), f1_score, confusion_matrix(alll, allp)

def train_epoch(model,dataloader,criterion,optimizer,n_categories,device,max_norm=1,base_decay=0):
    totalloss = .0

    model.train()

    all_labels = []
    all_predictions = []

    for i,batch in enumerate(dataloader, 0):
        optimizer.zero_grad()

        x, labels = batch

        y = id(labels, device=device, n=n_categories)

        yhat = model.forward(x)

        loss = criterion(yhat, y)
        if base_decay != 0:
            loss += base_decay * model.get_l1_weightdiff()
            # single_loss += criterion(yhat, y).item()
            # bd_loss += (base_decay * model.get_l1_weightdiff()).item()
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        _, predicted = torch.max(yhat.data, 1)

        totalloss += loss.item()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions, average='macro', zero_division=1)

    return totalloss / len(dataloader), f1_score

### UTILITY

def id(labels, device, n):
    return torch.eye(n, device=device)[labels]

def test_forwards(model,dataset,labelmap,device,batch_size=128,max_norm=1):
    device = next(model.parameters()).device

    dataloader = DataLoader(dataset, batch_size=batch_size)

    model.train()
    for i,batch in enumerate(dataloader, 0):
        x, label = batch

        y = id(label, device, n=len(labelmap))

        out = model.forward(x)

        # for z in out:
        #     print(z.shape)

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
