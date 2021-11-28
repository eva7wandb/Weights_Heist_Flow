from tqdm import tqdm
import torch.nn.functional as F

def train(
    model, device, 
    train_loader, 
    optimizer, criterion,
    epoch,
    lr_scheduler=None,
):
    model.train()
    pbar = tqdm(train_loader)
    
    train_batch_loss = []
    train_batch_acc = []
    
    correct = 0
    processed = 0
    train_loss = 0
    
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        
        loss = criterion(y_pred, target)
        train_loss += loss.item()
        train_batch_loss.append(loss.item())
        
        loss.backward()
        optimizer.step()
        if lr_scheduler != None:
          lr_scheduler.step()
        
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"TRAIN Epoch:{epoch} Loss:{round(loss.item(), 4)} Batch:{batch_idx} Acc:{100*correct/processed:0.2f}"
        )
        train_batch_acc.append(100*correct/processed)
    
    return train_batch_loss, train_batch_acc