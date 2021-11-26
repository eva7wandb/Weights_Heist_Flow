import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.setup import get_device
from collections import defaultdict

def test(
    model, device, 
    test_loader, 
    criterion,
    epoch,
    lr_scheduler=None
):
    model.eval()
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            test_loss += criterion(y_pred, target)
            pred = y_pred.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss = test_loss.item()
    test_acc = 100. * correct / len(test_loader.dataset)
    
    print(
        f'TEST \
        Loss:{test_loss:.4f} \
        Acc:{test_acc:.2f} \
        [{correct} / {len(test_loader.dataset)}]'
    )
    
    return test_loss, test_acc


def get_sample_predictions(trainer, correct_samples=20, mistake_samples=20):
    device = get_device()
    selected_preds = defaultdict(lambda : defaultdict(list))
    with torch.no_grad():
        for (data, target), (data_n, _) in tqdm(
            zip(trainer.test_loader, trainer.test_loader_unnormalized),
            desc='Generating sample predictions'
        ):
            data, target = data.to(device), target.to(device)
            y_pred = trainer.net(data)
            pred = y_pred.argmax(dim=1, keepdim=True)
            correctness = pred.eq(target.view_as(pred))

            for n, correct in enumerate(correctness):
                actual_class = target[n].item()
                pred_class = pred[n].item()
                scores = y_pred[n].cpu().numpy()

                temp_content = {
                    'pred_class': pred_class,
                    'scores': scores,
                    'data': data[n].cpu(),
                    'data_unnormalized': data_n[n].cpu(),
                    'actual_class': actual_class,
                    'pred_class': pred_class,     
                }

                if correct[0].item():
                    if len(selected_preds['correct'][actual_class]) >= correct_samples:
                        continue
                    selected_preds['correct'][actual_class].append(temp_content)
                else:
                    if len(selected_preds['mistakes'][actual_class]) >= mistake_samples:
                        continue
                    selected_preds['mistakes'][actual_class].append(temp_content)
    return selected_preds