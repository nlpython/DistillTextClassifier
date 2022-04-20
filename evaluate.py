from torch import nn
import torch
from tqdm import tqdm

def evaluate(model, valid, loss_fn, device):

    model.eval()

    valid_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(valid, desc="Validating"):

            input, label, text = batch
            # to device
            for key in input.keys():
                if key not in ['label']:
                    input[key] = input[key].to(device)
            label = label.to(device)

            logits = model(**input)

            valid_loss += loss_fn(logits, label).item()

            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += label.shape[0]


    valid_acc = correct / total

    return valid_loss, valid_acc
