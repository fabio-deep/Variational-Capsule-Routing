import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, args, dataloader):
    model.eval()
    sample_count = 0
    running_loss = 0
    running_acc = 0

    with torch.no_grad():

        for i, (inputs, labels) in enumerate(dataloader):

            labels = labels.type(torch.LongTensor)
            onehot_labels = torch.zeros(labels.size(0),
                args.n_classes).scatter_(1, labels.view(-1, 1), 1).cuda()
            inputs = inputs.type(torch.FloatTensor).cuda()

            yhat = model(inputs)

            loss = F.cross_entropy(yhat, labels.cuda())

            sample_count += inputs.size(0)
            running_loss += loss.item() * inputs.size(0) # smaller batches count less
            running_acc += (yhat.argmax(-1) == labels.data.cuda()).sum().item() # n_corrects

        loss = running_loss / sample_count
        acc = running_acc / sample_count

    return loss, acc
