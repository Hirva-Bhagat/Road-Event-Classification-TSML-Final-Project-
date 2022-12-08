import csv
import torch
from torch import tensor
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value

def top_k(logits, y, k : int = 1):
    """
    logits : (bs, n_labels)
    y : (bs,)
    """
    labels_dim = 1
    assert 1 <= k <= logits.size(labels_dim)
    #k_labels = torch.topk(input = logits, k = k, dim=labels_dim, largest=True, sorted=True)[1]

    # True (#0) if `expected label` in k_labels, False (0) if not
    #a = ~torch.prod(input = torch.abs(y.unsqueeze(labels_dim) - k_labels), dim=labels_dim).to(torch.bool)
    #print(a)
    # These two approaches are equivalent
#     if True :
#         y_pred = torch.empty_like(y)
#         for i in range(y.size(0)):
#             if a[i] :
#                 y_pred[i] = y[i]
#             else :
#                 y_pred[i] = k_labels[i][0]
        #correct = a.to(torch.int8).numpy()
   #else :
       #a = a.to(torch.int8)
       #y_pred = a * y + (1-a) * k_labels[:,0]
        #correct = a.numpy()

    print(logits)
    print(y)
    f1 = f1_score(logits, y, average='weighted')*100
    #acc = sum(correct)/len(correct)*100
    acc = accuracy_score(y_pred, y)*100

    iou = jaccard_score(y, y_pred, average="weighted")*100

    return acc, f1, iou, y_pred
def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets)
    n_correct_elems = correct.float().sum().item()

    return n_correct_elems / batch_size

def my_collate(batch):
    (data,label) = zip(*batch)
    timesteps = [inputs.size(1)//16 for inputs in data]# get the time steps for item in batch
    data_chunk = []
    for i in range(len(timesteps)):
        data_chunk.extend(torch.chunk(data[i], timesteps[i], dim=1))
    data_chunk = torch.stack(data_chunk)
    target = torch.LongTensor(label)
    return data_chunk, target, timesteps
