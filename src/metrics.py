import os
import mindspore as ms
import numpy as np
from mindspore import nn, ops
from scipy.special import softmax


# def softmax(x):
#     exp_x = np.exp(x)
#     return exp_x / np.sum(exp_x, axis=0)

class Evaluate(nn.Metric):
    def __init__(self):
        super().__init__()
        self.clear()

    def clear(self):
        self._correct_num = 0
        self._total_num = 0
        # self._loss = 0
        # self._steps = 0

    def update(self, *inputs):
        logits = inputs[0].asnumpy()
        labels = inputs[1].asnumpy()
        # loss = inputs[2].asnumpy()
        if len(labels.shape) == 2:
            labels = labels.reshape(-1)
        logits = softmax(logits)
        preds = np.argmax(logits, axis=1)
        self._correct_num += np.sum(preds == labels)

        self._total_num += labels.shape[0]
        # self._loss += loss
        # self._steps += 1

    def eval(self):
        return self._correct_num / self._total_num  # , self._loss / self._steps


class PretrainEvaluate(nn.Metric):
    def __init__(self, log=None):
        super().__init__()
        self.clear()
        self.logger = log
        self.epoch = 0

    def clear(self):
        self._total_loss = 0
        self._total_num = 0

    def update(self, *inputs):
        loss = inputs[0].asnumpy()
        self._total_loss += loss  # loss在网络中计算后已经做过平均
        self._total_num += 1

    def eval(self):
        self.epoch += 1
        if self.logger:
            self.logger.info(f"epoch {self.epoch} test loss: {str(self._total_loss / self._total_num)}")
        return self._total_loss / self._total_num  # , self._loss / self._steps


class EvalNet(nn.Cell):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss

    def construct(self, imgs, labels, index=None):
        logits, labels = self.net(imgs, labels)
        loss = self.loss(logits, labels)
        return loss, logits, labels


class RebuildEvalNet(nn.Cell):
    def __init__(self, net, loss):
        super().__init__()
        self.net = net
        self.loss = loss

    def construct(self, imgs, index=None):
        loss = self.net(imgs)
        return loss


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = ops.TopK()(output, maxk)
    pred = ops.Transpose()(pred, (1, 0))
    correct = ops.Equal()(pred, target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def final_test(data_loader, model, test_num_spatial_crops, file):
    model.set_train(False)
    final_result = []
    criterion = nn.CrossEntropyLoss()

    for batch in data_loader.create_tuple_iterator():
        videos = batch[0]
        target = batch[1]
        names = batch[2].asnumpy()
        ids = batch[3]

        chunk_nb = ids // test_num_spatial_crops
        split_nb = ids % test_num_spatial_crops

        logits, target = model(videos, target)
        loss = criterion(logits, target)

        for i in range(logits.shape[0]):
            string = "{} {} {} {} {}\n".format(names[i], \
                                               str(logits[i].asnumpy().tolist()), \
                                               str(int(target[i].asnumpy())), \
                                               str(int(chunk_nb[i].asnumpy())), \
                                               str(int(split_nb[i].asnumpy())))
            final_result.append(string)

        # top1 = nn.TopKCategoricalAccuracy(1)
        # top1.clear()
        # top1.update(logits, target)
        # acc1 = top1.eval()

        # top5 = nn.TopKCategoricalAccuracy(5)
        # top5.clear()
        # top5.update(logits, target)
        # acc5 = top5.eval()

        # acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        # batch_size = videos.shape[0]

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        for line in final_result:
            f.write(line)


def merge(eval_path, num_tasks, device_id=0):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    distribute = True if num_tasks > 1 else False
    device_id_src = device_id

    for x in range(num_tasks):
        if distribute:
            if eval_path.startswith('./'):
                eval_path = eval_path.split('./')[1]
            device_id_dst = device_id_src + x
            eval_path = os.path.join(os.getcwd().replace(f'device{device_id_src}', f'device{device_id_dst}'), eval_path)
            print("eval_path:", eval_path)
        eval_path = os.path.join(eval_path, str(x) + '.txt')
        lines = open(eval_path, 'r').readlines()
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    # p = Pool(64)
    p = Pool(num_tasks)
    ans = p.map(compute_video, input_lst)
    # ans = []
    # for lst in input_lst:
    #     ans_ = compute_video(lst)
    #     ans.append(ans_)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100

"""
在notebook上运行的单机版本
"""
def merge_notebook(eval_path, num_tasks, device_id=0):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    distribute = False if num_tasks > 8 else False
    device_id_src = device_id

    oral_eval_path=copy.deepcopy(eval_path)

    same_acc = 0
    for x in range(num_tasks):
        if distribute:
            print(f"distribute:{distribute}")
            if eval_path.startswith('./'):
                eval_path = eval_path.split('./')[1]
            device_id_dst = device_id_src + x
            eval_path = os.path.join(os.getcwd().replace(f'device{device_id_src}', f'device{device_id_dst}'), eval_path)
            print("eval_path:", eval_path)
        eval_path = os.path.join(oral_eval_path, str(x) + '.txt')
        print("eval_path:", eval_path)
        lines = open(eval_path, 'r').readlines()
        print(f"x:{x} length of line:{len(lines)}")


        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                same_acc+=1
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")
    print("same_acc:",same_acc)
    print(f"length of dict_feats{len(dict_feats)}")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    # from multiprocessing import Pool
    # p = Pool(num_tasks)
    # ans = p.map(compute_video, input_lst)
    ans = []
    for lst in input_lst:
        ans_ = compute_video(lst)
        ans.append(ans_)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1, final_top5 = np.mean(top1), np.mean(top5)
    return final_top1 * 100, final_top5 * 100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]
