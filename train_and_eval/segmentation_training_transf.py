import sys
import os
sys.path.insert(0, os.getcwd())
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils.lr_scheduler import build_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from models import get_model
from utils.config_files_utils import read_yaml, copy_yaml, get_params_values
from utils.torch_utils import get_device, get_net_trainable_params, load_from_checkpoint
from data import get_dataloaders
from metrics.torch_metrics import get_mean_metrics
from metrics.numpy_metrics import get_classification_metrics, get_per_class_loss
from metrics.loss_functions import get_loss
from utils.summaries import write_mean_summaries, write_class_summaries
from data import get_loss_data_input

# 启用异常检测
torch.autograd.set_detect_anomaly(True)


def train_and_evaluate(net, dataloaders, config, device, lin_cls=False):

    def train_step(net, sample, loss_fn, optimizer, device, loss_input_fn):
        optimizer.zero_grad()
        # print(sample['inputs'].shape)
        outputs, cls_embedding = net(sample['inputs'].to(device))
        outputs = outputs.permute(0, 2, 3, 1)
        ground_truth = loss_input_fn(sample, device)
        loss = loss_fn['mean'](outputs, ground_truth)
        # 这里添加一个监督损失函数
        '''Prior = [0.04038229070454944, 0.06657914164418673, 0.07865396226101676, 
                 0.06657914164418673, 0.04038229070454944, 0.017550071195609224, 
                 0.005465148750243429, 0.0012194395158768117, 0.00019496368015844016, 
                 2.23348138654879e-05, 1.8333561309276872e-06, 1.0796859408083036e-07, 
                 9.08885628000076e-09, 1.0796859408083036e-07, 1.8333561309276872e-06, 
                 2.23348138654879e-05, 0.00019496368015844019, 0.0012194395158768158, 
                 0.005465148750243941, 0.017550071195655256, 0.04038229070751874, 
                 0.06657914178141656, 0.07865396680544491, 0.06657924947555097, 
                 0.04038412405771107, 0.017572406009428675, 0.005660112430401359, 
                 0.002438879031753615, 0.005660112430401359, 0.017572406009428675, 
                 0.04038412405771107, 0.06657924947555097, 0.07865396680544491, 
                 0.06657914178141656, 0.04038229070751874, 0.017550071195655256, 
                 0.005465148750243941, 0.0012194395158768117, 0.00019496368015792877,
                 2.233481381945242e-05, 1.8333531616347876e-06, 1.078313642418516e-07, 
                 4.54442814000038e-09, 1.3722983897876466e-10, 2.969292899759878e-12, 
                 4.603548241045495e-14, 5.114080147340586e-16, 4.0707790810615364e-18, 
                 2.3217887948898078e-20, 9.488620133101572e-23, 2.7785523440006987e-25, 
                 5.830009703784104e-28, 8.765065084108651e-31, 9.442265015400106e-34, 
                 7.288403342248169e-37, 4.031101971930694e-40, 1.5975348564336958e-43, 
                 4.536407647223706e-47, 9.230154651480211e-51, 1.345677695966951e-54]
        '''
        
        Prior = [2.9514465812e-08, 1.651910581e-07, 8.2733843461e-07, 
                 3.7078736199e-06, 1.4870025167e-05, 5.3363406029e-05, 
                 0.00017136433386, 0.00049242760403, 0.0012662206763, 
                 0.0029135432486, 0.0059989963122, 0.011053015482, 
                 0.018223341731, 0.026885636205, 0.035494223031, 
                 0.041931469974, 0.044326920288, 0.041931469974, 
                 0.035494223031, 0.026885636205, 0.018223341731, 
                 0.011053015482, 0.0059989963122, 0.0029135432486, 
                 0.0012662206763, 0.00049242760403, 0.00017136433386, 
                 5.3363406036e-05, 1.487002526e-05, 3.7078746348e-06, 
                 8.2734833561e-07, 1.652774853e-07, 3.0189563947e-08, 
                 9.4375311609e-09, 3.0189571553e-08, 1.6527757841e-07, 
                 8.2734935058e-07, 3.7078845358e-06, 1.4870111687e-05, 
                 5.3364081134e-05, 0.00017136905263, 0.0004924571185, 
                 0.0012663858673, 0.0029143705871, 0.0060027041858, 
                 0.011067885507, 0.018276705137, 0.027057000539, 
                 0.035986650635, 0.04319769065, 0.047240463537, 
                 0.047930466286, 0.046547238513, 0.045108977936, 
                 0.045108977936, 0.046547238513, 0.047930466286, 
                 0.047240463537, 0.04319769065, 0.035986650635, 
                 0.027057000539, 0.018276705137, 0.011067885507, 
                 0.0060027041858, 0.0029143705871, 0.0012663858673, 
                 0.0004924571185, 0.00017136905263, 5.3364081127e-05, 
                 1.4870111594e-05, 3.7078835209e-06, 8.2733944958e-07, 
                 1.651911512e-07, 2.9514473455e-08, 4.7187658611e-09, 
                 6.7509813503e-10, 8.6427203401e-11, 9.9010003768e-12, 
                 1.0149689452e-12, 9.3104675518e-14]
     
        
        Prior = torch.tensor(Prior).unsqueeze(0).expand(int(cls_embedding.shape[0]/(16*16))*16*16, -1).to(device)
        # Prior.to(device)
        # cls_embedding = torch.tensor(cls_embedding)
        # cls_embedding.to(device)

        # 注意，上一行代码里，针对 PASTIS24_fold1 数据集，训练集的 batch_size 是 16


        loss_mse = nn.MSELoss()(Prior, cls_embedding)

        loss_all = loss + loss_mse

        loss_all.backward()
        optimizer.step()
        return outputs, ground_truth, loss_all
  
    def evaluate(net, evalloader, loss_fn, config):
        num_classes = config['MODEL']['num_classes']
        predicted_all = []
        labels_all = []
        losses_all = []
        net.eval()
        with torch.no_grad():
            for step, sample in enumerate(evalloader):
                logits, _ = net(sample['inputs'].to(device))
                logits = logits.permute(0, 2, 3, 1)
                _, predicted = torch.max(logits.data, -1)
                ground_truth = loss_input_fn(sample, device)
                loss = loss_fn['all'](logits, ground_truth)
                target, mask = ground_truth
                if mask is not None:
                    predicted_all.append(predicted.view(-1)[mask.view(-1)].cpu().numpy())
                    labels_all.append(target.view(-1)[mask.view(-1)].cpu().numpy())
                else:
                    predicted_all.append(predicted.view(-1).cpu().numpy())
                    labels_all.append(target.view(-1).cpu().numpy())
                losses_all.append(loss.view(-1).cpu().detach().numpy())
                
                # if step > 5:
                #    break

        print("finished iterating over dataset after step %d" % step)
        print("calculating metrics...")
        predicted_classes = np.concatenate(predicted_all)
        target_classes = np.concatenate(labels_all)
        losses = np.concatenate(losses_all)

        eval_metrics = get_classification_metrics(predicted=predicted_classes, labels=target_classes,
                                                  n_classes=num_classes, unk_masks=None)

        micro_acc, micro_precision, micro_recall, micro_F1, micro_IOU = eval_metrics['micro']
        macro_acc, macro_precision, macro_recall, macro_F1, macro_IOU = eval_metrics['macro']
        class_acc, class_precision, class_recall, class_F1, class_IOU = eval_metrics['class']

        un_labels, class_loss = get_per_class_loss(losses, target_classes, unk_masks=None)

        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")
        print("Mean (micro) Evaluation metrics (micro/macro), loss: %.7f, iou: %.4f/%.4f, accuracy: %.4f/%.4f, "
              "precision: %.4f/%.4f, recall: %.4f/%.4f, F1: %.4f/%.4f, unique pred labels: %s" %
              (losses.mean(), micro_IOU, macro_IOU, micro_acc, macro_acc, micro_precision, macro_precision,
               micro_recall, macro_recall, micro_F1, macro_F1, np.unique(predicted_classes)))
        print(
            "-----------------------------------------------------------------------------------------------------------------------------------------------------------------")

        return (un_labels,
                {"macro": {"Loss": losses.mean(), "Accuracy": macro_acc, "Precision": macro_precision,
                           "Recall": macro_recall, "F1": macro_F1, "IOU": macro_IOU},
                 "micro": {"Loss": losses.mean(), "Accuracy": micro_acc, "Precision": micro_precision,
                           "Recall": micro_recall, "F1": micro_F1, "IOU": micro_IOU},
                 "class": {"Loss": class_loss, "Accuracy": class_acc, "Precision": class_precision,
                           "Recall": class_recall,
                           "F1": class_F1, "IOU": class_IOU}}
                )

    #------------------------------------------------------------------------------------------------------------------#
    num_classes = config['MODEL']['num_classes']
    num_epochs = config['SOLVER']['num_epochs']
    lr = float(config['SOLVER']['lr_base'])
    train_metrics_steps = config['CHECKPOINT']['train_metrics_steps']
    eval_steps = config['CHECKPOINT']['eval_steps']
    save_steps = config['CHECKPOINT']["save_steps"]
    save_path = config['CHECKPOINT']["save_path"]
    # checkpoint = config['CHECKPOINT']["load_from_checkpoint"]
    checkpoint = None
    num_steps_train = len(dataloaders['train'])
    local_device_ids = config['local_device_ids']
    weight_decay = get_params_values(config['SOLVER'], "weight_decay", 0)

    start_global = 1
    start_epoch = 1
    if checkpoint:
        load_from_checkpoint(net, checkpoint, partial_restore=False, modified_model=True, device='cpu')
    # 添加一个参数 modified_model，用来控制是否预处理预训练模型参数，使得能够应用在当前网络上。
    


    print("current learn rate: ", lr)

    if len(local_device_ids) > 1:
        net = nn.DataParallel(net, device_ids=local_device_ids)
    net.to(device)

    if save_path and (not os.path.exists(save_path)):
        os.makedirs(save_path)

    copy_yaml(config)

    loss_input_fn = get_loss_data_input(config)
    
    loss_fn = {'all': get_loss(config, device, reduction=None),
               'mean': get_loss(config, device, reduction="mean")}

    trainable_params = get_net_trainable_params(net)
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    optimizer.zero_grad()

    scheduler = build_scheduler(config, optimizer, num_steps_train)

    writer = SummaryWriter(save_path)

    BEST_IOU = 0
    net.train()
    for epoch in range(start_epoch, start_epoch + num_epochs):  # loop over the dataset multiple times
        for step, sample in enumerate(dataloaders['train']):
            abs_step = start_global + (epoch - start_epoch) * num_steps_train + step
            logits, ground_truth, loss = train_step(net, sample, loss_fn, optimizer, device, loss_input_fn=loss_input_fn)
            if len(ground_truth) == 2:
                labels, unk_masks = ground_truth
            else:
                labels = ground_truth
                unk_masks = None
            # print batch statistics ----------------------------------------------------------------------------------#
            if abs_step % train_metrics_steps == 0:
                logits = logits.permute(0, 3, 1, 2)
                batch_metrics = get_mean_metrics(
                    logits=logits, labels=labels, unk_masks=unk_masks, n_classes=num_classes, loss=loss, epoch=epoch,
                    step=step)
                
                write_mean_summaries(writer, batch_metrics, abs_step, mode="train", optimizer=optimizer)
                
                print("abs_step: %d, epoch: %d, step: %5d, loss: %.7f, batch_iou: %.4f, batch accuracy: %.4f, batch precision: %.4f, "
                      "batch recall: %.4f, batch F1: %.4f" %
                      (abs_step, epoch, step + 1, loss, batch_metrics['IOU'], batch_metrics['Accuracy'], batch_metrics['Precision'],
                       batch_metrics['Recall'], batch_metrics['F1']))

            if abs_step % save_steps == 0:
                if len(local_device_ids) > 1:
                    torch.save(net.module.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))
                else:
                    torch.save(net.state_dict(), "%s/%depoch_%dstep.pth" % (save_path, epoch, abs_step))

            # evaluate model ------------------------------------------------------------------------------------------#
            if abs_step % eval_steps == 0:
                eval_metrics = evaluate(net, dataloaders['eval'], loss_fn, config)
                if eval_metrics[1]['macro']['IOU'] > BEST_IOU:
                    if len(local_device_ids) > 1:
                        torch.save(net.module.state_dict(), "%s/best.pth" % (save_path))
                    else:
                        torch.save(net.state_dict(), "%s/best.pth" % (save_path))
                    BEST_IOU = eval_metrics[1]['macro']['IOU']


                write_mean_summaries(writer, eval_metrics[1]['micro'], abs_step, mode="eval_micro", optimizer=None)
                write_mean_summaries(writer, eval_metrics[1]['macro'], abs_step, mode="eval_macro", optimizer=None)
                write_class_summaries(writer, [eval_metrics[0], eval_metrics[1]['class']], abs_step, mode="eval",
                                      optimizer=None)
                net.train()

        scheduler.step_update(abs_step)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--config', help='configuration (.yaml) file to use')
    parser.add_argument('--device', default='0,1', type=str,
                         help='gpu ids to use')
    parser.add_argument('--lin', action='store_true',
                         help='train linear classifier only')

    args = parser.parse_args()
    config_file = args.config
    print(args.device)
    device_ids = [int(d) for d in args.device.split(',')]
    lin_cls = args.lin

    device = get_device(device_ids, allow_cpu=False)

    config = read_yaml(config_file)
    config['local_device_ids'] = device_ids

    dataloaders = get_dataloaders(config)

    net = get_model(config, device)

    train_and_evaluate(net, dataloaders, config, device)
