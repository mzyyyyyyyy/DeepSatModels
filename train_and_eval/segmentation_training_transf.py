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
        Prior = np.random.rand(12, 80).astype(np.float32)
        Prior[1] = [9.413229200553205e-14, 1.0261723098002073e-12, 1.0010304797547711e-11, 
                   8.73815775607398e-11, 6.825561933535113e-10, 4.770935626318172e-09, 
                   2.984121300506083e-08, 1.6702411340276699e-07, 8.365563051497594e-07, 
                   3.7494761834611175e-06, 1.5038901292461017e-05, 5.3982163639709224e-05, 
                   0.0001734225211420432, 0.0004986984936138465, 0.0012839434545941603, 
                   0.00296073118096772, 0.006119153432498831, 0.011348251835047249, 
                   0.018922316170672887, 0.028462540180772735, 0.03883163357849915, 
                   0.048459425901813596, 0.0559911075740453, 0.06081867893875083, 
                   0.06306828204779596, 0.06306828204779596, 0.06081867893875083, 
                   0.0559911075740453, 0.048459425901813596, 0.03883163357849915, 
                   0.028462540180772735, 0.018922316170672887, 0.011348251835047249, 
                   0.006119153432498831, 0.00296073118096772, 0.0012839434545941603, 
                   0.0004986984936138465, 0.0001734225211420432, 5.3982163639709224e-05, 
                   1.5038901292461017e-05, 3.7494761834611175e-06, 8.365563051497594e-07, 
                   1.6702411340276699e-07, 2.984121300506083e-08, 4.77093562631818e-09, 
                   6.825561933537406e-10, 8.738157756617853e-11, 1.0010304913027944e-11, 
                   1.0261745039366307e-12, 9.416959680063727e-14, 8.294403671350564e-15, 
                   8.294398461838712e-15, 9.416948132904792e-14, 1.0261723098004989e-12, 
                   1.0010267608232848e-11, 8.738101000752645e-11, 6.825484665141601e-10, 
                   4.770841494141655e-09, 2.9840186834945166e-08, 1.6701410313527425e-07, 
                   8.364689241397574e-07, 3.7487936349946034e-06, 1.5034130450966874e-05, 
                   5.3952323452874274e-05, 0.00017325550703890794, 0.0004978620246897067, 
                   0.0012801946609591658, 0.0029456970505167524, 0.006065201109045956, 
                   0.01117499632800834, 0.01842445414598318, 0.02718234551981357, 
                   0.035885936527982394, 0.042394224792767644, 0.04481611124603697, 
                   0.042394224792767644, 0.035885936527982394, 0.02718234551981357, 
                   0.01842445414598318, 0.01117499632800834]
        Prior[2] = [8.907565828806358e-24, 2.3619997230612154e-22, 5.604613927871268e-21, 
                   1.190026461900836e-19, 2.261062634913717e-18, 3.8442677225856766e-17, 
                   5.848704974081653e-16, 7.96252597375163e-15, 9.700342458717447e-14, 
                   1.057470666277978e-12, 1.0315603693384261e-11, 9.004633194485406e-11, 
                   7.033677658646436e-10, 4.916363141459434e-09, 3.075038122095107e-08, 
                   1.721084177217874e-07, 8.619831517493785e-07, 3.863140470010761e-06, 
                   1.5492706035987192e-05, 5.5597993508172754e-05, 0.00017854019881123783, 
                   0.0005130479624448763, 0.0013192435449304687, 0.0030355475911009915, 
                   0.006250203704035928, 0.011515859439142696, 0.018986442407707412, 
                   0.028011469627764306, 0.03698054019601583, 0.043687346233959874, 
                   0.046183105798865615, 0.043687346233959874, 0.03698054019601586, 
                   0.02801146962776489, 0.018986442407715375, 0.011515859439239698, 
                   0.006250203705093399, 0.0030355476014165953, 0.0013192436349768006, 
                   0.0005130486658126421, 0.00017854511517437926, 5.562874388939371e-05, 
                   1.5664814453708978e-05, 4.725123621760139e-06, 4.725123621760139e-06, 
                   1.5664814453708978e-05, 5.562874388939371e-05, 0.00017854511517437926, 
                   0.0005130486658126421, 0.0013192436349768008, 0.003035547601416598, 
                   0.006250203705093438, 0.011515859439240283, 0.018986442407723337, 
                   0.028011469627861895, 0.036980540197073336, 0.04368734624427548, 
                   0.04618310588891194, 0.043687346937327634, 0.03698054511237897, 
                   0.028011500378145527, 0.018986614516125135, 0.011516721422294445, 
                   0.006254066844505939, 0.003051040297136979, 0.0013748415384386415, 
                   0.0006915881612561141, 0.0006915881612561141, 0.0013748415384386415, 
                   0.003051040297136979, 0.006254066844505939, 0.011516721422294445, 
                   0.018986614516125135, 0.028011500378145527, 0.03698054511237897, 
                   0.043687346937327634, 0.04618310588891194, 0.04368734624427548, 
                   0.0369805401970733, 0.028011469627861312]
        Prior[4] = [3.07503813968816e-08, 1.721084187064622e-07, 8.619831566809971e-07, 
                   3.863140492112742e-06, 1.5492706124624792e-05, 5.5597993826262615e-05, 
                   0.00017854019983271042, 0.0005130479653801505, 0.001319243552478187, 
                   0.0030355476084681096, 0.006250203739794889, 0.011515859505027773, 
                   0.01898644251633354, 0.028011469788024847, 0.03698054040759064, 
                   0.04368734648390598, 0.046183106063090576, 0.04368734648390598, 
                   0.036980540407590676, 0.02801146978802543, 0.018986442516341507, 
                   0.011515859505124777, 0.006250203740852359, 0.0030355476187837135, 
                   0.0013192436425245196, 0.0005130486687479203, 0.00017854511619588002, 
                   5.5628744207659504e-05, 1.5664814543331254e-05, 4.7251236487937394e-06, 
                   4.7251236487937394e-06, 1.5664814543331254e-05, 5.5628744207659504e-05, 
                   0.00017854511619588002, 0.0005130486687479203, 0.0013192436425245196, 
                   0.0030355476187837135, 0.006250203740852359, 0.011515859505124777, 
                   0.018986442516341507, 0.02801146978802543, 0.036980540407590676, 
                   0.04368734648390598, 0.046183106063090576, 0.04368734648390598, 
                   0.03698054040759064, 0.028011469788024847, 0.01898644251633354, 
                   0.011515859505027773, 0.006250203739794889, 0.0030355476084681123, 
                   0.0013192435524782256, 0.0005130479653807353, 0.00017854019984067294, 
                   5.559799392326604e-05, 1.5492707182095466e-05, 3.863150807716494e-06, 
                   8.620732030134571e-07, 1.7281178647635096e-07, 3.5666744566468764e-08, 
                   3.5666744566468764e-08, 1.7281178647635096e-07, 8.620732030134571e-07, 
                   3.863150807716494e-06, 1.5492707182095466e-05, 5.559799392326604e-05, 
                   0.00017854019984067294, 0.0005130479653807353, 0.0013192435524782256, 
                   0.0030355476084681123, 0.006250203739794889, 0.011515859505027773, 
                   0.01898644251633354, 0.028011469788024847, 0.03698054040759064, 
                   0.04368734648390598, 0.046183106063090576, 0.04368734648390598, 
                   0.03698054040759064, 0.028011469788024847]
        Prior[5] = Prior[4]
        Prior_others = [0.0125] * 80
        for i in range(12):
            if i not in [1, 2, 4, 5]:
                Prior[i] = Prior_others

        

        Prior = torch.tensor(Prior).unsqueeze(0).expand(int(cls_embedding.shape[0]/(16*16))*16*16, -1,-1).to(device)
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
        num_classes = config['MODEL']['num_classes'] - len(config['MODEL']['ignore_background'])
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
        # 这里 class_acc 有问题

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
    num_classes = config['MODEL']['num_classes'] - len(config['MODEL']['ignore_background'])
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
        load_from_checkpoint(net, checkpoint, partial_restore=False, device='cpu')

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
