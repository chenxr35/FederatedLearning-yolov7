import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import torch
from tqdm import tqdm

from .utils import get_lr, get_momentum

from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn

import copy


class YoloDatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.epoch_now = -1

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        return self.dataset.__getitem__(self.idxs[item])


def eval_model(args, model, ema, gen_val, optimizer, yolo_loss, eval_callback, epoch):

    val_loss = 0

    epoch_step_val = len(gen_val)

    gen_val.dataset.epoch_now = epoch * args.local_ep

    print('Start Validation')
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{args.global_ep}',
                postfix=dict, mininterval=0.3)

    if ema:
        model_eval = ema.ema
    else:
        model_eval = model.eval()

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if args.cuda:
                images = images.cuda(args.local_rank)
                targets = targets.cuda(args.local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_eval(images)
            loss_value = yolo_loss(outputs, targets, images)

        val_loss += loss_value.item()
        pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
        pbar.update(1)

    eval_callback.on_epoch_end(epoch + 1, model_eval)

    return val_loss / epoch_step_val


def save_model(args, model, ema, loss_history, loss, val_loss, epoch, ckpt_dir):

    loss_history.append_loss(epoch + 1, loss, val_loss)
    print('Epoch:' + str(epoch + 1) + '/' + str(args.global_ep))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss, val_loss))

    # -----------------------------------------------#
    #   保存权值
    # -----------------------------------------------#
    if ema:
        save_state_dict = ema.ema.state_dict()
    else:
        save_state_dict = model.state_dict()

    if (epoch + 1) % args.save_period == 0 or epoch + 1 == args.global_ep:
        torch.save(save_state_dict, os.path.join(ckpt_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            (epoch + 1) * args.local_ep, loss, val_loss)))

    if len(loss_history.val_loss) <= 1 or (val_loss) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        torch.save(save_state_dict, os.path.join(ckpt_dir, "best_epoch_weights.pth"))

    torch.save(save_state_dict, os.path.join(ckpt_dir, "last_epoch_weights.pth"))


def fit_multiple_epoch(args, model, yolo_loss, optim_params, gen, lr_scheduler_func, scaler, global_round, client_id, control_local=None, control_global=None):

    if control_local and control_global:
        control_local_w = control_local.state_dict()
        control_global_w = control_global.state_dict()
        step_count = 0

    model.cuda()
    model.train()
    epoch_loss = []

    # ---------------------------------------#
    #   保存全局模型权值
    # ---------------------------------------#
    model_glob = copy.deepcopy(model)
    model_glob.cuda()
    for params in model_glob.parameters():
        params.requires_grad = False

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    optimizer = {
        'adam': optim.Adam(pg0, lr=0.001, betas=(args.momentum, 0.999)),
        'sgd': optim.SGD(pg0, lr=0.001, momentum=args.momentum, nesterov=True)
    }[args.optim_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": pg2})

    optimizer.load_state_dict(optim_params)

    epoch_step = len(gen)

    for epoch in range(args.local_ep):

        global_epoch = epoch + global_round * args.local_ep

        gen.dataset.epoch_now = global_epoch

        lr = lr_scheduler_func(global_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # fedavgm effective learning rate
            # if args.algo == 'fedavgm':
            #     param_group['lr'] = lr / (1 - args.avgm_beta)

        loss = 0

        print('\nStart Train')
        print(f"---- Training Model ---- Client{client_id} ----")
        pbar = tqdm(total=epoch_step, desc=f'Epoch {global_epoch + 1}/{args.local_ep * args.global_ep}', postfix=dict, mininterval=0.3)
        model.train()
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break

            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if args.cuda:
                    images = images.cuda(args.local_rank)
                    targets = targets.cuda(args.local_rank)
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            if not args.fp16:
                # ----------------------#
                #   前向传播
                # ----------------------#
                outputs = model(images)
                loss_value = yolo_loss(outputs, targets, images)

                # ----------------------#
                #   FedProx算法
                # ----------------------#
                if args.algo == 'fedprox':
                    proximal_term = 0.0
                    # iterate through the current and global model parameters
                    for w, w_t in zip(model.parameters(), model_glob.parameters()):
                        # update the proximal term
                        proximal_term += (w - w_t).norm(2)

                    loss_value += (args.mu / 2) * proximal_term

                # ----------------------#
                #   反向传播
                # ----------------------#
                loss_value.backward()
                optimizer.step()
            else:
                from torch.cuda.amp import autocast
                with autocast():
                    # ----------------------#
                    #   前向传播
                    # ----------------------#
                    outputs = model(images)
                    loss_value = yolo_loss(outputs, targets, images)

                # ----------------------#
                #   反向传播
                # ----------------------#
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()


            loss += loss_value.item()

            # ----------------------#
            #   SCAFFOLD算法
            # ----------------------#
            if args.algo == 'scaffold':
                local_weights = model.state_dict()
                for w in local_weights:
                    # line 10 in algo
                    local_weights[w] = local_weights[w] - get_lr(optimizer) * (control_global_w[w] - control_local_w[w])

                # update local model params
                model.load_state_dict(local_weights)

                step_count += 1

            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                    'lr': get_lr(optimizer)})
            pbar.update(1)

        pbar.close()
        print('Finish Train')

        epoch_loss.append(loss / epoch_step)

    # Compute model update
    model_update = {}
    for key in model.state_dict():
        model_update[key] = torch.sub(model_glob.state_dict()[key], model.state_dict()[key])

    # ----------------------#
    #   SCAFFOLD算法
    # ----------------------#
    if args.algo == 'scaffold':
        new_control_local_w = control_local.state_dict()
        control_delta = copy.deepcopy(control_local_w)
        # model_weights -> y_(i)
        model_weights = model.state_dict()
        global_weights = model_glob.state_dict()
        local_delta = copy.deepcopy(model_weights)
        K = step_count
        rho = get_momentum(optimizer)
        new_K = (K - rho * (1.0 - pow(rho, K)) / (1.0 - rho)) / (1.0 - rho)
        for w in model_weights:
            # line 12 in algo
            new_control_local_w[w] = new_control_local_w[w] - control_global_w[w] + (
                        global_weights[w] - model_weights[w]) / (new_K * get_lr(optimizer))
            # line 13
            control_delta[w] = new_control_local_w[w] - control_local_w[w]
            local_delta[w] -= global_weights[w]
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), control_delta, local_delta, new_control_local_w

    return model.state_dict(), sum(epoch_loss) / len(epoch_loss), model_update