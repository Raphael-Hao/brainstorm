from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import os
import math

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')):
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth'))
    else:
        val_pred, val_target = tester.calc_logit(val_loader)
        test_pred, test_target = tester.calc_logit(test_loader)
        torch.save((val_pred, val_target, test_pred, test_target),
                    os.path.join(args.save, 'logits_single.pth'))

    flops = torch.load(os.path.join(args.save, 'flops.pth'))

    with open(os.path.join(args.save, 'dynamic.txt'), 'w') as fout:
        for p in range(1, 40):
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            probs = torch.exp(torch.log(_p) * torch.range(1, args.nBlocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(
                val_pred, val_target, probs, flops)
            acc_test, exp_flops = tester.dynamic_eval_with_threshold(
                test_pred, test_target, flops, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test flops: {:.2f}M'.format(acc_val, acc_test, exp_flops / 1e6))
            fout.write('{}\t{}\n'.format(acc_test, exp_flops.item()))


class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()
        self.softmax2 = nn.Softmax(dim=0).cuda()
        

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])

                    logits[b].append(_t)

            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def calc(self, dataloader):
        self.model.eval()
        n_stage = self.args.nBlocks
        logits = [[] for _ in range(n_stage)]
        targets = []

        for i, (input, target) in enumerate(dataloader):
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model(input_var)
                # output= self.softmax2(output)
                if i==0:
                    soft_max_result=output
                else:
                    soft_max_result=torch.cat((soft_max_result,output),0)
            if i==1:
                    break
            if i % self.args.print_freq == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))
                
        ##get the argmax_preds
        max_preds, argmax_preds = soft_max_result.max(dim=1, keepdim=False)
        ##for debug
        with open("/home/yichuanjiaoda/brainstorm_project/brainstorm/benchmark/msdnet/debug","w") as variable_name:
            for i,result in enumerate(argmax_preds) :
                variable_name.write(str(result.item())+"\n")
        
        print(argmax_preds.sum())
        ##compare to the gold_result targets
        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(soft_max_result.size(0)).copy_(targets)

        return argmax_preds, ts_targets
    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops