import imp
import math
import os
from turtle import speed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim

from brt.passes import (
    HorizFusePass,
    VerticalFusePass,
    TracePass,
    NoBatchPass,
    RouterFixPass,
    DeadPathEliminatePass,
    PermanentPathFoldPass,
    OnDemandMemoryPlanPass,
    PredictMemoryPlanPass,
    OperatorReorderPass,
    ConstantPropagationPass,
)
from brt.runtime.memory_planner import pin_memory
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
    profile,
)
from brt.router import switch_capture, ScatterRouter, GatherRouter

from msdnet import MSDNet
from adaptive_inference import Tester

def threshold_evaluate(
    model1: MSDNet, test_loader: DataLoader, val_loader: DataLoader, args
):
    print("threshold_evaluate  {}", args.thresholds)
    model1.build_routers(thresholds=args.thresholds)
    model1.eval()
    acc = 0
    for i, (input, target) in enumerate(val_loader):
        output = model1(input)
        pred = output.max(1, keepdim=True)[1]
        acc += pred.eq(target.view_as(pred)).sum().item()

    return acc * 100.0 / len(val_loader)


def threshold_dynamic_evaluate(
    model1: MSDNet, test_loader: DataLoader, val_loader: DataLoader, args
):
    model1 = switch_capture(model1, False).eval()
    tester = Tester(model1, args)
    # target_predict, val_target, n_batch = tester.calc(val_loader)
    model1 = switch_capture(model1, True, "max", "dispatch,combine").eval()
    if os.path.exists(os.path.join(args.save, "logits_single.pth")):
        val_pred, val_target, test_pred, test_target = torch.load(
            os.path.join(args.save, "logits_single.pth")
        )
    else:
        torch.cuda.empty_cache()
        benchmarker = Benchmarker()

        def liveness_benchmark():
            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            DeadPathEliminatePass_time = []
            PermanentPathFoldPass_time = []
            speed_up_of_deadpatheliminatepass = []
            speed_up_of_permaentpathfoldpass = []
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    if i == 0:
                        continue
                    input_var = torch.autograd.Variable(input)
                    timer.execute(lambda: naive_backbone(input_var), "naive")
                    baseline_time.append(timer.avg)
                    timer.execute(lambda: naive_backbone(input_var), "naive2")
                    eliminate_pass = DeadPathEliminatePass(
                        naive_backbone, runtime_load=1
                    )
                    eliminate_pass.run_on_graph()
                    new_backbone = eliminate_pass.finalize()
                    timer.execute(
                        lambda: new_backbone(input_var), "dead_path_eliminated"
                    )
                    DeadPathEliminatePass_time.append(timer.avg)
                    permanent_pass = PermanentPathFoldPass(
                        new_backbone, upper_perm_load=500
                    )
                    permanent_pass.run_on_graph()
                    new_backbone = permanent_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "path_permanent")
                    PermanentPathFoldPass_time.append(timer.avg)
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_permaentpathfoldpass.append(
                        baseline_time[-1] / PermanentPathFoldPass_time[-1]
                    )
                if i % 5 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max of speed_up_of_deadpatheliminatepass",
                        max(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "max of speed_up_of_permaentpathfoldpass",
                        max(speed_up_of_permaentpathfoldpass),
                    )
                    print(
                        "min of speed_up_of_deadpatheliminatepass",
                        min(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "min of speed_up_of_permaentpathfoldpass",
                        min(speed_up_of_permaentpathfoldpass),
                    )
                    print(
                        "avg of speed_up_of_deadpatheliminatepass",
                        sum(speed_up_of_deadpatheliminatepass)
                        / len(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "avg of speed_up_of_permaentpathfoldpass",
                        sum(speed_up_of_permaentpathfoldpass)
                        / len(speed_up_of_permaentpathfoldpass),
                    )
                    from torch.fx.passes.graph_drawer import FxGraphDrawer

                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("new_backbone.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())

        benchmarker.add_benchmark("liveness", liveness_benchmark)

        def reorder_operator_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            DeadPathEliminatePass_time = []
            reorder_operatorPass_time = []
            speed_up_of_deadpatheliminatepass = []
            speed_up_of_reorder_operatorpass = []
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input)
                    timer.execute(lambda: naive_backbone(input_var), "naive")
                    baseline_time.append(timer.avg)
                    import copy

                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    output_naive = naive_backbone(input_var)
                    eliminate_pass = DeadPathEliminatePass(model_copy, runtime_load=1)
                    eliminate_pass.run_on_graph()
                    new_backbone = eliminate_pass.finalize()
                    timer.execute(
                        lambda: new_backbone(input_var), "dead_path_eliminated"
                    )
                    DeadPathEliminatePass_time.append(timer.avg)
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("dce.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    reorder_operator_pass = OperatorReorderPass(
                        new_backbone, runtime_load=1
                    )
                    reorder_operator_pass.run_on_graph()
                    new_backbone = reorder_operator_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "reorder_operator")
                    reorder_operatorPass_time.append(timer.avg)
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("dce_trans.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    output_trans = new_backbone(input_var)
                    # print('naive output',output_naive)
                    # print('dce output',output_dce)
                    # print('trans output',output_trans)
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_reorder_operatorpass.append(
                        baseline_time[-1] / reorder_operatorPass_time[-1]
                    )

                if i % 5 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max of speed_up_of_deadpatheliminatepass",
                        max(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "max of speed_up_of_reorder_operatorpass",
                        max(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "min of speed_up_of_deadpatheliminatepass",
                        min(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "min of speed_up_of_reorder_operatorpass",
                        min(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "avg of speed_up_of_deadpatheliminatepass",
                        sum(speed_up_of_deadpatheliminatepass)
                        / len(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "avg of speed_up_of_reorder_operatorpass",
                        sum(speed_up_of_reorder_operatorpass)
                        / len(speed_up_of_reorder_operatorpass),
                    )

        benchmarker.add_benchmark("reorder_operator", reorder_operator_benchmark)

        def constant_propagation_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            DeadPathEliminatePass_time = []
            ConstProPass_time = []
            speed_up_of_deadpatheliminatepass = []
            speed_up_of_constpropogationpass = []
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input)
                    import copy

                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive")
                    timer.execute(lambda: model_copy(input_var), "naive2")

                    baseline_time.append(timer.avg)
                    output_naive = model_copy(input_var)
                    eliminate_pass = DeadPathEliminatePass(model_copy, runtime_load=1)
                    eliminate_pass.run_on_graph()
                    new_dce_backbone = eliminate_pass.finalize()
                    graph_drawer = FxGraphDrawer(new_dce_backbone, "new_backbone")
                    with open("dce_backbone_const.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_dce_backbone(input_var), "dead_path_eliminated"
                    )
                    DeadPathEliminatePass_time.append(timer.avg)
                    output_dce = new_dce_backbone(input_var)
                    constant_propagation_pass = ConstantPropagationPass(
                        new_dce_backbone, upper_perm_load=args.batch_size * n_batch
                    )
                    constant_propagation_pass.run_on_graph()
                    new_backbone_const = constant_propagation_pass.finalize()
                    graph_drawer = FxGraphDrawer(new_backbone_const, "new_backbone")
                    with open("dce_const_pro_backbone.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_backbone_const(input_var), "constant_propagation"
                    )
                    ConstProPass_time.append(timer.avg)
                    out_put_const = new_backbone_const(input_var)
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_constpropogationpass.append(
                        baseline_time[-1] / ConstProPass_time[-1]
                    )
                if i % 5 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max of speed_up_of_deadpatheliminatepass",
                        max(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "max of speed_up_of_constpropagation",
                        max(speed_up_of_constpropogationpass),
                    )
                    print(
                        "min of speed_up_of_deadpatheliminatepass",
                        min(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "min of speed_up_of_constpropagation",
                        min(speed_up_of_constpropogationpass),
                    )
                    print(
                        "avg of speed_up_of_deadpatheliminatepass",
                        sum(speed_up_of_deadpatheliminatepass)
                        / len(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "avg of speed_up_of_constpropagation",
                        sum(speed_up_of_constpropogationpass)
                        / len(speed_up_of_constpropogationpass),
                    )

        benchmarker.add_benchmark(
            "constant_propagation", constant_propagation_benchmark
        )

        def all_opt_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            num_trials = 100

            timer = CUDATimer(repeat=2, export_fname="msdnet_all_opt")
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            raw_time = []
            dpe_time = []
            cp_time = []
            vf_time = []
            opr_time = []
            sp_time = []
            hf_time = []

            with torch.no_grad():
                import copy

                naive_backbone.train(False)
                model_copy = copy.deepcopy(naive_backbone)
                model_copy.train(False)
                model_copy = model_copy.cuda()
                ## to solve the decorator issue caused by DeepCopy
                model_copy.final_gather.__class__ = (
                    naive_backbone.final_gather.__class__
                )
                for j in range(len(naive_backbone.scatters)):
                    model_copy.scatters[j].__class__ = naive_backbone.scatters[
                        j
                    ].__class__

                raw_pass = TracePass(model_copy)
                raw_pass.run_on_graph()
                raw_backbone = raw_pass.finalize()

                # Initialize load history
                raw_backbone = switch_capture(raw_backbone, True, mode="max", fabric_type="dispatch,combine").eval()
                for i, (input, target) in enumerate(test_loader):
                    if i >= 100:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    raw_backbone(input_var)
                raw_backbone = switch_capture(raw_backbone, False).eval()

                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: raw_backbone(input_var), "raw_backbone")
                    raw_time.append(timer.avg)

                rf_pass = RouterFixPass(raw_backbone)
                rf_pass.run_on_graph()
                rf_backbone = rf_pass.finalize()

                vf_pass = VerticalFusePass(rf_backbone, sample_inputs={"x": input_var}, fusing_head=True)
                vf_pass.run_on_graph()
                vf_backbone = vf_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: vf_backbone(input_var), "verti_fuse", export=True)
                    vf_time.append(timer.avg)

                dpe_pass = DeadPathEliminatePass(rf_backbone, runtime_load=1)
                dpe_pass.run_on_graph()
                dce_backbone = dpe_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(
                        lambda: dce_backbone(input_var), "dead_path_eliminated"
                    )
                    dpe_time.append(timer.avg)

                constant_propagation_pass = ConstantPropagationPass(
                    dce_backbone, upper_perm_load=1
                )
                constant_propagation_pass.run_on_graph()
                cp_backbone = constant_propagation_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(
                        lambda: cp_backbone(input_var), "constant_propagation"
                    )
                    cp_time.append(timer.avg)

                reorder_operator_pass = OperatorReorderPass(cp_backbone, runtime_load=1)
                reorder_operator_pass.run_on_graph()
                opr_backbone = reorder_operator_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: opr_backbone(input_var), "operator_reorder", export=True)
                    opr_time.append(timer.avg)

                sp_pass = VerticalFusePass(opr_backbone, sample_inputs={"x": input_var})
                sp_pass.run_on_graph()
                sp_backbone = sp_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: sp_backbone(input_var), "speculative_routing", export=True)
                    sp_time.append(timer.avg)

                horiz_fuse_pass = HorizFusePass(
                    opr_backbone, sample_inputs={"x": input_var}, fusing_head=True
                )
                horiz_fuse_pass.run_on_graph()
                hf_backbone = horiz_fuse_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: hf_backbone(input_var), "horiz_fuse_pass", export=True)
                    hf_time.append(timer.avg)

                t_baseline_time = torch.tensor(raw_time)
                t_dpe_time = torch.tensor(dpe_time)
                t_cp_time = torch.tensor(cp_time)
                t_vf_time = torch.tensor(vf_time)
                t_opr_time = torch.tensor(opr_time)
                t_sp_time = torch.tensor(sp_time)
                t_hf_time = torch.tensor(hf_time)

                print_stat(t_baseline_time, "Time of Raw Net")
                # print_stat(t_dpe_time, "Time of Dead Path Eliminate")
                # print_stat(t_cp_time, "Time of Constant Propagation")
                print_stat(t_vf_time, "Time of Vertical Fusion")
                # print_stat(t_ro_time, "Time of Operator Reorder")
                print_stat(t_sp_time, "Time of Speculative Routing")
                print_stat(t_hf_time, "Time of Horizontal Fusion")

                speed_up_of_dpe = t_baseline_time / t_dpe_time
                speed_up_of_cp = t_baseline_time / t_cp_time
                speed_up_of_vf = t_baseline_time / t_vf_time
                speed_up_of_opr = t_baseline_time / t_opr_time
                speed_up_of_sp = t_baseline_time / t_sp_time
                speed_up_of_hf = t_baseline_time / t_hf_time

                print("")
                # print_stat(speed_up_of_dpe, "Speedup of Dead Path Eliminate")
                # print_stat(speed_up_of_cp, "Speedup of Constant Propagation")
                print_stat(speed_up_of_vf, "Speedup of Vertical Fusion")
                # print_stat(speed_up_of_opr, "Speedup of Operator Reorder")
                print_stat(speed_up_of_sp, "Speedup of Speculative Routing")
                print_stat(speed_up_of_hf, "Speedup of Horizontal Fusion")
                print("")

        benchmarker.add_benchmark("all_opt", all_opt_benchmark)

        def fuse_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            VerticalFusePass_time = []
            speed_up_of_verticalfusetepass = []
            ConstProPass_time = []
            DeadPathEliminatePass_time = []
            reorder_operatorPass_time = []
            speed_up_of_constpropogationpass = []
            speed_up_of_reorder_operatorpass = []
            speed_up_of_deadpatheliminatepass = []

            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())
                    print("i", i)
                    import copy

                    naive_backbone.cuda()
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    model_copy = model_copy.cuda()
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    output_naive = model_copy(input_var)
                    raw_pass = TracePass(model_copy)
                    raw_pass.run_on_graph()
                    raw_pass_graph = raw_pass.finalize()
                    # graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    # with open("raw_const.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: raw_pass_graph(input_var), "raw_pass_graph")
                    baseline_time.append(timer.avg)
                    eliminate_pass = DeadPathEliminatePass(
                        raw_pass_graph, runtime_load=1
                    )
                    eliminate_pass.run_on_graph()
                    new_dce_backbone = eliminate_pass.finalize()
                    # graph_drawer = FxGraphDrawer(new_dce_backbone, "new_backbone")
                    # with open("_dce.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_dce_backbone(input_var), "dead_path_eliminated"
                    )
                    DeadPathEliminatePass_time.append(timer.avg)
                    output_dce = new_dce_backbone(input_var)
                    constant_propagation_pass = ConstantPropagationPass(
                        new_dce_backbone, upper_perm_load=args.batch_size * n_batch
                    )
                    constant_propagation_pass.run_on_graph()
                    new_backbone_const = constant_propagation_pass.finalize()
                    # graph_drawer = FxGraphDrawer(new_backbone_const, "new_backbone")
                    # with open("_dce_const.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_backbone_const(input_var), "constant_propagation"
                    )
                    ConstProPass_time.append(timer.avg)
                    out_put_const = new_backbone_const(input_var)
                    reorder_operator_pass = OperatorReorderPass(
                        new_backbone_const, runtime_load=1
                    )
                    reorder_operator_pass.run_on_graph()
                    new_backbone = reorder_operator_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "reorder_operator")
                    output_reorder = new_backbone(input_var)
                    reorder_operatorPass_time.append(timer.avg)
                    # graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    # with open("_dce_const_reorder.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    vertical_fuse_pass = VerticalFusePass(new_backbone)
                    vertical_fuse_pass.run_on_graph()
                    new_backbone = vertical_fuse_pass.finalize()
                    output1 = new_backbone(input_var)
                    # graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    # with open("_dce_const_reorder_vfuse.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: new_backbone(input_var), "vertical_fuse_pass")
                    VerticalFusePass_time.append(timer.avg)
                    output1 = new_backbone(input_var)
                    speed_up_of_verticalfusetepass.append(
                        baseline_time[-1] / VerticalFusePass_time[-1]
                    )
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_constpropogationpass.append(
                        baseline_time[-1] / ConstProPass_time[-1]
                    )
                    speed_up_of_reorder_operatorpass.append(
                        baseline_time[-1] / reorder_operatorPass_time[-1]
                    )

                if i % 2 == 0:
                    file = "recordingtest9.txt"

                    with open(file, "a") as f:
                        f.write(str(args.thresholds) + "\n")
                        f.write("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                        f.write("\n")
                        f.write("max speed up of dead_path_eliminate")
                        f.write(str(max(speed_up_of_deadpatheliminatepass)))
                        f.write("\n")
                        f.write("max speed up of constpropogationpass")
                        f.write(str(max(speed_up_of_constpropogationpass)))
                        f.write("\n")
                        f.write("max speed up of reorder_operatorpass")
                        f.write(str(max(speed_up_of_reorder_operatorpass)))
                        f.write("\n")
                        f.write("max speed up of vfusePass")
                        f.write(str(max(speed_up_of_verticalfusetepass)))
                        f.write("\n")

                        f.write("min speed up of dead_path_eliminate")
                        f.write(str(min(speed_up_of_deadpatheliminatepass)))
                        f.write("\n")
                        f.write("min speed up of constpropogationpass")
                        f.write(str(min(speed_up_of_constpropogationpass)))
                        f.write("\n")
                        f.write("min speed up of reorder_operatorpass")
                        f.write(str(min(speed_up_of_reorder_operatorpass)))
                        f.write("\n")
                        f.write("min speed up of vfusePass")
                        f.write(str(min(speed_up_of_verticalfusetepass)))
                        f.write("\n")
                        f.write("avg speed up of dead_path_eliminate")
                        f.write(
                            str(
                                sum(speed_up_of_deadpatheliminatepass)
                                / len(speed_up_of_deadpatheliminatepass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of constpropogationpass")
                        f.write(
                            str(
                                sum(speed_up_of_constpropogationpass)
                                / len(speed_up_of_constpropogationpass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of reorder_operatorpass")
                        f.write(
                            str(
                                sum(speed_up_of_reorder_operatorpass)
                                / len(speed_up_of_reorder_operatorpass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of vfusePass")
                        f.write(
                            str(
                                sum(speed_up_of_verticalfusetepass)
                                / len(speed_up_of_verticalfusetepass)
                            )
                        )
                        f.write("\n")
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max speed up of dead_path_eliminate",
                        max(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "max speed up of constpropogationpass",
                        max(speed_up_of_constpropogationpass),
                    )
                    print(
                        "max speed up of reorder_operatorpass",
                        max(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "max of speed_up_of_verticalfusetepass",
                        max(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "min speed up of dead_path_eliminate",
                        min(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "min speed up of constpropogationpass",
                        min(speed_up_of_constpropogationpass),
                    )
                    print(
                        "min speed up of reorder_operatorpass",
                        min(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "min of speed_up_of_verticalfusetepass",
                        min(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "avg speed up of dead_path_eliminate",
                        sum(speed_up_of_deadpatheliminatepass)
                        / len(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "avg speed up of constpropogationpass",
                        sum(speed_up_of_constpropogationpass)
                        / len(speed_up_of_constpropogationpass),
                    )
                    print(
                        "avg speed up of reorder_operatorpass",
                        sum(speed_up_of_reorder_operatorpass)
                        / len(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "avg of speed_up_of_verticalfusetepass",
                        sum(speed_up_of_verticalfusetepass)
                        / len(speed_up_of_verticalfusetepass),
                    )
                    if i == 8:
                        file = "recordingtest9.txt"
                        with open(file, "a") as f:
                            f.write("finish vfuse")
                        break

        benchmarker.add_benchmark("vfuse", fuse_benchmark)

        def only_vfuse_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            VerticalFusePass_time = []
            speed_up_of_verticalfusetepass = []
            ConstProPass_time = []
            DeadPathEliminatePass_time = []
            reorder_operatorPass_time = []
            speed_up_of_constpropogationpass = []
            speed_up_of_reorder_operatorpass = []
            speed_up_of_deadpatheliminatepass = []

            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())
                    print("i", i)
                    import copy

                    naive_backbone.cuda()
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    model_copy = model_copy.cuda()
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    output_naive = model_copy(input_var)
                    raw_pass = TracePass(model_copy)
                    raw_pass.run_on_graph()
                    raw_pass_graph = raw_pass.finalize()
                    # graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    # with open("raw_const.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: raw_pass_graph(input_var), "raw_pass_graph")
                    print(timer.avg)
                    baseline_time.append(timer.avg)
                    print(
                        "finish baseline all avg{}".format(
                            sum(baseline_time) / len(baseline_time)
                        )
                    )
                    if i == 8:
                        print("finish baseline")
                        break
                    continue
                    vertical_fuse_pass = VerticalFusePass(raw_pass_graph)
                    vertical_fuse_pass.run_on_graph()
                    new_backbone = vertical_fuse_pass.finalize()
                    output1 = new_backbone(input_var)
                    # graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    # with open("_dce_const_reorder_vfuse.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: new_backbone(input_var), "vertical_fuse_pass")
                    VerticalFusePass_time.append(timer.avg)
                    output1 = new_backbone(input_var)
                    speed_up_of_verticalfusetepass.append(
                        baseline_time[-1] / VerticalFusePass_time[-1]
                    )

                if i % 2 == 0:
                    file = "recordingtest9.txt"

                    with open(file, "a") as f:
                        f.write(str(args.thresholds) + "\n")
                        f.write("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                        f.write("\n")
                        f.write("max speed up of vfusePass")
                        f.write(str(max(speed_up_of_verticalfusetepass)))
                        f.write("\n")
                        f.write("min speed up of vfusePass")
                        f.write(str(min(speed_up_of_verticalfusetepass)))
                        f.write("\n")
                        f.write("avg speed up of vfusePass")
                        f.write(
                            str(
                                sum(speed_up_of_verticalfusetepass)
                                / len(speed_up_of_verticalfusetepass)
                            )
                        )
                        f.write("\n")
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max of speed_up_of_verticalfusetepass",
                        max(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "min of speed_up_of_verticalfusetepass",
                        min(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "avg of speed_up_of_verticalfusetepass",
                        sum(speed_up_of_verticalfusetepass)
                        / len(speed_up_of_verticalfusetepass),
                    )
                    if i == 8:
                        file = "recordingtest9.txt"
                        with open(file, "a") as f:
                            f.write("finish vfuse")
                        break

        benchmarker.add_benchmark("only_vfuse", only_vfuse_benchmark)

        def vfusetrans_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_capture(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            VerticalFusePass_time = []
            speed_up_of_verticalfusetepass = []
            ConstProPass_time = []
            DeadPathEliminatePass_time = []
            reorder_operatorPass_time = []
            speed_up_of_constpropogationpass = []
            speed_up_of_reorder_operatorpass = []
            speed_up_of_deadpatheliminatepass = []

            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())
                    print("i", i)
                    import copy

                    naive_backbone.cuda()
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    model_copy = model_copy.cuda()
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    output_naive = model_copy(input_var)
                    raw_pass = TracePass(model_copy)
                    raw_pass.run_on_graph()
                    new_backbone = raw_pass.finalize()
                    # graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    # with open("raw_const.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: new_backbone(input_var), "raw_pass_graph")
                    baseline_time.append(timer.avg)

                    vertical_fuse_pass = VerticalFusePass(new_backbone)
                    vertical_fuse_pass.run_on_graph()
                    new_backbone = vertical_fuse_pass.finalize()
                    output1 = new_backbone(input_var)
                    # graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    # with open("_dce_const_reorder_vfuse.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: new_backbone(input_var), "vertical_fuse_pass")
                    VerticalFusePass_time.append(timer.avg)
                    output1 = new_backbone(input_var)
                    eliminate_pass = DeadPathEliminatePass(new_backbone, runtime_load=1)
                    eliminate_pass.run_on_graph()
                    new_dce_backbone = eliminate_pass.finalize()
                    # graph_drawer = FxGraphDrawer(new_dce_backbone, "new_backbone")
                    # with open("_dce.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_dce_backbone(input_var), "dead_path_eliminated"
                    )
                    DeadPathEliminatePass_time.append(timer.avg)
                    output_dce = new_dce_backbone(input_var)
                    constant_propagation_pass = ConstantPropagationPass(
                        new_dce_backbone, upper_perm_load=args.batch_size * n_batch
                    )
                    constant_propagation_pass.run_on_graph()
                    new_backbone_const = constant_propagation_pass.finalize()
                    # graph_drawer = FxGraphDrawer(new_backbone_const, "new_backbone")
                    # with open("_dce_const.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: new_backbone_const(input_var), "constant_propagation"
                    )
                    ConstProPass_time.append(timer.avg)
                    out_put_const = new_backbone_const(input_var)
                    reorder_operator_pass = OperatorReorderPass(
                        new_backbone_const, runtime_load=1
                    )
                    reorder_operator_pass.run_on_graph()
                    new_backbone = reorder_operator_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "reorder_operator")
                    output_reorder = new_backbone(input_var)
                    reorder_operatorPass_time.append(timer.avg)
                    # graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    # with open("_dce_const_reorder.svg", "wb") as f:
                    #     f.write(graph_drawer.get_dot_graph().create_svg())

                    speed_up_of_verticalfusetepass.append(
                        baseline_time[-1] / VerticalFusePass_time[-1]
                    )
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_constpropogationpass.append(
                        baseline_time[-1] / ConstProPass_time[-1]
                    )
                    speed_up_of_reorder_operatorpass.append(
                        baseline_time[-1] / reorder_operatorPass_time[-1]
                    )
                    # import pdb; pdb.set_trace()

                if i % 2 == 0:
                    file = "recordingtest9.txt"

                    with open(file, "a") as f:
                        f.write(str(args.thresholds) + "\n")
                        f.write("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                        f.write("\n")
                        f.write('avg timer of "naive2"')
                        f.write(str(sum(baseline_time) / len(baseline_time)))
                        f.write("\n")
                        f.write('avg timer of "vertical_fuse_pass"')
                        f.write(
                            str(sum(VerticalFusePass_time) / len(VerticalFusePass_time))
                        )
                        f.write("\n")
                        f.write('avg timer of "dead_path_eliminated"')
                        f.write(
                            str(
                                sum(DeadPathEliminatePass_time)
                                / len(DeadPathEliminatePass_time)
                            )
                        )
                        f.write("\n")
                        f.write('avg timer of "constant_propagation"')
                        f.write(str(sum(ConstProPass_time) / len(ConstProPass_time)))
                        f.write("\n")
                        f.write('avg timer of "reorder_operator"')
                        f.write(
                            str(
                                sum(reorder_operatorPass_time)
                                / len(reorder_operatorPass_time)
                            )
                        )
                        f.write("\n")
                        f.write("max speed up of dead_path_eliminate")
                        f.write(str(max(speed_up_of_deadpatheliminatepass)))
                        f.write("\n")
                        f.write("max speed up of constpropogationpass")
                        f.write(str(max(speed_up_of_constpropogationpass)))
                        f.write("\n")
                        f.write("max speed up of reorder_operatorpass")
                        f.write(str(max(speed_up_of_reorder_operatorpass)))
                        f.write("\n")
                        f.write("max speed up of vfusePass")
                        f.write(str(max(speed_up_of_verticalfusetepass)))
                        f.write("\n")

                        f.write("min speed up of dead_path_eliminate")
                        f.write(str(min(speed_up_of_deadpatheliminatepass)))
                        f.write("\n")
                        f.write("min speed up of constpropogationpass")
                        f.write(str(min(speed_up_of_constpropogationpass)))
                        f.write("\n")
                        f.write("min speed up of reorder_operatorpass")
                        f.write(str(min(speed_up_of_reorder_operatorpass)))
                        f.write("\n")
                        f.write("min speed up of vfusePass")
                        f.write(str(min(speed_up_of_verticalfusetepass)))
                        f.write("\n")
                        f.write("avg speed up of dead_path_eliminate")
                        f.write(
                            str(
                                sum(speed_up_of_deadpatheliminatepass)
                                / len(speed_up_of_deadpatheliminatepass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of constpropogationpass")
                        f.write(
                            str(
                                sum(speed_up_of_constpropogationpass)
                                / len(speed_up_of_constpropogationpass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of reorder_operatorpass")
                        f.write(
                            str(
                                sum(speed_up_of_reorder_operatorpass)
                                / len(speed_up_of_reorder_operatorpass)
                            )
                        )
                        f.write("\n")
                        f.write("avg speed up of vfusePass")
                        f.write(
                            str(
                                sum(speed_up_of_verticalfusetepass)
                                / len(speed_up_of_verticalfusetepass)
                            )
                        )
                        f.write("\n")
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print(
                        "max speed up of dead_path_eliminate",
                        max(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "max speed up of constpropogationpass",
                        max(speed_up_of_constpropogationpass),
                    )
                    print(
                        "max speed up of reorder_operatorpass",
                        max(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "max of speed_up_of_verticalfusetepass",
                        max(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "min speed up of dead_path_eliminate",
                        min(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "min speed up of constpropogationpass",
                        min(speed_up_of_constpropogationpass),
                    )
                    print(
                        "min speed up of reorder_operatorpass",
                        min(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "min of speed_up_of_verticalfusetepass",
                        min(speed_up_of_verticalfusetepass),
                    )
                    print(
                        "avg speed up of dead_path_eliminate",
                        sum(speed_up_of_deadpatheliminatepass)
                        / len(speed_up_of_deadpatheliminatepass),
                    )
                    print(
                        "avg speed up of constpropogationpass",
                        sum(speed_up_of_constpropogationpass)
                        / len(speed_up_of_constpropogationpass),
                    )
                    print(
                        "avg speed up of reorder_operatorpass",
                        sum(speed_up_of_reorder_operatorpass)
                        / len(speed_up_of_reorder_operatorpass),
                    )
                    print(
                        "avg of speed_up_of_verticalfusetepass",
                        sum(speed_up_of_verticalfusetepass)
                        / len(speed_up_of_verticalfusetepass),
                    )
                    if i == 8:
                        file = "recordingtest9.txt"
                        with open(file, "a") as f:
                            f.write("finish vfuse")
                        break

        benchmarker.add_benchmark("vfuse_trans", vfusetrans_benchmark)

        def hfuse_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            num_trials = 1

            timer = CUDATimer(repeat=2)
            naive_backbone = model1
            # naive_backbone = switch_capture(naive_backbone, False).eval()
            raw_time = []
            hf_time = []
            cp_time = []
            dpe_time = []
            opr_time = []

            with torch.no_grad():
                import copy

                naive_backbone.train(False)
                model_copy = copy.deepcopy(naive_backbone)
                model_copy.train(False)
                model_copy = model_copy.cuda()
                ## to solve the decorator issue caused by DeepCopy
                model_copy.final_gather.__class__ = (
                    naive_backbone.final_gather.__class__
                )
                for j in range(len(naive_backbone.scatters)):
                    model_copy.scatters[j].__class__ = naive_backbone.scatters[
                        j
                    ].__class__
                model_copy = switch_capture(model_copy, True).eval()

                raw_pass = TracePass(model_copy)
                raw_pass.run_on_graph()
                raw_backbone = raw_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: raw_backbone(input_var), "raw_backbone")
                    raw_time.append(timer.avg)

                raw_backbone = switch_capture(raw_backbone, False).eval()

                dpe_pass = DeadPathEliminatePass(raw_backbone, runtime_load=1)
                dpe_pass.run_on_graph()
                dce_backbone = dpe_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(
                        lambda: dce_backbone(input_var), "dead_path_eliminated"
                    )
                    dpe_time.append(timer.avg)

                constant_propagation_pass = ConstantPropagationPass(
                    dce_backbone, upper_perm_load=args.batch_size * n_batch
                )
                constant_propagation_pass.run_on_graph()
                cp_backbone = constant_propagation_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(
                        lambda: cp_backbone(input_var), "constant_propagation"
                    )
                    cp_time.append(timer.avg)

                reorder_operator_pass = OperatorReorderPass(cp_backbone, runtime_load=1)
                reorder_operator_pass.run_on_graph()
                opr_backbone = reorder_operator_pass.finalize()
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: opr_backbone(input_var), "reorder_operator")
                    opr_time.append(timer.avg)

                horiz_fuse_pass = HorizFusePass(
                    opr_backbone, sample_inputs={"x": input_var}, fusing_head=True
                )
                horiz_fuse_pass.run_on_graph()
                hf_backbone = horiz_fuse_pass.finalize()
                print(hf_backbone)
                for i, (input, target) in enumerate(test_loader):
                    if i >= num_trials:
                        break
                    input_var = torch.autograd.Variable(input.cuda())
                    timer.execute(lambda: hf_backbone(input_var), "horiz_fuse_pass")
                    hf_time.append(timer.avg)

                t_baseline_time = torch.tensor(raw_time)
                t_dpe_time = torch.tensor(dpe_time)
                t_cp_time = torch.tensor(cp_time)
                t_ro_time = torch.tensor(opr_time)
                t_hf_time = torch.tensor(hf_time)

                print_stat(t_baseline_time, "Time of Raw Net")
                print_stat(t_dpe_time, "Time of Dead Path Eliminate")
                print_stat(t_cp_time, "Time of Constant Propagation")
                print_stat(t_ro_time, "Time of Operator Reorder")
                print_stat(t_hf_time, "Time of Horizontal Fusion")

                speed_up_of_dpe = t_baseline_time / t_dpe_time
                speed_up_of_cp = t_baseline_time / t_cp_time
                speed_up_of_ro = t_baseline_time / t_ro_time
                speed_up_of_hf = t_baseline_time / t_hf_time

                print("")
                print_stat(speed_up_of_dpe, "Speedup of Dead Path Eliminate")
                print_stat(speed_up_of_cp, "Speedup of Constant Propagation")
                print_stat(speed_up_of_ro, "Speedup of Operator Reorder")
                print_stat(speed_up_of_hf, "Speedup of Horizontal Fusion")
                print("")

        benchmarker.add_benchmark("hfuse", hfuse_benchmark)

        def capture_overhead_benchmark():
            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            targets = []
            active_time = []
            in_active_time = []
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    naive_backbone = switch_capture(naive_backbone, True).eval()
                    timer.execute(lambda: naive_backbone(input), "active_capture")
                    active_time.append(timer.avg)
                    naive_backbone = switch_capture(naive_backbone, False).eval()
                    timer.execute(lambda: naive_backbone(input), "inactive_capture")
                    in_active_time.append(timer.avg)

                if i % 5 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    print("avg active_time", sum(active_time) / len(active_time))
                    print(
                        "avg in_active_time", sum(in_active_time) / len(in_active_time)
                    )
                    if i == 5:
                        break
            return

        benchmarker.add_benchmark("capture", capture_overhead_benchmark)

        print("Benchmarking... optimizer")

        benchmarker.benchmarking(args.benchmark)


def print_stat(tensor: torch.Tensor, header: str = None):
    if header is not None:
        print(header)
    print(
        f"    avg: {tensor.mean():.4f}, "
        f"max: {tensor.max():.4f}, "
        f"min: {tensor.min():.4f}"
    )
