import imp
from turtle import speed
from brt.passes.base import PassBase
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
from msdnet import MSDNet
from adaptive_inference import Tester
from brt.passes import (
    HorizFusePass,
    VerticalFusePass,
    TracePass,
    NoBatchPass,
    DeadPathEliminatePass,
    PermanentPathFoldPass,
    OnDemandMemoryPlanPass,
    PredictMemoryPlanPass,
    OperatorReorderPass,
    ConstantPropagationPass,
)
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from brt.runtime.memory_planner import pin_memory
from brt.runtime.benchmark import (
    BenchmarkArgumentManager,
    Benchmarker,
    CUDATimer,
    MemoryStats,
    profile,
)
from brt.router import switch_router_mode


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
    tester = Tester(model1, args)
    if os.path.exists(os.path.join(args.save, "logits_single.pth")):
        val_pred, val_target, test_pred, test_target = torch.load(
            os.path.join(args.save, "logits_single.pth")
        )
    else:
        target_predict, val_target, n_batch = tester.calc(val_loader)
        torch.cuda.empty_cache()
        benchmarker = Benchmarker()

        def liveness_benchmark():
            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            naive_backbone = switch_router_mode(naive_backbone, False).eval()
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
                if i % 10 == 0:
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
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
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
                    reorder_operator_pass = OperatorReorderPass(new_backbone, runtime_load=1)
                    reorder_operator_pass.run_on_graph()
                    new_backbone = reorder_operator_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "reorder_operator")
                    output_dce = new_backbone(input_var)
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

                if i % 10 == 0:
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
                        sum(speed_up_of_reorder_operatorpass) / len(speed_up_of_reorder_operatorpass),
                    )

        benchmarker.add_benchmark("reorder_operator", reorder_operator_benchmark)

        def constant_propagation_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
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
                if i % 10 == 0:
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

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            DeadPathEliminatePass_time = []
            ConstProPass_time = []
            reorder_operatorPass_time = []
            speed_up_of_deadpatheliminatepass = []
            speed_up_of_constpropogationpass = []
            speed_up_of_reorder_operatorpass = []
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input)
                    import copy
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: naive_backbone(input_var), "naive")
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    
                    output_naive = model_copy(input_var)
                    
                    raw_pass=TracePass(model_copy)
                    raw_pass.run_on_graph()
                    raw_pass_graph=raw_pass.finalize()
                    graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    with open("raw_const.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: raw_pass_graph(input_var), "raw_pass_graph"
                    )
                    baseline_time.append(timer.avg)
                    
                    
                    eliminate_pass = DeadPathEliminatePass(raw_pass_graph, runtime_load=1)
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
                    reorder_operator_pass = OperatorReorderPass(new_backbone_const, runtime_load=1)
                    reorder_operator_pass.run_on_graph()
                    new_backbone = reorder_operator_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "reorder_operator")
                    output_reorder = new_backbone(input_var)
                    reorder_operatorPass_time.append(timer.avg)
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("dce_trans.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    
                    speed_up_of_deadpatheliminatepass.append(
                        baseline_time[-1] / DeadPathEliminatePass_time[-1]
                    )
                    speed_up_of_constpropogationpass.append(
                        baseline_time[-1] / ConstProPass_time[-1]
                    )
                    speed_up_of_reorder_operatorpass.append(
                        baseline_time[-1] / reorder_operatorPass_time[-1]
                    )
                if i % 10 == 0:
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
                        "max of speed_up_of_reorder_operatorpass",
                        max(speed_up_of_reorder_operatorpass),
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
                        "min of speed_up_of_reorder_operatorpass",
                        min(speed_up_of_reorder_operatorpass),
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
                    print(
                        "avg of speed_up_of_reorder_operatorpass",
                        sum(speed_up_of_reorder_operatorpass) / len(speed_up_of_reorder_operatorpass),
                    )

        benchmarker.add_benchmark("all_opt", all_opt_benchmark)

        
        def fuse_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            VerticalFusePass_time = []
            HorizonFusePass_time = []
            speed_up_of_verticalfusetepass = []
            speed_up_of_horizfusepass = []
            
            
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())
                    if i==0:
                        continue
                    print("i",i)
                    import copy
                    naive_backbone.cuda()
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    
                    model_copy=model_copy.cuda()
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    
                    baseline_time.append(timer.avg)
                    output_naive = model_copy(input_var)
                    
                    raw_pass=TracePass(model_copy)
                    raw_pass.run_on_graph()
                    raw_pass_graph=raw_pass.finalize()
                    graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    with open("raw_const.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: raw_pass_graph(input_var), "raw_pass_graph"
                    )
                    
                    print("begin pass")
                    vertical_fuse_pass=VerticalFusePass(raw_pass_graph)
                    vertical_fuse_pass.run_on_graph()
                    print("pass end")
                    
                    new_backbone=vertical_fuse_pass.finalize()
                    output1=new_backbone(input_var)
                    # import pdb;pdb.set_trace()
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("fuse.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    
                    print("begin test")
                    timer.execute(lambda: new_backbone(input_var), "vertical_fuse_pass")
                    output1=new_backbone(input_var)
                    
                    ## TODO check and add maxpool
                    # print("outputnaive",output_naive)
                    # print("output1",output1)
                    
                if i % 10 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    

        benchmarker.add_benchmark("vfuse", fuse_benchmark)
        
        
        def hfuse_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer

            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
            targets = []
            baseline_time = []
            DeadPathEliminatePass_time = []
            ConstProPass_time = []
            reorder_operatorPass_time = []
            speed_up_of_deadpatheliminatepass = []
            speed_up_of_constpropogationpass = []
            speed_up_of_reorder_operatorpass = []
            
            
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input.cuda())
                    if i==1: continue
                    print("i",i)
                    import copy
                    naive_backbone.train(False)
                    model_copy = copy.deepcopy(naive_backbone)
                    model_copy.train(False)
                    model_copy=model_copy.cuda()
                    ## to solve the decorator issue caused by DeepCopy
                    model_copy.final_gather.__class__ = (
                        naive_backbone.final_gather.__class__
                    )
                    for j in range(len(naive_backbone.scatters)):
                        model_copy.scatters[j].__class__ = naive_backbone.scatters[
                            j
                        ].__class__
                    timer.execute(lambda: model_copy(input_var), "naive2")
                    
                    baseline_time.append(timer.avg)
                    output_naive = model_copy(input_var)
                    
                    raw_pass=TracePass(model_copy)
                    raw_pass.run_on_graph()
                    raw_pass_graph=raw_pass.finalize()
                    graph_drawer = FxGraphDrawer(raw_pass_graph, "raw_pass_graph")
                    with open("raw_const.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(
                        lambda: raw_pass_graph(input_var), "raw_pass_graph"
                    )
                    print("begin pass")
                    vertical_fuse_pass=HorizFusePass(raw_pass_graph)
                    vertical_fuse_pass.run_on_graph()
                    print("pass end")
                    
                    new_backbone=vertical_fuse_pass.finalize()
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("hfuse.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    
                    print("begin test")
                    timer.execute(lambda: new_backbone(input_var), "vertical_fuse_pass")
                    output1=new_backbone(input_var)
                    
                    ## TODO check and add maxpool
                    # print("outputnaive",output_naive)
                    # print("output1",output1)
                    
                if i % 10 == 0:
                    print("Generate Logit: [{0}/{1}]".format(i, len(test_loader)))
                    

        benchmarker.add_benchmark("hfuse", hfuse_benchmark)
        
        def memroy_plan_benchmark():
            timer = CUDATimer(repeat=5)
            backbone_input = model1.backbone_input.detach().cuda()

            backbone = switch_router_mode(model1.backbone, False).eval()

            MemoryStats.reset_cuda_stats()

            timer.execute(lambda: backbone(backbone_input), "naive")

            MemoryStats.print_cuda_stats()

            backbone = pin_memory(backbone.cpu())

            # memory_plan_pass = OnDemandMemoryPlanPass(backbone)
            memory_plan_pass = PredictMemoryPlanPass(backbone, 500)
            memory_plan_pass.run_on_graph()
            new_backbone = memory_plan_pass.finalize()
            print(new_backbone.code)
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            MemoryStats.reset_cuda_stats()
            # profile(lambda: new_backbone(backbone_input))
            timer.execute(lambda: new_backbone(backbone_input), "on_demand_load")
            MemoryStats.print_cuda_stats()

        benchmarker.add_benchmark("memory_plan", memroy_plan_benchmark)

        print("Benchmarking... optimizer")

        benchmarker.benchmarking(args.benchmark)
