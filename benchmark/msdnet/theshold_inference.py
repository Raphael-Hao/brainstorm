import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import math
from msdnet import MSDNet
from adaptive_inference import Tester
from brt.passes import (
    DeadPathEliminatePass,
    PermanentPathFoldPass,
    OnDemandMemoryPlanPass,
    PredictMemoryPlanPass,
    TransformPass,
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

def threshold_evaluate(model1: MSDNet, test_loader: DataLoader,val_loader: DataLoader, args):
    print("threshold_evaluate  {}",args.thresholds)
    model1.build_routers(thresholds=args.thresholds)
    model1.eval()
    acc = 0
    for i, (input, target) in enumerate(val_loader):
        output = model1(input)
        pred = output.max(1, keepdim=True)[1]
        acc += pred.eq(target.view_as(pred)).sum().item()

    return acc * 100.0 / len(val_loader)


def threshold_dynamic_evaluate(model1: MSDNet, test_loader: DataLoader,val_loader: DataLoader, args):
    tester = Tester(model1, args)
    if os.path.exists(os.path.join(args.save, 'logits_single.pth')):
        val_pred, val_target, test_pred, test_target = \
            torch.load(os.path.join(args.save, 'logits_single.pth'))
    else:
        target_predict, val_target = tester.calc(val_loader)
        torch.cuda.empty_cache()

        benchmarker = Benchmarker()

        def liveness_benchmark():
            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            from torch.fx.passes.graph_drawer import FxGraphDrawer
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
            targets = []
            baseline_time=[]
            DeadPathEliminatePass_time=[]
            PermanentPathFoldPass_time=[]
            speed_up_of_deadpatheliminatepass=[]
            speed_up_of_permaentpathfoldpass=[]
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    if i==0:
                        continue
                    input_var = torch.autograd.Variable(input)
                    timer.execute(lambda: naive_backbone(input_var), "naive")
                    baseline_time.append(timer.avg)
                    timer.execute(lambda: naive_backbone(input_var), "naive2")
                    eliminate_pass = DeadPathEliminatePass(naive_backbone, runtime_load=1)
                    eliminate_pass.run_on_graph()
                    new_backbone = eliminate_pass.finalize() 
                    timer.execute(lambda: new_backbone(input_var), "dead_path_eliminated")
                    DeadPathEliminatePass_time.append(timer.avg)
                    permanent_pass = PermanentPathFoldPass(new_backbone, upper_perm_load=500)
                    permanent_pass.run_on_graph()
                    new_backbone = permanent_pass.finalize()
                    timer.execute(lambda: new_backbone(input_var), "path_permanent")
                    PermanentPathFoldPass_time.append(timer.avg)
                    speed_up_of_deadpatheliminatepass.append(baseline_time[-1]/DeadPathEliminatePass_time[-1])
                    speed_up_of_permaentpathfoldpass.append(baseline_time[-1]/PermanentPathFoldPass_time[-1])
                if i % 10 == 0:
                    print('Generate Logit: [{0}/{1}]'.format(i, len(test_loader)))
                    print('max of speed_up_of_deadpatheliminatepass',max(speed_up_of_deadpatheliminatepass))
                    print('max of speed_up_of_permaentpathfoldpass',max(speed_up_of_permaentpathfoldpass))
                    print('min of speed_up_of_deadpatheliminatepass',min(speed_up_of_deadpatheliminatepass))
                    print('min of speed_up_of_permaentpathfoldpass',min(speed_up_of_permaentpathfoldpass))
                    print('avg of speed_up_of_deadpatheliminatepass',sum(speed_up_of_deadpatheliminatepass)/len(speed_up_of_deadpatheliminatepass))
                    print('avg of speed_up_of_permaentpathfoldpass',sum(speed_up_of_permaentpathfoldpass)/len(speed_up_of_permaentpathfoldpass))
                    from torch.fx.passes.graph_drawer import FxGraphDrawer
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("new_backbone.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())



        benchmarker.add_benchmark("liveness", liveness_benchmark)

        
        
        def transform_benchmark():
            from torch.fx.passes.graph_drawer import FxGraphDrawer
            
            timer = CUDATimer(repeat=5)
            naive_backbone = model1
            naive_backbone = switch_router_mode(naive_backbone, False).eval()
            targets = []
            baseline_time=[]
            DeadPathEliminatePass_time=[]
            TransformPass_time=[]
            speed_up_of_deadpatheliminatepass=[]
            speed_up_of_transformpass=[]
            for i, (input, target) in enumerate(test_loader):
                targets.append(target)
                with torch.no_grad():
                    input_var = torch.autograd.Variable(input)
                    timer.execute(lambda: naive_backbone(input_var), "naive")
                    baseline_time.append(timer.avg)
                    timer.execute(lambda: naive_backbone(input_var), "naive2")  
                    output_naive=naive_backbone(input_var)
                    eliminate_pass = DeadPathEliminatePass(naive_backbone, runtime_load=1)
                    eliminate_pass.run_on_graph()
                    new_backbone = eliminate_pass.finalize() 
                    timer.execute(lambda: new_backbone(input_var), "dead_path_eliminated")
                    DeadPathEliminatePass_time.append(timer.avg)
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    transform_pass = TransformPass(new_backbone, runtime_load=1)
                    transform_pass.run_on_graph()
                    new_backbone=transform_pass.finalize()
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("dce_trans_backbone.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    timer.execute(lambda: new_backbone(input_var), "transform")
                    output_dce=new_backbone(input_var)
                    
                    TransformPass_time.append(timer.avg)
                    graph_drawer = FxGraphDrawer(new_backbone, "new_backbone")
                    with open("transform_dce_trans_backbone.svg", "wb") as f:
                        f.write(graph_drawer.get_dot_graph().create_svg())
                    output_trans=new_backbone(input_var)
                    
                    # print('naive output',output_naive)
                    # print('dce output',output_dce)
                    # print('trans output',output_trans)
                    
                    speed_up_of_deadpatheliminatepass.append(baseline_time[-1]/DeadPathEliminatePass_time[-1])
                    speed_up_of_transformpass.append(baseline_time[-1]/TransformPass_time[-1])
                if i % 10 == 0:
                    print('Generate Logit: [{0}/{1}]'.format(i, len(test_loader)))
                    print('max of speed_up_of_deadpatheliminatepass',max(speed_up_of_deadpatheliminatepass))
                    print('max of speed_up_of_transformpass',max(speed_up_of_transformpass))
                    print('min of speed_up_of_deadpatheliminatepass',min(speed_up_of_deadpatheliminatepass))
                    print('min of speed_up_of_transformpass',min(speed_up_of_transformpass))
                    print('avg of speed_up_of_deadpatheliminatepass',sum(speed_up_of_deadpatheliminatepass)/len(speed_up_of_deadpatheliminatepass))
                    print('avg of speed_up_of_transformpass',sum(speed_up_of_transformpass)/len(speed_up_of_transformpass))
                    
                    
                    
                    
        benchmarker.add_benchmark("transform", transform_benchmark)
        
        
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