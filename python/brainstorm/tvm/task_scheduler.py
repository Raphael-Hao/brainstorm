#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /task_scheduler.py
# \brief:
# Author: raphael hao

from tvm.auto_scheduler import TaskScheduler
from tvm.auto_scheduler.utils import array_mean

class EfficientTaskScheduler(TaskScheduler):
    """
    Allocate the time resources when tuning multiple tasks together.
    This implements two strategies: "round-robin" and "gradient".

    Parameters
    ----------
    tasks: List[SearchTask]
        All tasks to tune
    task_weights: Optional[List[float]]
        The weights of tasks.
        If provided, the task scheduler will set the objective function to
        sum(weight[t] * latency[t]), where weight[t] is the weight of a task
        and the lantecy[t] is the lantecy of the task.
        If not provided, the task scheduer will assign equal weights to all
        tasks (i.e., the objective function is sum(latency[t])).
    objective_func: Optional[Callable[List[float] -> float]]
        The objective function to be minimized.
        The objective function accepts the current latencies of all tasks and returns the
        objective.
        If not provided, the objective is the weighted sum of the latencies of all tasks.
    strategy: str = "gradient"
        The scheduling strategy.
        "round-robin": Tune tasks in round robin order.
        "gradient" : Tune tasks with gradient descent.
    load_model_file: Optional[str]
        Load pre-trained model from this file. If this is None, the cost model will
        be trained from scratch.
    load_log_file: Optional[str]
        Load measurement records from this file. If it is not None, the status of the
        task scheduler, search policies and cost models will be restored according to this file.
    verbose: int = 1
        The level of verbosity. 0 means silent.
    alpha: float = 0.2
        The parameter used for 'gradient' strategy
    beta: float = 2
        The parameter used for 'gradient' strategy
    backward_window_size: int = 3
        The parameter used for 'gradient' strategy
    callbacks: Optional[List[TaskSchedulerCallback]]
        The task scheduler callbacks that will be called before and after tuning a task.
        If None, PrintTableInfo and LogEstimatedLatency callback will be used.
    """

    def __init__(
        self,
        tasks,
        task_weights=None,
        objective_func=None,
        strategy="gradient",
        load_model_file: str = None,
        load_log_file: str = None,
        alpha: float = 0.2,
        beta: float = 2,
        gamma: float = 0.5,
        backward_window_size: int = 3,
        callbacks=None,
    ):
        super().__init__(
            tasks,
            task_weights,
            objective_func,
            strategy,
            load_model_file,
            load_log_file,
            alpha,
            beta,
            gamma,
            backward_window_size,
            callbacks,
        )

    def _tune_task(self, task_idx):
        """Tune the select task for one round"""

        # Run pre-tune callbacks
        for callback in self.callbacks:
            callback.pre_tune(self, task_idx)

        measure_inputs, measure_results = self.search_policies[
            task_idx
        ].continue_search_one_round(self.num_measures_per_round, self.measurer)

        self.task_cts[task_idx] += 1

        for inp, res in zip(measure_inputs, measure_results):
            cost = array_mean(res.costs)
            if cost < self.best_costs[task_idx]:
                self.task_best_cts[task_idx] = self.task_cts[task_idx]
                self.best_costs[task_idx] = cost

        # Stop tuning this task in the rest of the process if its search space has been
        # fully explored or it has no improvement for a long while.
        no_change_trials = (
            self.task_cts[task_idx] - self.task_best_cts[task_idx]
        ) * self.num_measures_per_round
        if len(measure_inputs) == 0 or no_change_trials > self.early_stopping_task:
            self.dead_tasks.add(task_idx)

        self.task_costs_history[task_idx].append(self.best_costs[task_idx])

        self.ct += len(measure_inputs)
        self.cur_score = self._compute_score(self.best_costs)

        # Run post-tune callbacks
        for callback in self.callbacks:
            callback.post_tune(self, task_idx)
