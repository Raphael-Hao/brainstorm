# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import contextlib
import io
import itertools
import json
import math
import os
import subprocess
import unittest
from unittest.mock import patch

import GPUtil


class HelloworldCaller():
    """A class for run tutel helloworld example with different arguments"""
    def run(
        self,
        nproc_per_node=1,
        helloworld_file='helloworld',
        top=2, dtype='float32',
        num_local_experts='2',
        hidden_size=2048,
        show_step_time=True,
        batch_size=16,
        is_round=True,
        a2a_ffn_overlap_degree=1,
        num_steps=100
        ):
        # Disable NCCL SHM because it's capacity is limited in Azure pipeline
        new_env = os.environ.copy()
        new_env['NCCL_SHM_DISABLE'] = '1'
        """Run helloworld example"""
        if helloworld_file == 'helloworld':
            command = 'python3 -m torch.distributed.launch --nproc_per_node=' + str(nproc_per_node) + ' tutel/examples/helloworld.py --top ' + str(top) + ' --dtype ' + dtype + ' --num_local_experts ' + str(num_local_experts) + ' --hidden_size ' + str(hidden_size) + ' --batch_size ' + str(batch_size) + ' --a2a_ffn_overlap_degree ' + str(a2a_ffn_overlap_degree) + ' --num_steps ' + str(num_steps)
        if helloworld_file == 'helloworld_megatron':
            command = 'python3 -m torch.distributed.launch --nproc_per_node=' + str(nproc_per_node) + ' tutel/examples/helloworld_megatron.py --dtype ' + dtype + ' --num_local_experts ' + str(num_local_experts) + ' --hidden_size ' + str(hidden_size) + ' --batch_size ' + str(batch_size) + ' --num_steps ' + str(num_steps)
        p = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=new_env)
        losses = []
        while p.poll() is None:
            line = p.stdout.readline().decode("utf8").split()
            if len(line) > 5:
                if line[2] == 'loss':
                    if is_round:
                        if dtype == 'float32':
                            losses.append(round(float(line[4][:-1]), 3))
                        else:
                            losses.append(round(float(line[4][:-1]), 1))
                    else:
                        losses.append(line[4][:-1])
                if show_step_time and line[0] == '[Summary]':
                    print('step time:', line[5])
        p.stdout.close()
        return losses

class TutelTestCase(unittest.TestCase):
    """A class for tutel test cases."""
    def setUp(self):
        """Hook method for setting up the test"""
        self.GPUtype = GPUtil.getGPUs()[0].name
        with open('tests/test_baseline.json') as f:
            self.data = json.load(f)
        for i in range(9):
            for j in range(len(self.data[i]['losses'])):
                if '32' in self.data[i]['dtype']:
                    self.data[i]['losses'][j] = round(float(self.data[i]['losses'][j]), 3)
                else:
                    self.data[i]['losses'][j] = round(float(self.data[i]['losses'][j]), 1)
        self.tutelCaller = HelloworldCaller()

    def test_top1_fp32_1_expert(self):
        """Test helloworld with top1 gate, float32 dtype and 1 expert(s)."""
        for i in range(len(self.data[2]['step_time'])):
            if self.data[2]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[2]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=1, dtype='float32', num_local_experts=1), self.data[2]['losses'])

    def test_top1_fp32_2_experts(self):
        """Test helloworld with top1 gate, float32 dtype and 2 expert(s)."""
        for i in range(len(self.data[3]['step_time'])):
            if self.data[3]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[3]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=1, dtype='float32', num_local_experts=2), self.data[3]['losses'])

    def test_top1_fp16_1_expert(self):
        """Test helloworld with top1 gate, float16 dtype and 1 expert(s)."""
        for i in range(len(self.data[0]['step_time'])):
            if self.data[0]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[0]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=1, dtype='float16', num_local_experts=1)[0:2], self.data[0]['losses'][0:2])

    def test_top1_fp16_2_experts(self):
        """Test helloworld with top1 gate, float16 dtype and 2 expert(s)."""
        for i in range(len(self.data[1]['step_time'])):
            if self.data[1]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[1]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=1, dtype='float16', num_local_experts=2)[0:2], self.data[1]['losses'][0:2])

    def test_top2_fp32_1_expert(self):
        """Test helloworld with top2 gate, float32 dtype and 1 expert(s)."""
        for i in range(len(self.data[6]['step_time'])):
            if self.data[6]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[6]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=2, dtype='float32', num_local_experts=1), self.data[6]['losses'])

    def test_top2_fp32_2_experts(self):
        """Test helloworld with top2 gate, float32 dtype and 2 expert(s)."""
        for i in range(len(self.data[7]['step_time'])):
            if self.data[7]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[7]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=2, dtype='float32', num_local_experts=2), self.data[7]['losses'])

    def test_top2_fp16_1_expert(self):
        """Test helloworld with top2 gate, float16 dtype and 1 expert(s)."""
        for i in range(len(self.data[4]['step_time'])):
            if self.data[4]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[4]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=2, dtype='float16', num_local_experts=1)[0:2], self.data[4]['losses'][0:2])

    def test_top2_fp16_2_experts(self): 
        """Test helloworld with top2 gate, float16 dtype and 2 expert(s)."""
        for i in range(len(self.data[5]['step_time'])):
            if self.data[5]['step_time'][i]['GPU'] in self.GPUtype:
                print('\nexpected time:', self.data[5]['step_time'][i]['value'])
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=2, dtype='float16', num_local_experts=2)[0:2], self.data[5]['losses'][0:2])

    def test_top2_fp64_2_experts(self):
        """Test helloworld with top2 gate, float64 dtype and 2 expert(s)."""
        self.assertEqual(self.tutelCaller.run(nproc_per_node=1, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=2, show_step_time=False, batch_size=1), self.data[8]['losses'])

    def test_compare_megatron_with_tutel(self):
        """Test helloworld_megatron and helloworld which should have same result"""
        self.assertEqual(
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float32', num_local_experts=-2, show_step_time=False),
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld_megatron', dtype='float32', num_local_experts=1, hidden_size=1024, show_step_time=False)
            )

    def test_a2a_ffn_overlap(self):
        """Test whether AllToAll-FFN overlapping works properly. Note that too small batch size might cause precision issue."""
        self.assertEqual(
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=-2, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=1),
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=-2, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=2)
            )

        self.assertEqual(
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=1, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=1),
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=1, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=2)
            )

        self.assertEqual(
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=2, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=1),
            self.tutelCaller.run(nproc_per_node=2, helloworld_file='helloworld', top=2, dtype='float64', num_local_experts=2, show_step_time=False, batch_size=1, a2a_ffn_overlap_degree=2)
            )

    def test_a2a_algos(self):
        def get_loss_and_step_time(args):
            with contextlib.redirect_stdout(io.StringIO()) as f:
                loss = self.tutelCaller.run(**args)
            step_time = float(f.getvalue().strip().split()[-1])
            return loss, step_time

        for nproc_per_node, dtype, num_local_experts in itertools.product(
            [1, 2],
            ['float32', 'float16'],
            [1, 2],
        ):
            test_case = {
                'nproc_per_node': nproc_per_node,
                'helloworld_file': 'helloworld',
                'top': 2,
                'dtype': dtype,
                'num_local_experts': num_local_experts,
                'show_step_time': True,
                'num_steps': 50,
            }
            with self.subTest(msg='Testing a2a algo with setting', test_case=test_case):
                loss_expected, step_time_expected = get_loss_and_step_time(test_case)
                for algo in ['LINEAR', '2D']:
                    with patch.dict('os.environ', {
                        'TUTEL_ALLTOALL_ALGO': algo,
                        'LOCAL_SIZE': str(nproc_per_node),
                    }):
                        loss, step_time = get_loss_and_step_time(test_case)
                        self.assertEqual(loss, loss_expected)
                        self.assertTrue(math.isclose(step_time, step_time_expected, rel_tol=0.01, abs_tol=0.01))
