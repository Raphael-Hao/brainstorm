import sys
sys.path.append("/home/zhehan/tvm")

from transformers.utils import fx as t_fx

from transformers.models.bert.configuration_bert import BertConfig
from transformers.models.bert import BertLayer, BertModel, BertTokenizer
import traceback
import torch
# from torch.fx.passes.shape_prop import ShapeProp
from torch.fx.passes import graph_drawer
import graphviz
from mono_infer import MonoTensorInference
from sympy import *
from symbolic.symbolic_tensor import SymbolicTensor, symbolify, _simplify_symbol
import os



def test_sigmoid_neg():
    def test_fn(x):
        return torch.sigmoid(x).neg()
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (3,3))
    result = MonoTensorInference(gm).run(input)
    print(result.data)

def test_true_div():
    def test_fn(x):
        return x / 8.0
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (3,3))
    result = MonoTensorInference(gm).run(input)
    print(result.data)

def test_get_item():
    def test_fn(x):
        return x[0]
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (3,3))
    result = MonoTensorInference(gm).run(input)
    print(result.data)

def test_linear():
    class MyModel(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.linear = torch.nn.Linear(3,3)
        def forward(self, x):
            return self.linear(x)

    gm = torch.fx.symbolic_trace(MyModel())

    input = SymbolicTensor("input", (3,3))
    result = MonoTensorInference(gm).run(input)
    print(result.data)

def test_layer_norm():
    class MyModel(torch.nn.Module):
        def __init__(self,):
            super().__init__()
            self.layer_norm = torch.nn.LayerNorm(768)
        def forward(self, x):
            return self.layer_norm(x)

    gm = torch.fx.symbolic_trace(MyModel())

    input = SymbolicTensor("input", (1,8,768))
    result = MonoTensorInference(gm).run(input)

def test_softmax():
    def test_fn(x):
        return torch.nn.functional.softmax(x, dim=1)
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (3,4))
    result = MonoTensorInference(gm).run(input)
    print(result.data)


def test_matmul():
    def test_fn(x, y):
        # return torch.matmul(x, torch.Tensor([[0,0,1],[0,1,0],[1,0,0]])).size()
        return torch.matmul(x, y)
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (4,3,2))
    input2 = SymbolicTensor("input2", (4,2,3))
    result = MonoTensorInference(gm).run(input, input2)
    print(result)

def test_permute():
    def test_fn(x):
        # return torch.permute(torch.permute(x, (0, 2, 1, 3)), (0, 2, 1, 3))
        return x.permute(0,2,1,3).permute(0,2,1,3)
    gm = torch.fx.symbolic_trace(test_fn)

    input = SymbolicTensor("input", (3,3,3,3))
    print(input.data)
    result = MonoTensorInference(gm).run(input)
    print(result)

def test_simplify_symbol():
    def test_fn(x):
        return x[0]*x[1] + x[1]*x[2] + x[2]*x[0] + 8

    gm = torch.fx.symbolic_trace(test_fn)
    input = SymbolicTensor("input", (3))
    result = MonoTensorInference(gm).run(input)
    print(type(result))
    result.mono_simplify()
    print(result.data)

def test_bert():

    config = BertConfig(num_hidden_layers=1)
    with torch.no_grad():
        bert_layer = BertModel(config)
        graph_bert_layer = t_fx.symbolic_trace(bert_layer)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        text= ["Hello, my dog is cute", "Hello, sentense 2"]
        encoded_input = tokenizer(text, max_length=100,
                                add_special_tokens=True, truncation=True,
                                padding=True, return_tensors="pt")
        input_ids = symbolify(encoded_input['input_ids'], 'input_ids')
        attention_mask = encoded_input['attention_mask'].numpy()
        token_type_ids = encoded_input['token_type_ids'].numpy()
        positional_inputs = (input_ids, attention_mask, token_type_ids, None, None, None, None, None, None, None, None, None, None)

        result = MonoTensorInference(graph_bert_layer).run(*positional_inputs)
        # result['pooler_output'].mono_simplify()
        print(result['pooler_output'].data)

# test_matmul()
# test_permute(
# test_true_div()
# test_softmax()
# test_get_item()
# test_simplify_symbol()
# test_linear()
# test_layer_norm()
# test_simplify_symbol()
test_bert()
# test_sigmoid_neg()