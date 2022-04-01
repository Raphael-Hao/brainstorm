#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /network.py
# \brief: visulizer for dynamic neural network
# Author: raphael hao

import networkx as nx
import abc


class Visualizer(abc.ABC):
    def __init__(
        self,
        net_name=None,
    ) -> None:
        self.graph = nx.grid_2d_graph(17, 4)
        
