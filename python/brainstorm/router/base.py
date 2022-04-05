#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# \file: /router.py
# \brief: base router for brainstorm
# Author: v-weihaocui
# Email: v-weihaocui@microsoft.com
from typing import List, Dict, OrderedDict
import abc


class Router(abc.ABC):
    def __init__(self) -> None:
        self.indices = None

    def route(self, input):
        if input <= 0:
            self.index = 0
        else:
            self.index = 1

    def scatter(self, input):
        return self.index, input
    
    def gather(self, input):
        return input

    def register_router(self):
        pass
