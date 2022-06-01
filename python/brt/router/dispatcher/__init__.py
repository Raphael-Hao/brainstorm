#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /__init__.py
# \brief:
# Author: raphael hao

from .base import Dispatcher
from .default_dispatcher import DefaultDispatcher
from .fused_dispatcher import FusedDispatcher
from .moe_dispatcher import MoEDispatcher, TutelMoEDispatcher
from .residual_dispatcher import ResidualDispatcher
