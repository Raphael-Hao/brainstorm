#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Motto: Were It to Benefit My Country, I Would Lay Down My Life!
# \file: /gating.py
# \brief:
# Author: raphael hao

import abc


class Gate(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    def dispatch(self, *input):
        raise NotImplementedError

    def combine(self, *input):
        raise NotImplementedError
