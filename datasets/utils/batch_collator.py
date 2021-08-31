#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

class BatchCollator(object):
    '''
    pack dict batch
    '''
    def __init__(self):
        super(BatchCollator,self).__init__()

    def __call__(self, batch):
        data= {}
        size = len(batch)
        for key in batch[0]:
            l = []
            for i in range(size):
                l = l + [batch[i][key]]
            data[key] = l
        return data
