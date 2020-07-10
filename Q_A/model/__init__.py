# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np


from model.CNNQLM_II import QA_quantum 
from model.CNNQLM_I_Flat import CNNQLM_I_Flat 
from model.vocab import CNNQLM_Vocab 
from model.dim   import CNNQLM_Dim
from model.CNNQLM_I   import CNNQLM_I 
from model.NNQLM_II   import NNQLM_II
from model.NNQLM_flat   import NNQLM_flat  

def setup(opt):
    
    if opt.model == 'QA_quantum':
        model = QA_quantum(opt)
    elif opt.model ==  'CNNQLM_I':
        model = CNNQLM_I(opt)
    elif opt.model == 'CNNQLM_I_Flat':
        model = CNNQLM_I_Flat(opt)
    elif opt.model == 'CNNQLM_Vocab' :
        model = CNNQLM_Vocab(opt)
    elif opt.model == 'CNNQLM_Dim' :
        model = CNNQLM_Dim(opt)
    elif opt.model == 'NNQLM_II' :
        model = NNQLM_II(opt)
    elif opt.model == 'NNQLM_flat' :
        model = NNQLM_flat(opt)

    else:
        raise Exception("model not supported: {}".format(opt.model))
    return model