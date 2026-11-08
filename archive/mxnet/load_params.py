# -*- coding:utf-8 -*-

import mxnet as mx

sym, arg_params, aux_params = mx.model.load_checkpoint('STSGCN', 200)

print(type(arg_params), type(aux_params))
