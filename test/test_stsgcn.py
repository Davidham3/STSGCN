# -*- coding:utf-8 -*-

import sys
import mxnet as mx

sys.path.append('.')
num_of_vertices = 358
batch_size = 16
filter_ = [3, 3, 3]
filter_list = [[3, 3, 3], [6, 6, 6], [9, 9, 9]]
predict_length = 12
data = mx.sym.var('data')
adj = mx.sym.var('adj')
label = mx.sym.var('label')


def test_position_embedding():
    from models.stsgcn import position_embedding

    for temporal_emb in (True, False):
        for spatial_emb in (True, False):
            net = position_embedding(
                data, 12, num_of_vertices, 32,
                temporal_emb, spatial_emb
            )
            assert net.infer_shape(
                data=(batch_size, 12, num_of_vertices, 32)
            )[1][0] == (batch_size, 12, num_of_vertices, 32)


def test_gcn_operation():
    from models.stsgcn import gcn_operation
    for activation in ('GLU', 'relu'):
        net = gcn_operation(
            data, adj, 64, 32, num_of_vertices, activation, "gcn_operation"
        )
        assert net.infer_shape(
            data=(3 * num_of_vertices, batch_size, 32),
            adj=(3 * num_of_vertices, 3 * num_of_vertices)
        )[1][0] == (3 * num_of_vertices, batch_size, 64)


def test_stsgcm():
    from models.stsgcn import stsgcm

    for activation in ('GLU', 'relu'):
        net = stsgcm(
            data, adj,
            filter_, 32, num_of_vertices,
            activation, "stsgcm"
        )
        assert net.infer_shape(
            data=(3 * num_of_vertices, batch_size, 32),
            adj=(3 * num_of_vertices, 3 * num_of_vertices)
        )[1][0] == (num_of_vertices, batch_size, filter_[-1])


def test_stsgcl():
    from models.stsgcn import stsgcl
    for module_type in ('sharing', 'individual'):
        for activation in ('GLU', 'relu'):
            for temporal_emb in (True, False):
                for spatial_emb in (True, False):
                    net = stsgcl(
                        data, adj, 12, num_of_vertices, 32, filter_,
                        module_type, activation,
                        temporal_emb, spatial_emb, "sthgcl"
                    )
                    assert net.infer_shape(
                        data=(batch_size, 12, num_of_vertices, 32),
                        adj=(3 * num_of_vertices, 3 * num_of_vertices)
                    )[1][0] == (
                        batch_size, 10, num_of_vertices, filter_[-1])


def test_output_layer():
    from models.stsgcn import output_layer
    net = output_layer(data, num_of_vertices, 12, 32,
                       num_of_filters=128, predict_length=predict_length)
    assert net.infer_shape(
        data=(batch_size, 12, num_of_vertices, 32)
    )[1][0] == (batch_size, predict_length, num_of_vertices)


def test_huber_loss():
    from models.stsgcn import huber_loss
    net = huber_loss(data, label, rho=1)
    assert net.infer_shape(
        data=(batch_size, 12, num_of_vertices),
        label=(batch_size, 12, num_of_vertices)
    )[1][0] == (batch_size, 12, num_of_vertices)


def test_weighted_loss():
    from models.stsgcn import weighted_loss
    net = weighted_loss(data, label, 12, rho=1)
    assert net.infer_shape(
        data=(batch_size, 12, num_of_vertices),
        label=(batch_size, 12, num_of_vertices)
    )[1][0] == (batch_size, 12, num_of_vertices)


def test_stsgcn():
    from models.stsgcn import stsgcn
    import numpy as np
    mask_init_value = mx.init.Constant(value=np.random.uniform(
        size=(3 * num_of_vertices, 3 * num_of_vertices)).tolist())
    for module_type in ('sharing', 'individual'):
        for activation in ('GLU', 'relu'):
            for temporal_emb in (True, False):
                for spatial_emb in (True, False):
                    for use_mask in (True, False):
                        net = stsgcn(
                            data, adj, label,
                            12, num_of_vertices, 32, filter_list,
                            module_type, activation,
                            use_mask, mask_init_value,
                            temporal_emb, spatial_emb,
                            prefix="stsgcn", rho=1, predict_length=12
                        )
                        assert net.infer_shape(
                            data=(batch_size, 12, num_of_vertices, 32),
                            adj=(3 * num_of_vertices, 3 * num_of_vertices),
                            label=(batch_size, 12, num_of_vertices)
                        )[1][0] == (batch_size, 12, num_of_vertices)


def test_model_construction():
    import json
    import os
    from utils import construct_model
    config_folder = 'config/PEMS08'
    for file_name in os.listdir(config_folder):
        config_file = os.path.join(config_folder, file_name)
        with open(config_file, 'r') as f:
            config = json.loads(f.read().strip())
        config['adj_filename'] = 'data/PEMS08/PEMS08.csv'
        config['id_filename'] = None
        construct_model(config)
