# -*- coding:utf-8 -*-

import mxnet as mx


def position_embedding(data,
                       input_length, num_of_vertices, embedding_size,
                       temporal=True, spatial=True,
                       init=mx.init.Xavier(magnitude=0.0003), prefix=""):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    input_length: int, length of time series, T

    num_of_vertices: int, N

    embedding_size: int, C

    temporal, spatial: bool, whether equip this type of embeddings

    init: mx.initializer.Initializer

    prefix: str

    Returns
    ----------
    data: output shape is (B, T, N, C)
    '''

    temporal_emb = None
    spatial_emb = None

    if temporal:
        # shape is (1, T, 1, C)
        temporal_emb = mx.sym.var(
            "{}_t_emb".format(prefix),
            shape=(1, input_length, 1, embedding_size),
            init=init
        )
    if spatial:
        # shape is (1, 1, N, C)
        spatial_emb = mx.sym.var(
            "{}_v_emb".format(prefix),
            shape=(1, 1, num_of_vertices, embedding_size),
            init=init
        )

    if temporal_emb is not None:
        data = mx.sym.broadcast_add(data, temporal_emb)
    if spatial_emb is not None:
        data = mx.sym.broadcast_add(data, spatial_emb)

    return data


def gcn_operation(data, adj,
                  num_of_filter, num_of_features, num_of_vertices,
                  activation, prefix=""):
    '''
    graph convolutional operation, a simple GCN we defined in paper

    Parameters
    ----------
    data: mx.sym.var, shape is (3N, B, C)

    adj: mx.sym.var, shape is (3N, 3N)

    num_of_filter: int, C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (3N, B, C')

    '''

    assert activation in {'GLU', 'relu'}

    # shape is (3N, B, C)
    data = mx.sym.dot(adj, data)

    if activation == 'GLU':

        # shape is (3N, B, 2C')
        data = mx.sym.FullyConnected(
            data,
            flatten=False,
            num_hidden=2 * num_of_filter
        )

        # shape is (3N, B, C'), (3N, B, C')
        lhs, rhs = mx.sym.split(data, num_outputs=2, axis=2)

        # shape is (3N, B, C')
        return lhs * mx.sym.sigmoid(rhs)

    elif activation == 'relu':

        # shape is (3N, B, C')
        return mx.sym.Activation(
            mx.sym.FullyConnected(
                data,
                flatten=False,
                num_hidden=num_of_filter
            ), activation
        )


def stsgcm(data, adj,
           filters, num_of_features, num_of_vertices,
           activation, prefix=""):
    '''
    STSGCM, multiple stacked gcn layers with cropping and max operation

    Parameters
    ----------
    data: mx.sym.var, shape is (3N, B, C)

    adj: mx.sym.var, shape is (3N, 3N)

    filters: list[int], list of C'

    num_of_features: int, C

    num_of_vertices: int, N

    activation: str, {'GLU', 'relu'}

    prefix: str

    Returns
    ----------
    output shape is (N, B, C')

    '''
    need_concat = []

    for i in range(len(filters)):
        data = gcn_operation(
            data, adj,
            filters[i], num_of_features, num_of_vertices,
            activation=activation,
            prefix="{}_gcn_{}".format(prefix, i)
        )
        need_concat.append(data)
        num_of_features = filters[i]

    # shape of each element is (1, N, B, C')
    need_concat = [
        mx.sym.expand_dims(
            mx.sym.slice(
                i,
                begin=(num_of_vertices, None, None),
                end=(2 * num_of_vertices, None, None)
            ), 0
        ) for i in need_concat
    ]

    # shape is (N, B, C')
    return mx.sym.max(mx.sym.concat(*need_concat, dim=0), axis=0)


def stsgcl(data, adj,
           T, num_of_vertices, num_of_features, filters,
           module_type, activation, temporal_emb=True, spatial_emb=True,
           prefix=""):
    '''
    STSGCL

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    module_type: str, {'sharing', 'individual'}

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    assert module_type in {'sharing', 'individual'}

    if module_type == 'individual':
        return sthgcn_layer_individual(
            data, adj,
            T, num_of_vertices, num_of_features, filters,
            activation, temporal_emb, spatial_emb, prefix
        )
    else:
        return sthgcn_layer_sharing(
            data, adj,
            T, num_of_vertices, num_of_features, filters,
            activation, temporal_emb, spatial_emb, prefix
        )


def sthgcn_layer_individual(data, adj,
                            T, num_of_vertices, num_of_features, filters,
                            activation, temporal_emb=True, spatial_emb=True,
                            prefix=""):
    '''
    STSGCL, multiple individual STSGCMs

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    # shape is (B, T, N, C)
    data = position_embedding(data, T, num_of_vertices, num_of_features,
                              temporal_emb, spatial_emb,
                              prefix="{}_emb".format(prefix))
    need_concat = []
    for i in range(T - 2):

        # shape is (B, 3, N, C)
        t = mx.sym.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

        # shape is (B, 3N, C)
        t = mx.sym.reshape(t, (-1, 3 * num_of_vertices, num_of_features))

        # shape is (3N, B, C)
        t = mx.sym.transpose(t, (1, 0, 2))

        # shape is (N, B, C')
        t = stsgcm(
            t, adj, filters, num_of_features, num_of_vertices,
            activation=activation,
            prefix="{}_stsgcm_{}".format(prefix, i)
        )

        # shape is (B, N, C')
        t = mx.sym.swapaxes(t, 0, 1)

        # shape is (B, 1, N, C')
        need_concat.append(mx.sym.expand_dims(t, axis=1))

    # shape is (B, T-2, N, C')
    return mx.sym.concat(*need_concat, dim=1)


def sthgcn_layer_sharing(data, adj,
                         T, num_of_vertices, num_of_features, filters,
                         activation, temporal_emb=True, spatial_emb=True,
                         prefix=""):
    '''
    STSGCL, multiple a sharing STSGCM

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    adj: mx.sym.var, shape is (3N, 3N)

    T: int, length of time series, T

    num_of_vertices: int, N

    num_of_features: int, C

    filters: list[int], list of C'

    activation: str, {'GLU', 'relu'}

    temporal_emb, spatial_emb: bool

    prefix: str

    Returns
    ----------
    output shape is (B, T-2, N, C')
    '''

    # shape is (B, T, N, C)
    data = position_embedding(data, T, num_of_vertices, num_of_features,
                              temporal_emb, spatial_emb,
                              prefix="{}_emb".format(prefix))
    need_concat = []
    for i in range(T - 2):
        # shape is (B, 3, N, C)
        t = mx.sym.slice(data, begin=(None, i, None, None),
                         end=(None, i + 3, None, None))

        # shape is (B, 3N, C)
        t = mx.sym.reshape(t, (-1, 3 * num_of_vertices, num_of_features))

        # shape is (3N, B, C)
        t = mx.sym.swapaxes(t, 0, 1)
        need_concat.append(t)

    # shape is (3N, (T-2)*B, C)
    t = mx.sym.concat(*need_concat, dim=1)

    # shape is (N, (T-2)*B, C')
    t = stsgcm(
        t, adj, filters, num_of_features, num_of_vertices,
        activation=activation,
        prefix="{}_stsgcm".format(prefix)
    )

    # shape is (N, T - 2, B, C)
    t = t.reshape((num_of_vertices, T - 2, -1, filters[-1]))

    # shape is (B, T - 2, N, C)
    return mx.sym.swapaxes(t, 0, 2)


def output_layer(data, num_of_vertices, input_length, num_of_features,
                 num_of_filters=128, predict_length=12):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T, N, C)

    num_of_vertices: int, N

    input_length: int, length of time series, T

    num_of_features: int, C

    num_of_filters: int, C'

    predict_length: int, length of predicted time series, T'

    Returns
    ----------
    output shape is (B, T', N)
    '''

    # data shape is (B, N, T, C)
    data = mx.sym.swapaxes(data, 1, 2)

    # (B, N, T * C)
    data = mx.sym.reshape(
        data, (-1, num_of_vertices, input_length * num_of_features)
    )

    # (B, N, C')
    data = mx.sym.Activation(
        mx.sym.FullyConnected(
            data,
            flatten=False,
            num_hidden=num_of_filters
        ), 'relu'
    )

    # (B, N, T')
    data = mx.sym.FullyConnected(
        data,
        flatten=False,
        num_hidden=predict_length
    )

    # (B, T', N)
    data = mx.sym.swapaxes(data, 1, 2)

    return data


def huber_loss(data, label, rho=1):
    '''
    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    rho: float

    Returns
    ----------
    loss: mx.sym
    '''

    loss = mx.sym.abs(data - label)
    loss = mx.sym.where(loss > rho, loss - 0.5 * rho,
                        (0.5 / rho) * mx.sym.square(loss))
    loss = mx.sym.MakeLoss(loss)
    return loss


def weighted_loss(data, label, input_length, rho=1):
    '''
    weighted loss build on huber loss

    Parameters
    ----------
    data: mx.sym.var, shape is (B, T', N)

    label: mx.sym.var, shape is (B, T', N)

    input_length: int, T'

    rho: float

    Returns
    ----------
    agg_loss: mx.sym
    '''

    # shape is (1, T, 1)
    weight = mx.sym.expand_dims(
        mx.sym.expand_dims(
            mx.sym.flip(mx.sym.arange(1, input_length + 1), axis=0),
            axis=0
        ), axis=-1
    )
    agg_loss = mx.sym.broadcast_mul(
        huber_loss(data, label, rho),
        weight
    )
    return agg_loss


def stsgcn(data, adj, label,
           input_length, num_of_vertices, num_of_features,
           filter_list, module_type, activation,
           use_mask=True, mask_init_value=None,
           temporal_emb=True, spatial_emb=True,
           prefix="", rho=1, predict_length=12):
    '''
    data shape is (B, T, N, C)
    adj shape is (3N, 3N)
    label shape is (B, T, N)
    '''
    if use_mask:
        if mask_init_value is None:
            raise ValueError("mask init value is None!")
        mask = mx.sym.var("{}_mask".format(prefix),
                          shape=(3 * num_of_vertices, 3 * num_of_vertices),
                          init=mask_init_value)
        adj = mask * adj

    for idx, filters in enumerate(filter_list):
        data = stsgcl(data, adj, input_length, num_of_vertices,
                      num_of_features, filters, module_type,
                      activation=activation,
                      temporal_emb=temporal_emb,
                      spatial_emb=spatial_emb,
                      prefix="{}_stsgcl_{}".format(prefix, idx))
        input_length -= 2
        num_of_features = filters[-1]

    # (B, 1, N)
    need_concat = []
    for i in range(predict_length):
        need_concat.append(
            output_layer(
                data, num_of_vertices, input_length, num_of_features,
                num_of_filters=128, predict_length=1
            )
        )
    data = mx.sym.concat(*need_concat, dim=1)

    loss = huber_loss(data, label, rho=rho)
    return mx.sym.Group([loss, mx.sym.BlockGrad(data, name='pred')])
