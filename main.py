# -*- coding:utf-8 -*-

import time
import json
import argparse

import numpy as np
import mxnet as mx

from utils import (construct_model, generate_data,
                   masked_mae_np, masked_mape_np, masked_mse_np)

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
parser.add_argument("--test", action="store_true", help="test program")
parser.add_argument("--plot", help="plot network graph", action="store_true")
parser.add_argument("--save", action="store_true", help="save model")
args = parser.parse_args()

config_filename = args.config

with open(config_filename, 'r') as f:
    config = json.loads(f.read())

print(json.dumps(config, sort_keys=True, indent=4))

net = construct_model(config)

batch_size = config['batch_size']
num_of_vertices = config['num_of_vertices']
graph_signal_matrix_filename = config['graph_signal_matrix_filename']
if isinstance(config['ctx'], list):
    ctx = [mx.gpu(i) for i in config['ctx']]
elif isinstance(config['ctx'], int):
    ctx = mx.gpu(config['ctx'])

loaders = []
true_values = []
for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
    if args.test:
        x = x[: 100]
        y = y[: 100]
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    loaders.append(
        mx.io.NDArrayIter(
            x, y if idx == 0 else None,
            batch_size=batch_size,
            shuffle=(idx == 0),
            label_name='label'
        )
    )
    if idx == 0:
        training_samples = x.shape[0]
    else:
        true_values.append(y)

train_loader, val_loader, test_loader = loaders
val_y, test_y = true_values

global_epoch = 1
global_train_steps = training_samples // batch_size + 1
all_info = []
epochs = config['epochs']

mod = mx.mod.Module(
    net,
    data_names=['data'],
    label_names=['label'],
    context=ctx
)

mod.bind(
    data_shapes=[(
        'data',
        (batch_size, config['points_per_hour'], num_of_vertices, 1)
    ), ],
    label_shapes=[(
        'label',
        (batch_size, config['points_per_hour'], num_of_vertices)
    )]
)

mod.init_params(initializer=mx.init.Xavier(magnitude=0.0003))
lr_sch = mx.lr_scheduler.PolyScheduler(
    max_update=global_train_steps * epochs * config['max_update_factor'],
    base_lr=config['learning_rate'],
    pwr=2,
    warmup_steps=global_train_steps
)

mod.init_optimizer(
    optimizer=config['optimizer'],
    optimizer_params=(('lr_scheduler', lr_sch),)
)

num_of_parameters = 0
for param_name, param_value in mod.get_params()[0].items():
    # print(param_name, param_value.shape)
    num_of_parameters += np.prod(param_value.shape)
print("Number of Parameters: {}".format(num_of_parameters), flush=True)

metric = mx.metric.create(['RMSE', 'MAE'], output_names=['pred_output'])

if args.plot:
    graph = mx.viz.plot_network(net)
    graph.format = 'png'
    graph.render('graph')


def training(epochs):
    global global_epoch
    lowest_val_loss = 1e6
    for _ in range(epochs):
        t = time.time()
        info = [global_epoch]
        train_loader.reset()
        metric.reset()
        for idx, databatch in enumerate(train_loader):
            mod.forward_backward(databatch)
            mod.update_metric(metric, databatch.label)
            mod.update()
        metric_values = dict(zip(*metric.get()))

        print('training: Epoch: %s, RMSE: %.2f, MAE: %.2f, time: %.2f s' % (
            global_epoch, metric_values['rmse'], metric_values['mae'],
            time.time() - t), flush=True)
        info.append(metric_values['mae'])

        val_loader.reset()
        prediction = mod.predict(val_loader)[1].asnumpy()
        loss = masked_mae_np(val_y, prediction, 0)
        print('validation: Epoch: %s, loss: %.2f, time: %.2f s' % (
            global_epoch, loss, time.time() - t), flush=True)
        info.append(loss)

        if loss < lowest_val_loss:

            test_loader.reset()
            prediction = mod.predict(test_loader)[1].asnumpy()
            tmp_info = []
            for idx in range(config['num_for_predict']):
                y, x = test_y[:, : idx + 1, :], prediction[:, : idx + 1, :]
                tmp_info.append((
                    masked_mae_np(y, x, 0),
                    masked_mape_np(y, x, 0),
                    masked_mse_np(y, x, 0) ** 0.5
                ))
            mae, mape, rmse = tmp_info[-1]
            print('test: Epoch: {}, MAE: {:.2f}, MAPE: {:.2f}, RMSE: {:.2f}, '
                  'time: {:.2f}s'.format(
                    global_epoch, mae, mape, rmse, time.time() - t))
            print(flush=True)
            info.extend((mae, mape, rmse))
            info.append(tmp_info)
            all_info.append(info)
            lowest_val_loss = loss

        global_epoch += 1


if args.test:
    epochs = 5
training(epochs)

the_best = min(all_info, key=lambda x: x[2])
print('step: {}\ntraining loss: {:.2f}\nvalidation loss: {:.2f}\n'
      'tesing: MAE: {:.2f}\ntesting: MAPE: {:.2f}\n'
      'testing: RMSE: {:.2f}\n'.format(*the_best))
print(the_best)

if args.save:
    mod.save_checkpoint('STSGCN', epochs)
