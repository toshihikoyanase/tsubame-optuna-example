#!/usr/bin/env python
# 
# Based on: https://github.com/pfnet/optuna/blob/master/examples/chainermn_simple.py
#

from __future__ import print_function

import chainer
import chainer.functions as F
import chainer.links as L
import chainermn
import numpy as np
import sys

import optuna

N_TRAIN_EXAMPLES = 3000
N_TEST_EXAMPLES = 1000
BATCHSIZE = 128
EPOCH = 10


def create_model(trial):
    # We optimize the numbers of layers and their units.
    n_layers = trial.suggest_int('n_layers', 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = int(trial.suggest_loguniform(
            'n_units_l{}'.format(i), 4, 128))
        layers.append(L.Linear(None, n_units))
        layers.append(F.relu)
    layers.append(L.Linear(None, 10))

    return chainer.Sequential(*layers)


# FYI: Objective functions can take additional arguments
# (https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args).
def objective(trial, comm):
    device = comm.intra_rank
    chainer.cuda.get_device_from_id(device).use()

    # Sample an architecture.
    model = L.Classifier(create_model(trial))
    model.to_gpu()

    # Setup optimizer.
    optimizer = chainer.optimizers.MomentumSGD()
    optimizer.setup(model)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)

    # Setup dataset and iterator. Only worker 0 loads the whole dataset.
    # The dataset of worker 0 is evenly split and distributed to all workers.
    if comm.rank == 0:
        train, test = chainer.datasets.get_mnist()
        rng = np.random.RandomState(0)
        train = chainer.datasets.SubDataset(
            train, 0, N_TRAIN_EXAMPLES, order=rng.permutation(len(train)))
        test = chainer.datasets.SubDataset(
            test, 0, N_TEST_EXAMPLES, order=rng.permutation(len(test)))
    else:
        train, test = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm)

    train_iter = chainer.iterators.SerialIterator(
        train, BATCHSIZE, shuffle=True)
    test_iter = chainer.iterators.SerialIterator(
        test, BATCHSIZE, repeat=False, shuffle=False)

    # Setup trainer.
    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (EPOCH, 'epoch'))

    if comm.rank == 0:
        trainer.extend(chainer.training.extensions.ProgressBar())

    # Run training.
    trainer.run()

    # Evaluate.
    evaluator = chainer.training.extensions.Evaluator(
        test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    report = evaluator()

    return report['main/accuracy']


if __name__ == '__main__':
    comm = chainermn.create_communicator('pure_nccl')

    # Create a study in rank-0 node and share it among all nodes.
    if comm.rank == 0:
        study = optuna.create_study()
        comm.mpi_comm.bcast(study.study_name)
        print('Study name:', study.study_name)
        print('Number of nodes:', comm.size)
    else:
        study_name = comm.mpi_comm.bcast(None)
        study = optuna.create_study(study_name=study_name)

    # Run optimization!
    chainermn_study = optuna.integration.ChainerMNStudy(study, comm)
    chainermn_study.optimize(objective, n_trials=10)

    if comm.rank == 0:
        print('Number of finished trials: ', len(study.trials))
        print('Best trial:')
        trial = study.best_trial
        print('  Value: ', trial.value)
        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))
