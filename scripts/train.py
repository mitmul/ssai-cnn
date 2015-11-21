#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.insert(0, 'scripts/utils')
os.environ["CHAINER_TYPE_CHECK"] = "0"

import time
import imp
import shutil
import logging
import lmdb
import chainer
import numpy as np
from chainer import cuda, optimizers, serializers, Variable
from transform import Transform
from create_args import create_args
from multiprocessing import Process, Queue
from draw_loss import draw_loss


def create_result_dir(args):
    result_dir = 'results/{}_{}'.format(
        os.path.splitext(os.path.basename(args.model))[0],
        time.strftime('%Y-%m-%d_%H-%M-%S'))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log_fn = '%s/log.txt' % result_dir
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] %(message)s',
        filename=log_fn, level=logging.DEBUG)
    logging.info(args)

    return log_fn, result_dir


def get_model_optimizer(args):
    model_fn = os.path.basename(args.model)
    model = imp.load_source(model_fn.split('.')[0], args.model).model

    if 'result_dir' in args:
        dst = '%s/%s' % (args.result_dir, model_fn)
        if not os.path.exists(dst):
            shutil.copy(args.model, dst)

        dst = '%s/%s' % (args.result_dir, os.path.basename(__file__))
        if not os.path.exists(dst):
            shutil.copy(__file__, dst)

    # prepare model
    if args.gpu >= 0:
        model.to_gpu()

    if 'opt' in args:
        # prepare optimizer
        if args.opt == 'MomentumSGD':
            optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
            optimizer.add_hook(chainer.optimizer.WeightDecay(args.decay))
        elif args.opt == 'Adam':
            optimizer = optimizers.Adam(alpha=args.alpha)
        elif args.opt == 'AdaGrad':
            optimizer = optimizers.AdaGrad(lr=args.lr)
        else:
            raise Exception('No optimizer is selected')
        optimizer.setup(model)

        return model, optimizer
    else:
        print('No optimizer generated.')
        return model


def create_minibatch(args, o_cur, l_cur, data_queue):
    trans = Transform(args)
    x_minibatch = []
    y_minibatch = []
    while True:
        o_key, o_val = o_cur.item()
        l_key, l_val = l_cur.item()
        if o_key != l_key:
            raise ValueError(
                'Keys of ortho and label patches are different: '
                '{} != {}'.format(o_key, l_key))

        # prepare patch
        o_side = args.ortho_original_side
        l_side = args.label_original_side
        o_patch = np.fromstring(
            o_val, dtype=np.uint8).reshape((o_side, o_side, 3))
        l_patch = np.fromstring(
            l_val, dtype=np.uint8).reshape((l_side, l_side, 1))
        o_aug, l_aug = trans.transform(o_patch, l_patch)

        # add patch
        x_minibatch.append(o_aug)
        y_minibatch.append(l_aug)

        o_ret = o_cur.next()
        l_ret = l_cur.next()

        if ((len(x_minibatch) == args.batchsize)
                or ((not o_ret) and (not l_ret))):
            x_minibatch = np.asarray(
                x_minibatch, dtype=np.float32).transpose((0, 3, 1, 2))
            y_minibatch = np.asarray(
                y_minibatch, dtype=np.int32).transpose((0, 3, 1, 2))
            data_queue.put((x_minibatch, y_minibatch))
            x_minibatch = []
            y_minibatch = []

        if ((not o_ret) and (not l_ret)):
            break

    data_queue.put(None)


def get_cursor(db_fn):
    env = lmdb.open(db_fn)
    txn = env.begin(write=False, buffers=False)
    cur = txn.cursor()
    cur.next()

    return cur, txn, env.stat()['entries']


def one_epoch(args, model, optimizer, epoch, train):
    model.train = train
    xp = cuda.cupy if args.gpu >= 0 else np

    # open datasets
    ortho_db = args.train_ortho_db if train else args.valid_ortho_db
    label_db = args.train_label_db if train else args.valid_label_db
    o_cur, o_txn, args.N = get_cursor(ortho_db)
    l_cur, l_txn, _ = get_cursor(label_db)

    # for parallel augmentation
    data_queue = Queue()
    mbatch_worker = Process(target=create_minibatch,
                            args=(args, o_cur, l_cur, data_queue))
    mbatch_worker.start()

    sum_loss = 0
    num = 0
    while True:
        minibatch = data_queue.get()
        if minibatch is None:
            break
        x, t = minibatch
        volatile = 'off' if train else 'on'
        x = Variable(xp.asarray(x), volatile=volatile)
        t = Variable(xp.asarray(t), volatile=volatile)

        if train:
            optimizer.update(model, x, t)
        else:
            model(x, t)

        sum_loss += float(model.loss.data) * t.data.shape[0]
        num += t.data.shape[0]

        del x, t

    if train and (epoch == 1 or epoch % args.snapshot == 0):
        model_fn = '{}/epoch-{}.model'.format(args.result_dir, epoch)
        opt_fn = '{}/epoch-{}.state'.format(args.result_dir, epoch)
        serializers.save_hdf5(model_fn, model)
        serializers.save_hdf5(opt_fn, optimizer)

    if train:
        logging.info(
            'epoch:{}\ttrain loss:{}'.format(epoch, sum_loss / num))
    else:
        logging.info(
            'epoch:{}\tvalidate loss:{}'.format(epoch, sum_loss / num))

    mbatch_worker.join()


if __name__ == '__main__':
    args = create_args()
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
    xp = cuda.cupy if args.gpu >= 0 else np
    xp.random.seed(args.seed)

    # create result dir
    log_fn, args.result_dir = create_result_dir(args)

    # create model and optimizer
    model, optimizer = get_model_optimizer(args)

    # augmentation setting
    trans = Transform(args)

    # start logging
    logging.info('start training...')
    for epoch in range(1, args.epoch + 1):
        # learning rate reduction
        if args.opt == 'MomentumSGD' and epoch % args.lr_decay_freq == 0:
            optimizer.lr *= args.lr_decay_ratio
            logging.info('learning rate:', optimizer.lr)

        one_epoch(args, model, optimizer, epoch, True)
        one_epoch(args, model, optimizer, epoch, False)

        # draw curve
        draw_loss('{}/log.txt'.format(args.result_dir),
                  '{}/log.png'.format(args.result_dir))
