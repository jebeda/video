#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 17:15:34 2020

@author: ocrusr
"""
# vim: expandtab:ts=4:sw=4
import functools
import os
import numpy as np
import scipy.io as sio
import train_app as train_app_MTMC
#import train_app_MTMC
#from datasets import market1501
from datasets import MTMC
from datasets import util
import nets.deep_sort.network_definition as net


class DukeMTMC(object):

    def __init__(self, dataset_dir, num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed

    def read_train(self):
        filenames, ids, camera_indices = MTMC.read_train_split_to_str(
            self._dataset_dir)
        #print(filenames,"############read_train in clssDUKE")#loading filenames confirmed
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        camera_indices = [camera_indices[i] for i in train_indices]
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, camera_indices = MTMC.read_train_split_to_str(
            self._dataset_dir)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        camera_indices = [camera_indices[i] for i in valid_indices]
        return filenames, ids, camera_indices

    def read_test(self):
        return MTMC.read_test_split_to_str(self._dataset_dir)


def main():
    arg_parser = train_app_MTMC.create_default_argument_parser("DukeMTMC-reID")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to DukeMTMC dataset directory.",
        default="./DukeMTMC-reID/DukeMTMC-reID")
    #print("######################arg_parser",arg_parser)
    arg_parser.add_argument(
        "--sdk_dir", help="Path to DukeMTMC baseline evaluation software.",
        default="resources/Market-1501-v15.09.15-baseline")
    args = arg_parser.parse_args()
    dataset = DukeMTMC(args.dataset_dir, num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d identities" % (
            len(train_x), len(np.unique(train_y))))
        print("##############################",train_x[0])

        network_factory = net.create_network_factory(
            is_training=True, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app_MTMC.to_train_kwargs(args)
        train_app_MTMC.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=MTMC.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d identities" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app_MTMC.to_eval_kwargs(args)
        train_app_MTMC.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=MTMC.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        # Export one specific model.
        gallery_filenames, _, query_filenames, _, _ = dataset.read_test()

        network_factory = net.create_network_factory(
            is_training=False, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        gallery_features = train_app_MTMC.encode(
            net.preprocess, network_factory, args.restore_path,
            gallery_filenames, image_shape=MTMC.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_test.mat"),
            {"features": gallery_features})

        network_factory = net.create_network_factory(
            is_training=False, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=False, reuse=True)
        query_features = train_app_MTMC.encode(
            net.preprocess, network_factory, args.restore_path,
            query_filenames, image_shape=MTMC.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_query.mat"),
            {"features": query_features})
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app_MTMC.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=MTMC.IMAGE_SHAPE,
            output_filename="./MTMC.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=MTMC.MAX_LABEL + 1,
            add_logits=False, reuse=None)
        train_app_MTMC.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=MTMC.IMAGE_SHAPE,
            output_filename="./MTMC.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()
