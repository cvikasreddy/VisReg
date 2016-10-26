import cPickle as pickle
import numpy as np
from numpy.random import permutation
import tarfile
import ntpath
import cPickle, gzip
import os
from six.moves import urllib
import sys
import time
import math

class CifarDataLoader(object):
    """
        The main part is get_cifar_data(), next_batch(), reset_index()
    """

    def __init__(self, batch_size=128):
        #self.data_dir = "../data"
        self.data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
        self.batch_size = batch_size

        self.get_cifar_data() #Loading the entire dataset

    def load_cifar_10_dataset(self):
        #print "Opening CIFAR 10 dataset"
        dataset = {}
        with tarfile.open(self.data_dir + "/cifar-10-python.tar.gz", "r:gz") as tar:
            for member in tar.getmembers():
                if member.isfile():
                    if "_batch" in member.name:
                        file_name = ntpath.basename(member.name)
                        f = tar.extractfile(member)
                        batch_dataset = cPickle.load(f) 
                        dataset[file_name] = batch_dataset
                    elif member.name.endswith("batches.meta"):
                        f = tar.extractfile(member)
                        label_names = cPickle.load(f) 
                        dataset["meta"] = label_names
        #print "Finished opening CIFAR 10 dataset"
        return dataset

    def merge_datasets(self, dataset_one, dataset_two):
        return {
            "data": np.concatenate((dataset_one["data"], dataset_two["data"])),
            "labels": dataset_one["labels"] + dataset_two["labels"], 
        }

    def get_merged_training_datasets(self, dataset_batches_dict):
        training_dataset_names = [ "data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4" ]
        training_datasets = map(lambda name: dataset_batches_dict[name], training_dataset_names)
        training_dataset_and_labels = reduce(self.merge_datasets, training_datasets)
        validation_dataset_and_labels = dataset_batches_dict["data_batch_5"]
        test_dataset_and_labels = dataset_batches_dict["test_batch"]
        return (
            np.asarray(training_dataset_and_labels["data"]), np.asarray(training_dataset_and_labels["labels"]),
            np.asarray(validation_dataset_and_labels["data"]), np.asarray(validation_dataset_and_labels["labels"]),
            np.asarray(test_dataset_and_labels["data"]), np.asarray(test_dataset_and_labels["labels"])
        )

    def reformat(self, dataset, labels):

        image_size = 32
        num_labels = 10
        num_channels = 3 # RGB

        dataset = dataset
        x = dataset.reshape((-1, num_channels, image_size * image_size)) # break the channels into their own axes.
        y = x.transpose([0, 2, 1]) # This transpose the matrix by swapping the second and third axes, but not the first. This puts matching RGB values together
        reformated_dataset = y.reshape((-1, image_size, image_size, num_channels)).astype(np.float32) # Turn the dataset into a 4D tensor of a collection of images, with axes of width, height and colour channels.
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return reformated_dataset, labels

    def get_cifar_data(self):
        dataset_batches_dict = self.load_cifar_10_dataset()
        self.label_names = dataset_batches_dict["meta"]["label_names"]
        self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels = self.get_merged_training_datasets(dataset_batches_dict)


        self.train_dataset, self.train_labels = self.reformat(self.train_dataset, self.train_labels)
        self.valid_dataset, self.valid_labels = self.reformat(self.valid_dataset, self.valid_labels)
        self.test_dataset, self.test_labels = self.reformat(self.test_dataset, self.test_labels)


        #Just making training set contain 512 images
        # perm_train = permutation(500)
        # self.train_dataset = self.train_dataset[perm_train]
        # self.train_labels = self.train_labels[perm_train]

        #Just making test set contain 512 images
        perm_test = permutation(1000)
        self.test_dataset = self.test_dataset[perm_test]
        self.test_labels = self.test_labels[perm_test]


        self.train_size = self.train_dataset.shape[0]

        self.reset_index()

        #return self.train_dataset, self.train_labels, self.valid_dataset, self.valid_labels, self.test_dataset, self.test_labels, self.label_names

    def next_batch(self, data_type='train'):
        if data_type == 'train':
            if self.train_index + self.batch_size > (self.train_dataset).shape[0]:
                self.train_index = 0
            batch_images = self.train_dataset[self.train_index : self.train_index + self.batch_size]
            batch_labels = self.train_labels[self.train_index : self.train_index + self.batch_size]
        
            self.train_index += self.batch_size

        if data_type == 'test':
            if self.test_index + self.batch_size > (self.test_dataset).shape[0]:
                self.test_index = 0
            batch_images = self.test_dataset[self.test_index : self.test_index + self.batch_size]
            batch_labels = self.test_labels[self.test_index : self.test_index + self.batch_size]
        
            self.test_index += self.batch_size

        if data_type == 'val':
            if self.valid_index + self.batch_size > (self.test_dataset).shape[0]:
                self.valid_index = 0
            batch_images = self.valid_dataset[self.valid_index : self.valid_index + self.batch_size]
            batch_labels = self.valid_labels[self.valid_index : self.valid_index + self.batch_size]
        
            self.valid_index += self.batch_size

        return {'images': batch_images, 'labels': batch_labels}

    def reset_index(self):
        #Set start index to 0 at start of every epoch
        self.train_index = 0
        self.test_index = 0
        self.valid_index = 0
        
        #Shuffle train data every epoch
        #self.train_size = self.train_dataset.shape[0]
        # perm_train = permutation(self.train_size)
        # self.train_dataset = self.train_dataset[perm_train]
        # self.train_labels = self.train_labels[perm_train]

        #Shuffle test data every epoch
        self.test_size = self.test_dataset.shape[0]
        perm_test = permutation(self.test_size)
        self.test_dataset = self.test_dataset[perm_test]
        self.test_labels = self.test_labels[perm_test]

        #Shuffle test data every epoch
        self.valid_size = self.valid_dataset.shape[0]
        perm_valid = permutation(self.valid_size)
        #perm_valid = permutation(1000)
        self.valid_dataset = self.valid_dataset[perm_valid]
        self.valid_labels = self.valid_labels[perm_valid]