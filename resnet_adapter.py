import dtlpy as dl
import tensorflow as tf
from  tensorflow import keras
import traceback
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
# from tensorflow.keras.backend import tf
import numpy as np
import json
import os
import itertools
from skimage.transform import resize

# implementation base on https://keras.io/api/applications/

class ModelAdapter(dl.BaseModelAdapter):
    """
    Specific Model adapter.
    The class bind Dataloop model and snapshot entities with model code implementation
    """
    configuration = {
        'weights_filename': 'model.h5',
        'classes_filename': 'classes.json',
        'input_shape': (224, 224, 3)
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self.graph = None
        print("Initialized model_adapter {!r}.".format(self.model_name))

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================
    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: not used
        """
        classes_filename = self.configuration['classes_filename']
        weights_filename = self.configuration['weights_filename']
        # load classes
        with open(os.path.join(local_path, classes_filename)) as f:
            self.label_map = json.load(f)

        # self.sess = tf.Session()
        # self.graph = tf.get_default_graph()
        # tf.keras.backend.set_session(self.sess)
        model_path = os.path.join(local_path, weights_filename)
        self.model = keras.models.load_model(model_path)
        self.logger.info("Loaded model from {} successfully".format(model_path))
        self.model.summary()

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally
            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity
        :param local_path: `str` directory path in local FileSystem
        """
        # no training hence no saving implemented
        weights_filename = kwargs.get('weights_filename', 'model.hdf5')
        classes_filename = kwargs.get('classes_filename', 'classes.json')

        self.model.save(os.path.join(local_path, weights_filename))
        with open(os.path.join(local_path, classes_filename), 'w') as f:
            json.dump(self.label_map, f)
        self.snapshot.configuration['weights_filename'] = weights_filename
        self.snapshot.configuration['classes_filename'] = classes_filename
        self.snapshot.configuration['label_map'] = self.label_map
        self.snapshot.update()

    def train(self, local_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        """
        # Currently no training for the resnet model
        raise NotImplementedError("Please implement 'train' method in {}".format(self.__class__.__name__))

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

        :param batch: `np.ndarray`
        :param reshape: `bool` is True reshape the input image of the batch to single size, default: False
        :return: `List[dl.AnnotationCollection]`  prediction results by len(batch)
        """
        out_shape_wh = self.configuration['input_shape'][:2]
        batch_reshape = []
        for img in batch:
            batch_reshape.append(resize(img, output_shape=out_shape_wh))
            # batch_reshape.append(self._predict_preprocess(img, output_wh=out_shape))
        # construct as batch
        batch = np.array(batch_reshape)

        # with self.graph.as_default():
        x = preprocess_input(batch)   # , mode='tf')
        preds = self.model.predict(x)

        # pred is a list (by scores) of tuples (idx, label_name, score)
        batch_predictions = []
        for pred in decode_predictions(preds):
            pred_label, pred_score = pred[0][1:3]   # 0 - state the first (highest) predictions
            item_pred = dl.ml.predictions_utils.add_classification(
                label=pred_label,
                score=pred_score,
                adapter=self,
                collection=None
            )
            batch_predictions.append(item_pred)
            self.logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))

        return batch_predictions

    def convert_from_dtlpy(self, local_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param local_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """
        # Not implemented because we don't train the model
        raise NotImplementedError("Please implement 'convert' method in {}".format(self.__class__.__name__))

    def _predict_preprocess(self, img, output_wh):
        img = resize(img, output_shape=output_wh)
        img /= 255
        img *= 2
        img -= 1
        return x


def model_and_snapshot_creation(env='prod'):
    dl.setenv(env)
    project = dl.projects.get('DataloopModels')
    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/keras_adapters.git',
                              git_tag='main')
    model = project.models.create(model_name='ResNet-Keras',
                                  description='Global Dataloop ResNet implemented in keras',
                                  output_type=dl.AnnotationType.CLASSIFICATION,
                                  is_global=True,
                                  tags=['keras', 'classification'],
                                  entry_point='resnet_adapter.py',
                                  # class_name='ModelAdapter',
                                  codebase=codebase)

    bucket = dl.GCSBucket(gcs_project_name='viewo-main',
                          gcs_bucket_name='model-mgmt-snapshots',
                          gcs_prefix='ResNet50_keras')
    snapshot = model.snapshots.create(snapshot_name='pretrained-resnet',
                                      description='inception pretrrained using imagenet',
                                      tags=['pretrained', 'imagenet'],
                                      dataset_id=None,
                                      is_global=True,
                                      # status='trained',
                                      configuration={'weights_filename': 'resnet50.h5',
                                                     'classes_filename': 'classes.json',
                                                     'input_shape': (224, 224, 3)
                                                     },
                                      project_id=project.id,
                                      bucket=bucket,
                                      # TODO: add the laabel - best as an dl.ml utility
                                      labels=json.load(
                                          open(os.path.join(os.path.dirname(__file__), 'imagenet_labels_list.json'))
                                      )
                                      )
