import dtlpy as dl
import keras
import traceback
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import os
import itertools
V = '0.8.3'

# implementation base on https://keras.io/api/applications/

class ModelAdapter(dl.BaseModelAdapter):
    """
    Specific Model adapter.
    The class bind Dataloop model and snapshot entities with model code implementation
    """
    _defaults = {
        'weights_source': 'imagenet'
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        print("Initialized model_adapter {!r}. Version {}".format(self.model_name, V))

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: not used
        """

        input_shape = getattr(self, 'input_shape', None)
        self.model = ResNet50(weights=self.weights_source, input_shape=input_shape, include_top=True)
        # try:
        #     # https://stackoverflow.com/a/59238039/16076929
        #     self.model._make_predict_function()
        #     print("Keras workaround worked!")
        # except Exception as err:
        #     print("Keras workaround Failed...." + str(err))
        #     traceback.print_exc()
        # try:
        #     keras.backend.clear_session()
        #     print("Keras workaround 2 Worked!")
        # except Exception as err:
        #     print("Keras workaround 2 Failed...." + str(err))
        #     traceback.print_exc()

        msg = "ResNet50 Model loaded. Keras version {}".format(keras.__version__)
        self.logger.info(msg)
        print(msg)
        #keras_dir = os.environ.get('KERAS_HOME', '')
        keras_dir = os.path.expanduser('~/.keras')
        keras_models_dir = os.path.join(keras_dir, 'models')
        msg = "Load complete. Keras dir {} content: {}".format(keras_models_dir, os.listdir(keras_models_dir))
        self.logger.info(msg)
        print(msg)
        self.model.summary()

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        # no training hence no saving implemented
        raise NotImplementedError("Please implement 'save' method in {}".format(self.__class__.__name__))

    def train(self, local_path, dump_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        """
        # Currently no training for the resnet model
        raise NotImplementedError("Please implement 'train' method in {}".format(self.__class__.__name__))

    def predict(self, batch, reshape=True):
        """ Model inference (predictions) on batch of images

        :param batch: `np.ndarray`
        :param reshape: `bool` is True reshape the input image of the batch to single size, default: False
        :return: `List[dl.AnnotationCollection]`  prediction results by len(batch)
        """
        if reshape:
            from skimage.transform import resize
            batch_reshape = []
            for img in batch:
                batch_reshape.append(resize(img, output_shape=(224, 224)))
            # construct as batch
            batch = np.array(batch_reshape)

        x = preprocess_input(batch, mode='tf')
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

    def convert(self, local_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            Virtual method - need to implement

            e.g. take dlp dir structure and construct annotation file

        :param local_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """
        # Not implemented because we don't train the model
        raise NotImplementedError("Please implement 'convert' method in {}".format(self.__class__.__name__))


