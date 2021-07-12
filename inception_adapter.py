import dtlpy as dl
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras import Model
import numpy as np
import itertools


class ModelAdapter(dl.BaseModelAdapter):
    """
    Inception V3 adapter
    implementation base on https://keras.io/api/applications/
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    _defaults = {
        'weights_source': 'imagenet',
        'model_fname': 'my_ception.h5',
        'input_shape': (229,229),
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)
        self.graph = None

    # ===============================
    # NEED TO IMPLEMENT THESE METHODS
    # ===============================

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: if None uses default of kears.application
        """
        use_pretrained = getattr(self, 'use_pretrained', False)
        if local_path is None or use_pretrained :
            input_shape = getattr(self, 'input_shape', None)
            include_top = getattr(self, 'include_top', True)
            if include_top:
                self.model = InceptionV3(weights=self.weights_source, input_shape=input_shape, include_top=include_top)
                self.logger.info("Loaded pretrained InceptionV3 model ({})".format(self.model.name))
            else:  # we build a new model
                # TODO: change to resnet shapes...
                # create the base pre-trained model
                base_model = InceptionV3(weights=self.weights_source, input_shape=input_shape, include_top=False)

                # add a global spatial average pooling layer
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                # let's add a fully-connected layer
                x = Dense(1024, activation='relu')(x)
                # and a logistic layer -- number of classes should be an attribute
                predictions = Dense(self.nof_classes, activation='softmax')(x)

                # this is the model we will train
                self.model = Model(inputs=base_model.input, outputs=predictions)

                # first: train only the top layers (which were randomly initialized)
                # i.e. freeze all convolutional InceptionV3 layers
                for layer in base_model.layers:
                    layer.trainable = False

                # compile the model (should be done *after* setting layers to non-trainable)
                self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
                self.logger.info("Created new trainalbe InceptionV3 model with {} classes. ({})".
                                 format(self.nof_classes, self.model.name))
        else:
            model_path = "{d}/{f}.h5".format(d=local_path,f=self.model_name)
            self.model = keras.models.load_model(model_path)
            self.logger.info("Loaded model from {} succesfully".format(model_path))

        self.graph = tf.get_default_graph()

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        # See https://keras.io/guides/serialization_and_saving/  - which best method to save  (they recomand without h5 file)
        # self.model.save(local_path="{}.{}".format(local_path, self.model_fname))
        self.model.save("{d}/{f}.h5".format(d=local_path,f=self.model_name))

    def train(self, local_path, dump_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        """
        from PIL import Image

        # TODO: create the new dataset and test it
        with open("{}/ann.txt".format(local_path), 'r') as f:
            anns = f.readlines()

        X, Y = [], []
        for ann in anns:
            img_path, label = ann.strip().split(' - ')
            X.append(np.array(Image.open(img_path).resize(self.input_shape)))
            Y.append(int(label))
        X = np.array(X)
        Y = keras.utils.to_categorical(Y)  # Convert to matrix - due to sparce categorical cross entropy loss
        self.model.fit(X, Y, epochs=5, callbacks=None)  # validation_data=...
        self.logger.info("Training completed")

    def predict(self, batch, reshape=False, verbose=True):
        """ Model inference (predictions) on batch of images

        :param batch: `np.ndarray`
        :return: `List[list[self.ClassPrediction]]`  prediction results by len(batch)
        """
        if reshape:
            # self._np_resize_util(batch, output_shape=self.input_shape)
            from skimage.transform import resize
            batch_reshape = []
            for img in batch:
                batch_reshape.append(resize (img, output_shape=self.input_shape))
            # construct as batch
            batch = np.array(batch_reshape)

        with self.graph.as_default():
            x = preprocess_input(batch)
            preds = self.model.predict(x)

        batch_predictions = []
        for pred in decode_predictions(preds):
            # pred is a list (by scores) of tuples (idx, label_name, score)
            pred_label, pred_score = pred[0][1:]   # 0 - state the first (highest) predictions
            item_pred = dl.ml.predictions_utils.add_classification(
                label=pred_label,
                score=pred_score,
                adapter=self,
                collection=None
            )
            self.logger.debug("Predicted {:20} ({:1.3f})".format(pred_label, pred_score))
            batch_predictions.appedns(item_pred)
        return batch_predictions

    def convert(self, local_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            e.g. take dlp dir structure and construct annotation file

        :param local_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """
        from glob import glob
        import json
        ann_path = local_path + '/ann.txt'
        with open(ann_path, 'w') as f:
            for ann_json in glob("{}/json/*.json".format(local_path)):
                dlp_ann = json.load(open(ann_json, 'r'))
                img_path = local_path + '/items/' + dlp_ann['filename']
                f.write("{p} - {l}\n".format(p=img_path,
                                             l=self.snapshot.label_map.get(dlp_ann['annotations'][0]['label'], -1 )
                                             ))
        self.logger.info("created annotaion file : {}".format(ann_path))
        return ann_path

    def convert_dlp(self, items):
        """ This should implement similar to convert only to work on dlp items.  -> meaning create the converted version from items entities"""
        # TODO
        pass


