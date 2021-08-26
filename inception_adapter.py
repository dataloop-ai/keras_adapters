import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import json
import os
from skimage.transform import resize

import dtlpy as dl
from dtlpy.ml import train_utils
from dtlpy.ml.ml_dataset import get_keras_dataset


class ModelAdapter(dl.BaseModelAdapter):
    """
    Inception V3 adapter
    implementation base on https://keras.io/api/applications/
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    configurations = {
        'weights_filename': 'model.hdf5',
        'classes_filename': 'classes.json',
        'input_shape': (299, 299)
    }

    def __init__(self, model_entity):
        super(ModelAdapter, self).__init__(model_entity)

    def load(self, local_path, **kwargs):
        """ Loads model and populates self.model with a `runnable` model

            This function is called by load_from_snapshot (download to local and then loads)

        :param local_path: if None uses default of kears.application
        """
        classes_filename = self.snapshot.configuration.get('classes_filename', 'classes.json')
        weights_filename = self.snapshot.configuration.get('weights_filename', 'model.hdf5')
        # load classes
        with open(os.path.join(local_path, classes_filename)) as f:
            self.label_map = json.load(f)

        # self.sess = tf.Session()
        # self.graph = tf.get_default_graph()
        # tf.keras.backend.set_session(self.sess)
        model_path = os.path.join(local_path, weights_filename)
        self.model = keras.models.load_model(os.path.join(local_path, weights_filename))
        self.logger.info("Loaded model from {} successfully".format(model_path))
        self.model.summary()

    def save(self, local_path, **kwargs):
        """ saves configuration and weights locally

            Virtual method - need to implement

            the function is called in save_to_snapshot which first save locally and then uploads to snapshot entity

        :param local_path: `str` directory path in local FileSystem
        """
        # See https://keras.io/guides/serialization_and_saving/  - which best method to save  (they recomand without h5 file)
        # self.model.save(local_path="{}.{}".format(local_path, self.model_fname))
        weights_filename = kwargs.get('weights_filename', 'model.hdf5')
        classes_filename = kwargs.get('classes_filename', 'classes.json')

        self.model.save(os.path.join(local_path, weights_filename))
        with open(os.path.join(local_path, classes_filename), 'w') as f:
            json.dump(self.label_map, f)
        self.snapshot.configuration['weights_filename'] = weights_filename
        self.snapshot.configuration['classes_filename'] = classes_filename
        self.snapshot.configuration['label_map'] = self.label_map
        self.snapshot.update()

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        """
        config = self.configurations
        config.update(self.snapshot.configuration)
        num_epochs = config.get('num_epochs', 10)
        batch_size = config.get('batch_size', 64)
        input_size = config.get('input_size', (299, 299))

        ####################
        # Prepare the data #
        ####################
        def preprocess(x):
            x = resize(x, output_shape=input_size)
            x /= 255
            x *= 2
            x -= 1
            return x

        transforms = [
            preprocess
        ]
        train_dataset = get_keras_dataset()(data_path=os.path.join(data_path, 'train'),
                                            dataset_entity=self.snapshot.dataset,
                                            annotation_type=dl.AnnotationType.CLASSIFICATION,
                                            transforms=transforms,
                                            batch_size=batch_size,
                                            to_categorical=True)
        val_dataset = get_keras_dataset()(data_path=os.path.join(data_path, 'validation'),
                                          dataset_entity=self.snapshot.dataset,
                                          annotation_type=dl.AnnotationType.CLASSIFICATION,
                                          batch_size=batch_size,
                                          transforms=transforms,
                                          to_categorical=True)

        # replace head with new number of calsses
        output = self.model.layers[-2].output
        pred = Dense(train_dataset.num_classes, activation='softmax')(output)
        self.model = Model(inputs=self.model.input, outputs=pred)

        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy')
        self.model.fit(train_dataset,
                       validation_data=val_dataset,
                       epochs=num_epochs)
        self.logger.info("Training completed")

    def predict(self, batch, **kwargs):
        """ Model inference (predictions) on batch of images

        :param batch: `np.ndarray`
        :return: `List[dl.AnnotationCollection]`  prediction results by len(batch)
        """
        config = self.configurations
        config.update(self.snapshot.configuration)
        input_size = config.get('input_size', (299, 299))

        def preprocess(x):
            x = resize(x, output_shape=input_size)
            x /= 255
            x *= 2
            x -= 1
            return x

        batch_reshape = list()
        for img in batch:
            batch_reshape.append(preprocess(img))
        batch = np.array(batch_reshape)
        preds = self.model.predict(batch)
        batch_collection = list()
        for pred in preds:
            label = self.label_map[str(np.argmax(pred))]
            score = np.max(pred)
            annotations = dl.AnnotationCollection()
            annotations.add(annotation_definition=dl.Classification(label=label),
                            model_info={
                                'name': self.model_entity.name,
                                'confidence': float(score),
                                'modelId': self.model_entity.id,
                                'snapshotId': self.snapshot.id
                            })
            batch_collection.append(annotations)
        return batch_collection

    def convert_from_dtlpy(self, data_path, **kwargs):
        """ Convert Dataloop structure data to model structured

            e.g. take dlp dir structure and construct annotation file

        :param local_path: `str` local File System directory path where we already downloaded the data from dataloop platform
        :return:
        """
        ...


def train():
    dl.setenv('prod')
    project = dl.projects.get('COCO ors', '729659ec-6d7f-11e8-8d00-42010a8a002b')
    model = project.models.get('inceptionv3')
    snapshot = model.snapshots.get('sheep-soft-augmentations')
    model.snapshots.list().to_df()
    self = model.build()
    self.load_from_snapshot(snapshot=snapshot,
                            local_path=snapshot.bucket.local_path)
    root_path, data_path, output_path = train_utils.prepare_training(snapshot=snapshot,
                                                                     adapter=self,
                                                                     root_path=os.path.join('tmp', snapshot.id))
    # Start the Train
    print("Training {!r} with snapshot {!r} on data {!r}".format(model.name, snapshot.id, data_path))
    print("Starting train with data at {}".format(data_path))

    self.snapshot.configuration = {'batch_size': 16,
                                   'start_epoch': 0,
                                   'num_epochs': 2,
                                   'input_size': 256}


def global_creation():
    project = dl.projects.get('COCO ors')
    # codebase = dl.create(
    #     src_path=r'E:\ModelsZoo\pytorch_adapters-master',
    #     entry_point='resnet_adapter.py'
    # )
    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/keras_adapters.git',
                              git_tag='master')
    model = project.models.create(model_name='InceptionV3',
                                  output_type=dl.AnnotationType.CLASSIFICATION,
                                  is_global=True,
                                  entry_point='inception_adapter.py',
                                  class_name='ModelAdapter',
                                  codebase=codebase)

    bucket = dl.LocalBucket(local_path=r'E:\ModelsZoo\YOLOX-main\YOLOX_outputs\yolox_l')
    bucket = dl.GCSBucket(gcs_project_name='viewo-main',
                          gcs_bucket_name='model-mgmt-snapshots',
                          gcs_prefix='InceptionV3')
    snapshot = model.snapshots.create(snapshot_name='imagenet-pretrained',
                                      description='COCO pretrained model',
                                      dataset_id=None,
                                      configuration={'weights_filename': 'model.hdf5',
                                                     'classes_filename': 'classes.json'},
                                      project_id=project.id,
                                      bucket=bucket,
                                      labels=[])

#
# if __name__ == "__main__":
#     train()
