import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import numpy as np
import json
import os
from skimage.transform import resize

import dtlpy as dl
from dtlpy.utilities.dataset_generators.dataset_generator import collate_tf
from dtlpy.utilities.dataset_generators.dataset_generator_tensorflow import DatasetGeneratorTensorflow


class ModelAdapter(dl.BaseModelAdapter):
    """
    Inception V3 adapter
    implementation base on https://keras.io/api/applications/
    The class bind Dataloop model and snapshot entities with model code implementation
    """

    configuration = {
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
        weights_filename = self.snapshot.configuration.get('weights_filename', 'model.hdf5')
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
        self.model.save(os.path.join(local_path, weights_filename))

    def train(self, data_path, output_path, **kwargs):
        """ Train the model according to data in local_path and save the snapshot to dump_path

            Virtual method - need to implement
        """
        num_epochs = self.configuration.get('num_epochs', 10)
        batch_size = self.configuration.get('batch_size', 64)
        input_size = self.configuration.get('input_size', (299, 299))

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
        train_dataset = DatasetGeneratorTensorflow(data_path=os.path.join(data_path, 'train'),
                                                   dataset_entity=self.snapshot.dataset,
                                                   annotation_type=dl.AnnotationType.CLASSIFICATION,
                                                   transforms=transforms,
                                                   batch_size=batch_size,
                                                   to_categorical=True,
                                                   collate_fn=collate_tf)
        val_dataset = DatasetGeneratorTensorflow(data_path=os.path.join(data_path, 'validation'),
                                                 dataset_entity=self.snapshot.dataset,
                                                 annotation_type=dl.AnnotationType.CLASSIFICATION,
                                                 batch_size=batch_size,
                                                 transforms=transforms,
                                                 to_categorical=True,
                                                 collate_fn=collate_tf)

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
        configuration = self.configuration
        input_size = configuration.get('input_size', (299, 299))

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
            # label = self.snapshot.id_to_label_map[str(np.argmax(pred))]
            label = np.argmax(pred)
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
    project = dl.projects.get('My Project')
    model = project.models.get('InceptionV3')
    snapshot = model.snapshots.get('sheep-soft-augmentations')
    model.snapshots.list().to_df()
    adapter = model.build()
    adapter.load_from_snapshot(snapshot=snapshot,
                               local_path=snapshot.bucket.local_path)
    root_path, data_path, output_path = adapter.prepare_training(root_path=os.path.join('tmp', snapshot.id))
    # Start the Train
    print("Training {!r} with snapshot {!r} on data {!r}".format(model.name, snapshot.id, data_path))
    print("Starting train with data at {}".format(data_path))
    adapter.train(data_path=data_path,
                  output_path=output_path)


def model_and_snapshot_creation():
    project = dl.projects.get('DataloopModels')
    codebase = dl.GitCodebase(git_url='https://github.com/dataloop-ai/keras_adapters.git',
                              git_tag='main')
    model = project.models.create(model_name='InceptionV3',
                                  description='Global Dataloop inception V3 implemented in keras',
                                  output_type=dl.AnnotationType.CLASSIFICATION,
                                  is_global=True,
                                  tags=['keras', 'classification'],
                                  entry_point='inception_adapter.py',
                                  # class_name='ModelAdapter',
                                  codebase=codebase)

    bucket = dl.GCSBucket(gcs_project_name='viewo-main',
                          gcs_bucket_name='model-mgmt-snapshots',
                          gcs_prefix='InceptionV3')
    snapshot = model.snapshots.create(snapshot_name='pretrained-inception',
                                      description='inception pretrrained using imagenet',
                                      tags=['pretrained', 'imagenet'],
                                      dataset_id=None,
                                      is_global=True,
                                      # status='trained',
                                      configuration={'weights_filename': 'model.hdf5',
                                                     'classes_filename': 'classes.json'},
                                      project_id=project.id,
                                      bucket=bucket,
                                      # TODO: add the laabel - best as an dl.ml utility
                                      labels=json.load(
                                          open(os.path.join(os.path.dirname(__file__), 'imagenet_labels_list.json'))
                                      )
                                      )

#
# if __name__ == "__main__":
#     train()
