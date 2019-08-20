import csv

from models.mapf_model import MapfModel

import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.utils import to_categorical
from sklearn import preprocessing

print("TensorFlow version is ", tf.__version__)
import numpy as np
from sklearn.metrics import accuracy_score
from src.metrics import coverage_score, cumsum_score
from tensorflow.keras.layers import (Conv1D, MaxPool1D, Dropout, Flatten, Dense, Lambda,
                                     Input, concatenate, GlobalAveragePooling2D, BatchNormalization)

from tensorflow.keras import backend as K
from sklearn.utils import class_weight

experiments_dir = '../edgelists/new_format/AllData/'


class CNNRegModel(MapfModel):

    def __init__(self, *args):
        super(CNNRegModel, self).__init__(*args)
        self.fname = 'mapf_dims_64_epochs_4_lr_0.1_embeddings - 4000.txt'
        self.label_fname = 'mapf.Labels'
        self.modelname = "Pretrained VGG16, GAP layer, features concat, Linear regression (5)"
        self.image_size = 224
        self.batch_size = 64
        self.IMG_SHAPE = (self.image_size, self.image_size, 3)
        self.base_model = tf.keras.applications.VGG16(input_shape=self.IMG_SHAPE,
                                                      include_top=False,
                                                      weights='imagenet')
        self.train_features_norm = []
        self.valid_features_norm = []

        self.base_model.trainable = True
        # Fine tune from this layer onwards
        fine_tune_at = 12
        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        self.trained = False
        self.multi_model = {}

        self.train_images = []
        self.train_features = []
        self.train_labels = []
        self.valid_images = []
        self.valid_features = []
        self.valid_labels = []

        self.class_weights = []

    @staticmethod
    def folder_from_label(label):
        return {
            'EPEA*+ID Runtime': 'epea',
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime': 'ma-cbs',
            'ICTS 3E +ID Runtime': 'icts',
            'A*+OD+ID Runtime': 'astar',
            'Basic-CBS/(A*/SIC)+ID Runtime': 'basic-cbs'
        }[label]

    @staticmethod
    def round_up_to_5(num):
        return str(5 - (int(num) % 5) + int(num))

    @staticmethod
    def obstacle_density_to_num(density):
        if density == '0.0':
            return 0
        density = density.split('.')[1]
        if len(density) > 2:
            return CNNRegModel.round_up_to_5(density[:2])
        elif density[0] == '0':
            return density[1:]
        elif len(density) == 1:
            return density + '0'

        return density

    @staticmethod
    def suffix_from_label(label):
        return {
            'EPEA*+ID Runtime': 0,
            'MA-CBS-Global-10/(EPEA*/SIC) choosing the first conflict in CBS nodes Runtime': 1,
            'ICTS 3E +ID Runtime': 2,
            'A*+OD+ID Runtime': 3,
            'Basic-CBS/(A*/SIC)+ID Runtime': 4
        }[label]

    @staticmethod
    def name_from_row(x):
        prefix = x.GridName if x.GridName != 'Unknown' else 'Instance'
        data = []
        if prefix == 'Instance':
            data = [str(d) for d in
                    [prefix, x.GridRows, CNNRegModel.obstacle_density_to_num(str(x.ObstacleDensity)),
                     x.NumOfAgents, x.InstanceId]]
        else:
            data = [str(d) for d in
                    [prefix, x.NumOfAgents, x.InstanceId]]

        data.append('label' + str(CNNRegModel.suffix_from_label(x['Y'])) + '.png')
        return '-'.join(data)

    @staticmethod
    def file_name_from(row):
        if row.GridName == 'Unknown':
            data = ['Instance', row.GridRows, CNNRegModel.obstacle_density_to_num(str(row.ObstacleDensity)),
                    row.NumOfAgents,
                    row.InstanceId]
            line = '-'.join(map(str, data)) + '.gexf'
        else:
            data = [row.GridName, str(row.NumOfAgents), str(row.InstanceId)]
            line = '-'.join(map(str, data)) + '.gexf'
        return line

    def embedding_for(self, file):
        if file in self.graph_embedding_dict:
            embedding = self.graph_embedding_dict[file]
        else:
            embedding = [0] * 64
        embedding = np.array(embedding)
        return embedding

    @staticmethod
    def load_image(path):
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        imgdata = np.array(img, dtype=np.float32)
        imgdata /= 225.0
        return imgdata

    def create_data_for(self, train=False):
        if train:
            df = self.X_train
        else:
            df = self.X_test
        images = []
        features = []
        labels = []
        for index, row in df.iterrows():
            images.append(CNNRegModel.load_image(row['img_path']))
            features.append(row[self.features_cols].values)
            labels.append(to_categorical(CNNRegModel.suffix_from_label(row['Y']), num_classes=5))

        images = np.array(images)
        features = np.array(features)

        labels = np.array(labels)

        return images, features, labels

    def prepare_data(self):
        self.X_train['img_path'] = self.X_train.apply(lambda x:
                                                      experiments_dir + CNNRegModel.folder_from_label(
                                                          x['Y'])
                                                      + "/" + CNNRegModel.name_from_row(x), axis=1)

        self.X_test['img_path'] = self.X_test.apply(
            lambda x: experiments_dir + CNNRegModel.folder_from_label(x['Y'])
                      + "/" + CNNRegModel.name_from_row(x), axis=1)

        self.X_train['mapf_file'] = self.X_train.apply(lambda x: CNNRegModel.file_name_from(x), axis=1)
        self.X_test['mapf_file'] = self.X_test.apply(lambda x: CNNRegModel.file_name_from(x), axis=1)

        # self.mapf_df = pd.read_csv(self.label_fname, delim_whitespace=True, header=None,
        #                            names=['mapf_file', 'label', 'embedding'], )

        # with open(self.fname, 'r') as fh:
        #     self.graph_embedding_dict = json.load(fh)
        #
        # self.graph_embedding_dict = dict(
        #     [(k.split('\\')[4].split('.g2v3')[0], v) for k, v in self.graph_embedding_dict.items()])
        #
        # self.mapf_train_df = self.mapf_df.merge(self.X_train, on=['mapf_file'], how='inner')
        # self.mapf_test_df = self.mapf_df.merge(self.X_test, on=['mapf_file'], how='inner')
        #
        # embedding_cols = ['embedding' + str(x) for x in range(1, 65)]
        # train_word2vec_df = pd.DataFrame([[k] + v for k, v in self.graph_embedding_dict.items()],
        #                                  columns=['mapf_file'] + embedding_cols)
        #
        # self.merged_train_df = self.mapf_train_df.merge(train_word2vec_df, on=['mapf_file'])
        # self.merged_test_df = self.mapf_test_df.merge(train_word2vec_df, on=['mapf_file'])
        #
        # self.merged_train_df['int_label'] = self.merged_train_df.apply(lambda x: self.conversions[x['label']], axis=1)
        # self.merged_test_df['int_label'] = self.merged_test_df.apply(lambda x: self.conversions[x['label']], axis=1)
        #
        # self.X_train = self.merged_train_df
        # self.X_test = self.merged_test_df

        self.train_images, self.train_features, self.train_labels = self.create_data_for(train=True)
        self.train_features_norm = preprocessing.normalize(self.train_features, axis=0, norm='max')

        log_alg_runtime_cols = ['log-' + col for col in self.alg_runtime_cols]

        self.X_train[log_alg_runtime_cols] = np.log1p(self.X_train[self.alg_runtime_cols])

        self.valid_images, self.valid_features, self.valid_labels = self.create_data_for(train=False)
        self.valid_features_norm = preprocessing.normalize(self.valid_features, norm='l2')

        self.X_test[log_alg_runtime_cols] = np.log1p(self.X_test[self.alg_runtime_cols])
        self.class_weights = class_weight.compute_class_weight(
            'balanced',
            np.unique(np.argmax(self.train_labels, axis=1)),
            np.argmax(self.train_labels, axis=1))

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def build_model(self):
        image = tf.keras.Input(shape=self.IMG_SHAPE, name='image')
        features = tf.keras.Input(shape=(12,), name='features')
        # binary_clfs = tf.keras.Input(shape=(5,), name='bi_features')
        # graph_embedding = tf.keras.Input(shape=(64,), name='g_embed')

        last = self.base_model.output
        x = GlobalAveragePooling2D()(last)
        # x = concatenate([x, graph_embedding])
        # x = Flatten()(last)
        # x = Dense(256, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(64, activation='relu')(x)
        # x = Dropout(0.5)(x)
        # x = BatchNormalization()(x)
        merged = concatenate([x, features], name="Concat")
        # merged = Dense(64, activation='relu')(merged)
        # merged = concatenate([merged, binary_clfs])
        # merged = concatenate([merged, graph_embedding])

        # merged = BatchNormalization()(merged)
        # preds = Dense(5, activation='softmax')(merged)
        preds = Dense(5, activation='linear')(merged)
        # out1 = Lambda(lambda x: x[..., :1])(preds)
        # out2 = Lambda(lambda x: x[..., 1:2])(preds)
        # out3 = Lambda(lambda x: x[..., 2:3])(preds)
        # out4 = Lambda(lambda x: x[..., 3:4])(preds)
        # out5 = Lambda(lambda x: x[..., 4:5])(preds)

        self.multi_model = tf.keras.Model(inputs=[self.base_model.inputs, features], outputs=preds)

        self.multi_model.compile(loss=["mean_squared_error"],
                                 optimizer=tf.keras.optimizers.Adam(lr=0.0001, decay=0.0001 / 200),
                                 metrics=['mean_squared_logarithmic_error', CNNRegModel.root_mean_squared_error])

        print(self.multi_model.summary())

    @staticmethod
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    @staticmethod
    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    @staticmethod
    def f1_m(y_true, y_pred):
        precision = CNNRegModel.precision_m(y_true, y_pred)
        recall = CNNRegModel.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def load_weights(self, weigths='models/pretrained/vgg16_finetune12_gap_concat_linear.weights'):
        self.multi_model.load_weights(weigths)
        self.trained = True

    def train(self):
        epochs = 30
        save_best = keras.callbacks.ModelCheckpoint('best.weights', monitor='val_loss', verbose=1, save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        # In order to add binary classifier outputs:
        # train_features_with_bi = np.concatenate((train_features_norm, bi_clf_train_res.transpose()[0]),axis=1)

        # valid_features_norm = std_scale.transform(valid_features)
        # valid_features_with_bi = np.concatenate((valid_features_norm, bi_clf_test_res.transpose()[0]),axis=1)

        history = self.multi_model.fit([self.train_images, self.train_features_norm]
                                       , self.X_train[self.only_alg_runtime_cols],
                                       epochs=epochs,
                                       workers=1,
                                       sample_weight=[self.train_samples_weight],
                                       class_weight=self.class_weights,
                                       validation_data=(
                                           [self.valid_images, self.valid_features_norm],
                                           self.X_test[self.only_alg_runtime_cols]),
                                       # validation_data=validation_generator,
                                       # validation_steps=validation_steps,
                                       callbacks=[save_best, early_stop])

        self.trained = True

    def print_results(self, results_file='xgbmodel-results.csv'):
        if not self.trained:
            print("ERROR! Can't print model results before training")
            return
        preds = self.multi_model.predict((self.valid_images, self.valid_features_norm)).argmin(axis=1)
        preds = [self.conversions[p] for p in preds]
        model_acc =  accuracy_score(self.y_test, preds)
        model_coverage = coverage_score(self.X_test, preds)
        model_cumsum = cumsum_score(self.X_test, preds)

        with open(results_file, 'a+', newline='') as csvfile:
            fieldnames = ['Model', 'Accuracy', 'Coverage', 'Cumsum(minutes)', 'Notes']
            res_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            res_writer.writerow({'Model': self.modelname,
                                 'Accuracy': "{0:.2%}".format(model_acc),
                                 'Coverage': "{0:.2%}".format(model_coverage),
                                 'Cumsum(minutes)': int(model_cumsum),
                                 'Notes': 'Fine tune vgg from layer 12 '})
