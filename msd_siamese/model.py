import os 
import sys
import numpy as np
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate, LeakyReLU, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
sys.path.append('../')
import msd_config as config


def hinge_loss(y_true, y_pred):
    y_pos = y_pred[:, :config.num_pos_tracks]
    y_neg = y_pred[:, config.num_pos_tracks:]
    total_loss = 0.0
    for i in range(config.num_pos_tracks):
        loss = K.sum(K.maximum(0., config.margin - y_pos[:, i:i+1] + y_neg))
        total_loss += loss
    return total_loss


def siamese_cnn(num_frame, num_neg_singers, num_pos_track):
    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_track)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_singers)]

    # audio model 
    conv1 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn1 = BatchNormalization()
    activ1 = LeakyReLU(0.2)
    # activ1 = Activation('relu')
    mp1 = MaxPool1D(pool_size=3)

    conv2 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn2 = BatchNormalization()
    activ2 = LeakyReLU(0.2)
    # activ2 = Activation('relu')
    mp2 = MaxPool1D(pool_size=3)
    do2 = Dropout(0.5)
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization()
    activ3 = LeakyReLU(0.2)
    # activ3 = Activation('relu')
    mp3 = MaxPool1D(pool_size=3)
    do3 = Dropout(0.5)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization()
    activ4 = LeakyReLU(0.2)
    # activ4 = Activation('relu')
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization()
    activ5 = LeakyReLU(0.2)
    # activ5 = Activation('relu')
    do5 = Dropout(0.5)

    ap = GlobalAvgPool1D()

    classification_dense = Dense(config.num_singers, activation='softmax', name='classification')
    
    # euc_dist = Lambda(euclidean_dist, euclidean_dist_output_shape)
    # negative_sampling = Lambda(neg_sample, neg_sample_output_shape)

    # l2_dist = Lambda(lambda  x: K.l2_normalize(x[0] - x[1],axis=1))

    # Anchor 
    anchor_out = mp1(activ1(bn1(conv1(anchor))))
    anchor_out = mp2(activ2(bn2(conv2(anchor_out))))
    anchor_out = mp3(activ3(bn3(conv3(anchor_out))))
    anchor_out = do3(anchor_out)
    anchor_out = mp4(activ4(bn4(conv4(anchor_out))))
    anchor_out = activ5(bn5(conv5(anchor_out)))
    anchor_out = do5(anchor_out)
    anchor_out = ap(anchor_out)
    classification_out = classification_dense(anchor_out)

    # Pos 
    pos_outs = [mp1(activ1(bn1(conv1(pos_item)))) for pos_item in pos_items]
    pos_outs = [mp2(activ2(bn2(conv2(pos_out)))) for pos_out in pos_outs]
    pos_outs = [mp3(activ3(bn3(conv3(pos_out)))) for pos_out in pos_outs]
    pos_outs = [do3(pos_out) for pos_out in pos_outs]
    pos_outs = [mp4(activ4(bn4(conv4(pos_out)))) for pos_out in pos_outs]
    pos_outs = [activ5(bn5(conv5(pos_out))) for pos_out in pos_outs]
    pos_outs = [do5(pos_out) for pos_out in pos_outs]
    pos_outs = [ap(pos_out) for pos_out in pos_outs]


    # Negs
    neg_outs = [mp1(activ1(bn1(conv1(neg_item)))) for neg_item in neg_items]
    neg_outs = [mp2(activ2(bn2(conv2(neg_out)))) for neg_out in neg_outs]
    neg_outs = [mp3(activ3(bn3(conv3(neg_out)))) for neg_out in neg_outs]
    neg_outs = [do3(neg_out) for neg_out in neg_outs]
    neg_outs = [mp4(activ4(bn4(conv4(neg_out)))) for neg_out in neg_outs]
    neg_outs = [activ5(bn5(conv5(neg_out))) for neg_out in neg_outs]
    neg_outs = [do5(neg_out) for neg_out in neg_outs]
    neg_outs = [ap(neg_out) for neg_out in neg_outs]


    #### cosine  
    pos_dists = [dot([anchor_out, pos_out], axes=1, normalize=True) for pos_out in pos_outs]
    neg_dists = [dot([anchor_out, neg_out], axes=1, normalize=True) for neg_out in neg_outs]
    # pos_dists = [l2_dist([anchor_out, pos_out]) for pos_out in pos_outs]
    # neg_dists = [l2_dist([anchor_out, neg_out]) for neg_out in neg_outs]
    
    all_dists = concatenate(pos_dists + neg_dists)
    # all_dists = negative_sampling(all_dists)
    outputs = Activation('linear', name='distance')(all_dists)

    ### euclidean 
    '''
    distance  = Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape)
    pos_dists = [distance([anchor_out, pos_out]) for pos_out in pos_outs]
    neg_dists = [distance([anchor_out, neg_out]) for neg_out in neg_outs]

    all_dists = concatenate(pos_dists + neg_dists)

    outputs = Activation('softmax', name='distance')(all_dists)
    '''

    # model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=[classification_out, outputs])
    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)
    return model 




def finetuning_siamese_cnn(mymodel_tmp, num_frame, num_neg_singers, num_pos_tracks):
    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_tracks)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_singers)]

    dense = Dense(256)
    ap = GlobalAvgPool1D()

    anchor_out = mymodel_tmp(anchor)
    pos_outs = [mymodel_tmp(pos_item) for pos_item in pos_items]
    neg_outs = [mymodel_tmp(neg_item) for neg_item in neg_items]


    ### cosine 
    pos_dists = [dot([anchor_out, pos_out], axes=1, normalize=True) for pos_out in pos_outs]
    neg_dists = [dot([anchor_out, neg_out], axes=1, normalize=True) for neg_out in neg_outs]
    
    all_dists = concatenate(pos_dists + neg_dists)

    outputs = Activation('linear')(all_dists)

    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)

    return model 







if __name__ == '__main__':
    model = siamese_cnn(129, 4)
    print (model.summary())
