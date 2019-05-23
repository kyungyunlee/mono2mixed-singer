import os 
import sys
import numpy as np
import tensorflow as tf 
import keras.backend as K
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, GlobalAvgPool1D, Dense, Dropout, Activation, Reshape, Input, Concatenate, dot, Add, Flatten, concatenate, LeakyReLU, Lambda, merge
from keras.layers import Conv2D, MaxPool2D, GlobalAvgPool2D
from keras.models import Model
from keras.regularizers import l2

import config 


def euclidean_dist(tensors):
    x,y = tensors
    print (x.shape)
    sum_square = K.sum(K.square(x-y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square,K.epsilon()))
    # return sum_square



def euclidean_dist_output_shape(input_shapes):
    shapes1, shapes2 = input_shapes
    return (shapes1[0],1)


def neg_sample(tensors):
    pos_tensor = tensors[:, :1]
    neg_tensors = tensors[:, 1:]
    print (neg_tensors)
    out, _ = tf.nn.top_k(-neg_tensors, k=4)
    out = -out 
    print (out)
    all_out = K.concatenate([pos_tensor, out])
    print (all_out)
    return all_out 


def neg_sample_output_shape(input_shapes):

    return (config.batch_size, config.num_neg_artist + 1)



def vocal_cnn_block(num_frame):
    anchor = Input(shape=(num_frame, 128))
    
    # audio model 
    conv1 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn1 = BatchNormalization()
    activ1 = LeakyReLU(0.2)
    mp1 = MaxPool1D(pool_size=3)

    conv2 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn2 = BatchNormalization()
    activ2 = LeakyReLU(0.2)
    mp2 = MaxPool1D(pool_size=3)
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization()
    activ3 = LeakyReLU(0.2)
    mp3 = MaxPool1D(pool_size=3)
    # do3 = Dropout(0.2)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization()
    activ4 = LeakyReLU(0.2)
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization()
    activ5 = LeakyReLU(0.2)
    do5 = Dropout(0.3)

    ap = GlobalAvgPool1D()
    
    # euc_dist = Lambda(euclidean_dist, euclidean_dist_output_shape)
    # negative_sampling = Lambda(neg_sample, neg_sample_output_shape)

    # l2_dist = Lambda(lambda  x: K.l2_normalize(x[0] - x[1],axis=1))

    # Anchor 
    anchor_out = mp1(activ1(bn1(conv1(anchor))))
    anchor_out = mp2(activ2(bn2(conv2(anchor_out))))
    anchor_out = mp3(activ3(bn3(conv3(anchor_out))))
    # anchor_out = do3(anchor_out)
    anchor_out = mp4(activ4(bn4(conv4(anchor_out))))
    anchor_out = activ5(bn5(conv5(anchor_out)))
    anchor_out = do5(anchor_out)
    anchor_out = ap(anchor_out)




    model = Model(inputs=anchor, outputs=anchor_output)
    return model 




def mix_cnn_block(num_frame):
    anchor = Input(shape=(num_frame, 128))
    
    # audio model 
    conv1 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn1 = BatchNormalization()
    activ1 = LeakyReLU(0.2)
    mp1 = MaxPool1D(pool_size=3)

    conv2 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn2 = BatchNormalization()
    activ2 = LeakyReLU(0.2)
    mp2 = MaxPool1D(pool_size=3)
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization()
    activ3 = LeakyReLU(0.2)
    mp3 = MaxPool1D(pool_size=3)
    # do3 = Dropout(0.2)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization()
    activ4 = LeakyReLU(0.2)
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization()
    activ5 = LeakyReLU(0.2)
    do5 = Dropout(0.3)

    ap = GlobalAvgPool1D()
    
    # euc_dist = Lambda(euclidean_dist, euclidean_dist_output_shape)
    # negative_sampling = Lambda(neg_sample, neg_sample_output_shape)

    # l2_dist = Lambda(lambda  x: K.l2_normalize(x[0] - x[1],axis=1))

    # Anchor 
    anchor_out = mp1(activ1(bn1(conv1(anchor))))
    anchor_out = mp2(activ2(bn2(conv2(anchor_out))))
    anchor_out = mp3(activ3(bn3(conv3(anchor_out))))
    # anchor_out = do3(anchor_out)
    anchor_out = mp4(activ4(bn4(conv4(anchor_out))))
    anchor_out = activ5(bn5(conv5(anchor_out)))
    anchor_out = do5(anchor_out)
    anchor_out = ap(anchor_out)


    model = Model(inputs=anchor, outputs=anchor_output)
    return model 






def siamese_cnn_mono2mix(num_frame, num_neg_artist, num_pos_track):
    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_track)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_artist)]

    # vocal audio model 
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
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization()
    activ3 = LeakyReLU(0.2)
    # activ3 = Activation('relu')
    mp3 = MaxPool1D(pool_size=3)
    # do3 = Dropout(0.2)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization()
    activ4 = LeakyReLU(0.2)
    # activ4 = Activation('relu')
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization()
    activ5 = LeakyReLU(0.2)
    # activ5 = Activation('relu')
    do5 = Dropout(0.3)

    ap = GlobalAvgPool1D()
    

    # mix audio model 
    m_conv1 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    m_bn1 = BatchNormalization()
    m_activ1 = LeakyReLU(0.2)
    # activ1 = Activation('relu')
    m_mp1 = MaxPool1D(pool_size=3)

    m_conv2 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    m_bn2 = BatchNormalization()
    m_activ2 = LeakyReLU(0.2)
    # activ2 = Activation('relu')
    m_mp2 = MaxPool1D(pool_size=3)
    
    m_conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    m_bn3 = BatchNormalization()
    m_activ3 = LeakyReLU(0.2)
    # activ3 = Activation('relu')
    m_mp3 = MaxPool1D(pool_size=3)
    do3 = Dropout(0.5)
    
    m_conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    m_bn4 = BatchNormalization()
    m_activ4 = LeakyReLU(0.2)
    # activ4 = Activation('relu')
    m_mp4 = MaxPool1D(pool_size=3)

    m_conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    m_bn5 = BatchNormalization()
    m_activ5 = LeakyReLU(0.2)
    # activ5 = Activation('relu')
    m_do5 = Dropout(0.3)

    m_ap = GlobalAvgPool1D()
    
    # v_classification = Dense(config.num_singers, activation='softmax', name='v_classification')
    m_classification = Dense(config.num_singers, activation='softmax', name='m_classification')

    
    # Anchor 
    anchor_out = mp1(activ1(bn1(conv1(anchor))))
    anchor_out = mp2(activ2(bn2(conv2(anchor_out))))
    anchor_out = mp3(activ3(bn3(conv3(anchor_out))))
    anchor_out = do3(anchor_out)
    anchor_out = mp4(activ4(bn4(conv4(anchor_out))))
    anchor_out = activ5(bn5(conv5(anchor_out)))
    anchor_out = do5(anchor_out)
    anchor_out = ap(anchor_out)

    # v_classification_out = v_classification(anchor_out)

    # Pos 
    pos_outs = [m_mp1(m_activ1(m_bn1(m_conv1(pos_item)))) for pos_item in pos_items]
    pos_outs = [m_mp2(m_activ2(m_bn2(m_conv2(pos_out)))) for pos_out in pos_outs]
    pos_outs = [m_mp3(m_activ3(m_bn3(m_conv3(pos_out)))) for pos_out in pos_outs]
    pos_outs = [do3(pos_out) for pos_out in pos_outs]
    pos_outs = [m_mp4(m_activ4(m_bn4(m_conv4(pos_out)))) for pos_out in pos_outs]
    pos_outs = [m_activ5(m_bn5(m_conv5(pos_out))) for pos_out in pos_outs]
    pos_outs = [m_do5(pos_out) for pos_out in pos_outs]
    pos_outs = [m_ap(pos_out) for pos_out in pos_outs]
    
    # pos_classification_out = [m_classification(pos_out) for pos_out in pos_outs]
    # pos_classification_out = m_classification(pos_outs[0])


    # Negs
    neg_outs = [m_mp1(m_activ1(m_bn1(m_conv1(neg_item)))) for neg_item in neg_items]
    neg_outs = [m_mp2(m_activ2(m_bn2(m_conv2(neg_out)))) for neg_out in neg_outs]
    neg_outs = [m_mp3(m_activ3(m_bn3(m_conv3(neg_out)))) for neg_out in neg_outs]
    neg_outs = [do3(neg_out) for neg_out in neg_outs]
    neg_outs = [m_mp4(m_activ4(m_bn4(m_conv4(neg_out)))) for neg_out in neg_outs]
    neg_outs = [m_activ5(m_bn5(m_conv5(neg_out))) for neg_out in neg_outs]
    neg_outs = [m_do5(neg_out) for neg_out in neg_outs]
    neg_outs = [m_ap(neg_out) for neg_out in neg_outs]

    # neg_classification_out = [m_classification(neg_out) for neg_out in neg_outs]


    #### cosine  
    pos_dists = [dot([anchor_out, pos_out], axes=1, normalize=True) for pos_out in pos_outs]
    neg_dists = [dot([anchor_out, neg_out], axes=1, normalize=True) for neg_out in neg_outs]
    # pos_dists = [l2_dist([anchor_out, pos_out]) for pos_out in pos_outs]
    # neg_dists = [l2_dist([anchor_out, neg_out]) for neg_out in neg_outs]
    
    all_dists = concatenate(pos_dists + neg_dists)
    # all_dists = negative_sampling(all_dists)
    outputs = Activation('linear', name='siamese')(all_dists)

    # m_classification_out = concatenate(pos_classification_out +  neg_classification_out, name='m_classification')

     

    # model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=[outputs, pos_classification_out])
    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)
    return model 




def finetuning_mono2mix(vocal_model, mix_model, num_frame,num_neg_artist, num_pos_track):

    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_track)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_artist)]

    # dense = Dense(256)
    # ap = GlobalAvgPool1D()

    anchor_out = vocal_model(anchor)
    # anchor_out = mix_model(anchor)
    pos_outs = [mix_model(pos_item) for pos_item in pos_items]
    neg_outs = [mix_model(neg_item) for neg_item in neg_items]


    ### cosine 
    pos_dists = [dot([anchor_out, pos_out], axes=1, normalize=True) for pos_out in pos_outs]
    neg_dists = [dot([anchor_out, neg_out], axes=1, normalize=True) for neg_out in neg_outs]
    
    all_dists = concatenate(pos_dists + neg_dists)

    outputs = Activation('linear', name='siamese')(all_dists)
    '''

    # euc distance 
    norm = Lambda(lambda x: K.l2_normalize(x, axis=1), name='l2_norm')
    anchor_out = norm(anchor_out)
    pos_outs = [norm(pos_out) for pos_out in pos_outs]
    neg_outs = [norm(neg_out) for neg_out in neg_outs]

    distance = Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape, name='euclidean')
    pos_dists = [distance([anchor_out, pos_out]) for pos_out in pos_outs]
    neg_dists = [distance([anchor_out, neg_out]) for neg_out in neg_outs]

    outputs = concatenate(pos_dists + neg_dists)
    '''

    '''
    distance  = Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape, name='euclidean')
    pos_dist = distance([anchor_out, pos_outs[0]]) 

    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=[outputs, pos_dist])
    '''
    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)

    return model 





def siamese_cnn(num_frame, num_neg_artist, num_pos_track):
    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_track)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_artist)]

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
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization()
    activ3 = LeakyReLU(0.2)
    # activ3 = Activation('relu')
    mp3 = MaxPool1D(pool_size=3)
    # do3 = Dropout(0.2)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization()
    activ4 = LeakyReLU(0.2)
    # activ4 = Activation('relu')
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization()
    activ5 = LeakyReLU(0.2)
    # activ5 = Activation('relu')
    do5 = Dropout(0.3)

    ap = GlobalAvgPool1D()
    
    # euc_dist = Lambda(euclidean_dist, euclidean_dist_output_shape)
    # negative_sampling = Lambda(neg_sample, neg_sample_output_shape)

    # l2_dist = Lambda(lambda  x: K.l2_normalize(x[0] - x[1],axis=1))

    # Anchor 
    anchor_out = mp1(activ1(bn1(conv1(anchor))))
    anchor_out = mp2(activ2(bn2(conv2(anchor_out))))
    anchor_out = mp3(activ3(bn3(conv3(anchor_out))))
    # anchor_out = do3(anchor_out)
    anchor_out = mp4(activ4(bn4(conv4(anchor_out))))
    anchor_out = activ5(bn5(conv5(anchor_out)))
    anchor_out = do5(anchor_out)
    anchor_out = ap(anchor_out)

    # Pos 
    pos_outs = [mp1(activ1(bn1(conv1(pos_item)))) for pos_item in pos_items]
    pos_outs = [mp2(activ2(bn2(conv2(pos_out)))) for pos_out in pos_outs]
    pos_outs = [mp3(activ3(bn3(conv3(pos_out)))) for pos_out in pos_outs]
    # pos_outs = [do3(pos_out) for pos_out in pos_outs]
    pos_outs = [mp4(activ4(bn4(conv4(pos_out)))) for pos_out in pos_outs]
    pos_outs = [activ5(bn5(conv5(pos_out))) for pos_out in pos_outs]
    pos_outs = [do5(pos_out) for pos_out in pos_outs]
    pos_outs = [ap(pos_out) for pos_out in pos_outs]


    # Negs
    neg_outs = [mp1(activ1(bn1(conv1(neg_item)))) for neg_item in neg_items]
    neg_outs = [mp2(activ2(bn2(conv2(neg_out)))) for neg_out in neg_outs]
    neg_outs = [mp3(activ3(bn3(conv3(neg_out)))) for neg_out in neg_outs]
    # neg_outs = [do3(neg_out) for neg_out in neg_outs]
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
    outputs = Activation('linear')(all_dists)

    

    ### euclidean 
    '''
    distance  = Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape)
    pos_dists = [distance([anchor_out, pos_out]) for pos_out in pos_outs]
    neg_dists = [distance([anchor_out, neg_out]) for neg_out in neg_outs]
    all_dists = concatenate(pos_dists + neg_dists)

    outputs = all_dists 
    '''

    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)
    return model 



def track_average(tensors):

    return K.mean(tf.convert_to_tensor(tensors), axis=0, keepdims=False)

def track_average_output_shape(input_shapes):
    return tuple(shape)

def siamese_cnn_track_level(num_frame, num_neg_artist, num_vocal_segments):
    anchor_items = [Input(shape=(num_frame,config.n_mels)) for i in range(num_vocal_segments)]
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_vocal_segments)]
    neg_items_of_items= [[Input(shape=(num_frame, config.n_mels)) for i in range(num_vocal_segments)] for j in range(num_neg_artist)]

    # audio model 
    conv1 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn1 = BatchNormalization(axis=2)
    activ1 = LeakyReLU(0.2)
    mp1 = MaxPool1D(pool_size=3)

    conv2 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn2 = BatchNormalization(axis=2)
    activ2 = LeakyReLU(0.2)
    mp2 = MaxPool1D(pool_size=3)
    
    conv3 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn3 = BatchNormalization(axis=2)
    activ3 = LeakyReLU(0.2)
    mp3 = MaxPool1D(pool_size=3)
    do3 = Dropout(0.5)
    
    conv4 = Conv1D(128, kernel_size=3, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn4 = BatchNormalization(axis=2)
    activ4 = LeakyReLU(0.2)
    mp4 = MaxPool1D(pool_size=3)

    conv5 = Conv1D(256, kernel_size=1, padding='same', use_bias=True, kernel_regularizer=l2(1e-5), kernel_initializer='he_normal')
    bn5 = BatchNormalization(axis=2)
    activ5 = LeakyReLU(0.2)
    do5 = Dropout(0.5)

    ap = GlobalAvgPool1D()

    track_avg = Lambda(track_average, track_average_output_shape)


    # Anchor 
    anchor_outs = [mp1(activ1(bn1(conv1(anchor)))) for anchor in anchor_items]
    anchor_outs = [mp2(activ2(bn2(conv2(anchor_out)))) for anchor_out in anchor_outs]
    anchor_outs = [mp3(activ3(bn3(conv3(anchor_out)))) for anchor_out in anchor_outs]
    anchor_outs = [do3(anchor_out) for anchor_out in anchor_outs]
    anchor_outs = [mp4(activ4(bn4(conv4(anchor_out)))) for anchor_out in anchor_outs]
    anchor_outs = [activ5(bn5(conv5(anchor_out))) for anchor_out in anchor_outs]
    anchor_outs = [do5(anchor_out) for anchor_out in anchor_outs]
    anchor_outs = [ap(anchor_out) for anchor_out in anchor_outs]
    print ('anchor out', len(anchor_outs), np.array(anchor_outs).shape, anchor_outs[0].shape)

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
    neg_outs_of_outs = [[mp1(activ1(bn1(conv1(neg_item)))) for neg_item in neg_items] for neg_items in neg_items_of_items]
    neg_outs_of_outs = [[mp2(activ2(bn2(conv2(neg_out)))) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[mp3(activ3(bn3(conv3(neg_out)))) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[do3(neg_out) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[mp4(activ4(bn4(conv4(neg_out)))) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[activ5(bn5(conv5(neg_out))) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[do5(neg_out) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]
    neg_outs_of_outs = [[ap(neg_out) for neg_out in neg_outs] for neg_outs in neg_outs_of_outs]

    
    # track level averaging 


    anchor_mean = track_avg(anchor_outs)
    pos_mean = track_avg(pos_outs)
    neg_means = [track_avg(neg_outs)for neg_outs in neg_outs_of_outs]

    print ('mean', anchor_mean.shape)


    pos_dist = dot([anchor_mean, pos_mean], axes=1, normalize=True)
    neg_dists = [dot([anchor_mean, neg_mean], axes=1, normalize=True) for neg_mean in neg_means]

    all_dists = concatenate([pos_dist] +  neg_dists)

    outputs = Activation('linear')(all_dists)

    
    inputs = [] 
    for track_specs in neg_items_of_items:
        for ts in track_specs:
            inputs.append(ts)
    inputs = anchor_items + pos_items + inputs 
    print ('inputs', len(inputs))
    model = Model(inputs=inputs, outputs=outputs)
    return model 











def skeleton_cnn(num_frame, weights):
    x_input = Input(shape=(num_frame, 128))
    
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
    
    # Anchor 
    out = mp1(activ1(bn1(conv1(x_input))))
    out = mp2(activ2(bn2(conv2(out))))
    out = mp3(activ3(bn3(conv3(out))))
    out = do3(out)
    out = mp4(activ4(bn4(conv4(out))))
    out = activ5(bn5(conv5(out)))
    out = do5(out)
    out = ap(out)
   
    # out = Dense(num_artist, activation='softmax')(out)
    out = dot([out, out], axes=1, normalize=True)
    out = Activation('linear')(out)
    model = Model(inputs=x_input, outputs = out)

    model.load_weights(weights)
    return model



def finetuning_siamese_cnn(mymodel_tmp, num_frame, num_neg_artist, num_pos_tracks):
    anchor = Input(shape=(num_frame,config.n_mels))
    pos_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_pos_tracks)]
    neg_items = [Input(shape=(num_frame, config.n_mels)) for i in range(num_neg_artist)]

    dense = Dense(256)
    ap = GlobalAvgPool1D()

    anchor_out = mymodel_tmp(anchor)
    pos_outs = [mymodel_tmp(pos_item) for pos_item in pos_items]
    neg_outs = [mymodel_tmp(neg_item) for neg_item in neg_items]

    # anchor_out = dense(anchor_out)
    # pos_outs = [dense(pos_out) for pos_out in pos_outs]
    # neg_outs = [dense(neg_out) for neg_out in neg_outs]

    # anchor_out = ap(anchor_out)
    # pos_outs = [ap(pos_out) for pos_out in pos_outs]
    # neg_outs = [ap(neg_out) for neg_out in neg_outs]



    ### euclidean 
    '''
    distance  = Lambda(euclidean_dist, output_shape=euclidean_dist_output_shape)
    pos_dists = [distance([anchor_out, pos_out]) for pos_out in pos_outs]
    neg_dists = [distance([anchor_out, neg_out]) for neg_out in neg_outs]
    all_dists = concatenate(pos_dists + neg_dists)

    outputs = all_dists 
    '''

    ### cosine 
    pos_dists = [dot([anchor_out, pos_out], axes=1, normalize=True) for pos_out in pos_outs]
    neg_dists = [dot([anchor_out, neg_out], axes=1, normalize=True) for neg_out in neg_outs]
    
    all_dists = concatenate(pos_dists + neg_dists)

    outputs = Activation('linear')(all_dists)

    model = Model(inputs=[anchor]+ pos_items + neg_items, outputs=outputs)

    return model 




###################### Singer classification model ##########################33

def single_layer_dnn(latent_dim, num_artist):

    x_input = Input(shape=(latent_dim,))

    output = Dense(num_artist, activation='softmax')(x_input)
    
    model = Model(inputs=x_input, outputs=output)
    return model 


def double_layer_dnn(latent_dim, num_artist):

    x_input = Input(shape=(latent_dim,))
    output = Dense(1024)(x_input)
    output = Dense(num_artist, activation='softmax')(x_input)

    model = Model(inputs=x_input, outputs=output)
    return model 





if __name__ == '__main__':
    model = siamese_cnn(129, 4)
    print (model.summary())
