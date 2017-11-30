import tensorflow as tf
def conv_block(input,growth_rate,is_train,dropout_rate=None):
    x=tf.layers.batch_normalization(input, training=is_train)
    x=tf.nn.relu(x)
    x = tf.layers.conv2d(x, growth_rate, 3, 1, 'SAME')
    if dropout_rate is not None:
        x = tf.nn.dropout(x, dropout_rate)
    return x
def dense_block(x,nb_layers,growth_rate,nb_filter,is_train,droput_rate=0.2):
    for i in range(nb_layers):
        cb = conv_block(x,growth_rate,is_train,droput_rate)
        x = tf.concat([x,cb],3)
        nb_filter +=growth_rate
    return x ,nb_filter

def transition_block(x, c, is_train, dropout_kp=None,pooltype=1):

    y = x
    x = tf.layers.batch_normalization(x, training=is_train)
    x = tf.nn.relu(x)
    x = tf.layers.conv2d(x, c, 1, 1, "SAME")
    if dropout_kp is not None:
        x = tf.nn.dropout(x, dropout_kp)
    if (pooltype == 2):
        x = tf.nn.avg_pool(x, [1,2,2,1], [1,2,2,1], "VALID")
    elif (pooltype == 1):
        x = tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 1, 1], "SAME")
    elif (pooltype == 3):
        x = tf.nn.avg_pool(x, [1,2,2,1], [1,1,2,1], "SAME")
    return x,c
