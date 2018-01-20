import model
import utils
import time
import tensorflow as tf
import numpy as np
import os
import logging,datetime

FLAGS=utils.FLAGS
logger = logging.getLogger('Traing for ocr using LSTM+CTC')
logger.setLevel(logging.INFO)

def train(train_dir=None,val_dir=None):
    g = model.Graph(is_training=True)
    print('loading train data, please wait---------------------','end= ')
    train_feeder=utils.DataIterator(data_dir=train_dir)
    print('get image: ',train_feeder.size)
    print('loading validation data, please wait---------------------','end= ')
    val_feeder=utils.DataIterator(data_dir=val_dir)
    print('get image: ',val_feeder.size)

    num_train_samples = train_feeder.size
    num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)
    num_val_samples=val_feeder.size
    num_val_per_epoch=int(num_val_samples/FLAGS.batch_size)

    config=tf.ConfigProto(log_device_placement=False,allow_soft_placement=False)
    with tf.Session(graph=g.graph,config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)
        g.graph.finalize()
        train_writer=tf.summary.FileWriter(FLAGS.log_dir+'/train',sess.graph)
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess,ckpt)
                print('restore from the checkpoint{0}'.format(ckpt))

        print('=============================begin training=============================')
        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
        #print(len(val_inputs))
        val_feed={g.inputs: val_inputs,
                  g.labels: val_labels,
                 g.seq_len: np.array([g.cnn_time]*val_inputs.shape[0])}
        for cur_epoch in range(FLAGS.num_epochs):
            shuffle_idx=np.random.permutation(num_train_samples)
            train_cost = 0
            start_time = time.time()
            batch_time = time.time()
            #the tracing part
            for cur_batch in range(num_batches_per_epoch):
                if (cur_batch+1)%100==0:
                    print('batch',cur_batch,': time',time.time()-batch_time)
                batch_time = time.time()
                indexs = [shuffle_idx[i%num_train_samples] for i in range(cur_batch*FLAGS.batch_size,(cur_batch+1)*FLAGS.batch_size)]
                batch_inputs,batch_seq_len,batch_labels=train_feeder.input_index_generate_batch(indexs)
                #batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)
                feed={g.inputs: batch_inputs,
                        g.labels:batch_labels,
                        g.seq_len:np.array([g.cnn_time]*batch_inputs.shape[0])}

                # if summary is needed
                #batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)
                summary_str, batch_cost,step,_ = sess.run([g.merged_summay,g.cost,g.global_step,g.optimizer],feed)
                #calculate the cost
                train_cost+=batch_cost*FLAGS.batch_size
                train_writer.add_summary(summary_str,step)

                # save the checkpoint
                if step%FLAGS.save_steps == 1000:
                    if not os.path.isdir(FLAGS.checkpoint_dir):
                        os.mkdir(FLAGS.checkpoint_dir)
                    logger.info('save the checkpoint of{0}',format(step))
                    saver.save(sess,os.path.join(FLAGS.checkpoint_dir,'ocr-model'),global_step=step)
                #train_err+=the_err*FLAGS.batch_size
                #do validation
                if step%FLAGS.validation_steps == 0:
                    dense_decoded,lastbatch_err,lr = sess.run([g.dense_decoded,g.lerr,g.learning_rate],val_feed)
                    # print the decode result
                    acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
                    avg_train_cost=train_cost/((cur_batch+1)*FLAGS.batch_size)
                    #train_err/=num_train_samples
                    now = datetime.datetime.now()
                    log = "{}/{} {}:{}:{} Epoch {}/{}, accuracy = {:.3f},avg_train_cost = {:.3f}, lastbatch_err = {:.3f}, time = {:.3f},lr={:.8f}"
                    print(log.format(now.month,now.day,now.hour,now.minute,now.second,
                        cur_epoch+1,FLAGS.num_epochs,acc,avg_train_cost,lastbatch_err,time.time()-start_time,lr))
if __name__ == '__main__':
    #train(train_dir='train',val_dir='val')
    train(train_dir='train', val_dir='test')
