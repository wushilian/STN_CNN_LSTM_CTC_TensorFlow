import cv2,time,os,re
import tensorflow as tf
import numpy as np
import utils
import model

FLAGS = utils.FLAGS

inferFolder = 'svttest'
imgList = []
for root,subFolder,fileList in os.walk(inferFolder):
    for fName in fileList:
        #if re.match(r'.*\.[jpg|png|jpeg]',fName.lower()):
        img_Path = os.path.join(root,fName)
        imgList.append(img_Path)


def main():
    g = model.Graph()
    with tf.Session(graph = g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')
        
        imgStack = []
        right = total  = 0
        for img in imgList:
            #pattern = r'.*_(.*)\..*'
            try:
                org = img.split('_')[1]
            except:
                print('>>>>>>>>the img name does not match the pattern: ',img)
                continue
            total+=1
            im = cv2.imread(img,0).astype(np.float32)
            im = cv2.resize(im,(utils.image_width,utils.image_height))
            im = im.swapaxes(0,1)
            im=im[:,:,np.newaxis]/255.
            imgStack.append(im)

            start = time.time()
            def get_input_lens(seqs):
                leghs = np.array([len(s) for s in seqs],dtype=np.int64)
                return seqs,leghs
            inp,seq_len = get_input_lens(np.array([im]))
            feed={g.inputs : inp,
                 g.seq_len : np.array([27])}
            d = sess.run(g.decoded[0],feed)
            dense_decoded = tf.sparse_tensor_to_dense(d,default_value=-1).eval(session=sess)
            res = '' 
            for d in dense_decoded:
                for i in d:
                    if i == -1:
                        res+=''
                    else:
                        res+=utils.decode_maps[i]
            print('cost time: ',time.time()-start)
            if res == org: right+=1
            else: print('ORG: ',org,' decoded: ',res)
        print('total accuracy: ',right*1.0/total)
def acc():
    g = model.Graph()
    with tf.Session(graph = g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess,ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore') 
        val_feeder=utils.DataIterator(data_dir=inferFolder)
        val_inputs,val_seq_len,val_labels=val_feeder.input_index_generate_batch()
        val_feed={g.inputs: val_inputs,
                  g.labels: val_labels,
                 g.seq_len: np.array([27]*val_inputs.shape[0])}
        dense_decoded= sess.run(g.dense_decoded,val_feed)
        
                        # print the decode result
        acc = utils.accuracy_calculation(val_feeder.labels,dense_decoded,ignore_value=-1,isPrint=True)
        print(acc)
    

if __name__ == '__main__':
    acc()
