import tensorflow as tf
import config as params
import os
import time
'''
build ResNet-18,17 convolution and 1 full connection
'''

class ResNet18(object):
    def __init__(self,log_dir=None,model_path="checkpoint",grpc_target=None,*args,**kwargs):
        '''
        Args:
        log_dir=the directory to save the log file
        model_path:the directory to save the tf model file
        grpc_target:optional param. grpc connect,use grpc://ip:port,use remote device to caculate op
        
        Returns:
        None

        Raise:
        ValueError:
        if model_path miss
        '''
        if not os.path.exists(model_path):
            print("create dir %s to save the model file"%model_path)
            os.makedirs(model_path)

        self.lr=params.LEARNING_RATE
        self.epoches=params.EPOCHES
        self.batch_size=params.BATCH_SIZE
        self.batches=params.BATCHES
        self.log_dir=log_dir
        self.model_path=model_path
        self.class_map=params.DECODE_MAP
        self.num_classes=params.NUM_CLASSES
        self.decay_steps=params.DECAY_STEPS
        self.decay_rate=params.DECAY_RATE
        # self.global_step=tf.get_variable(
        #     name="global_step",
        #     shape=(),dtype=tf.float32,
        #     initializer=tf.zeros_initializer,
        #     trainable=False
        # )
        self.is_training=True
        if grpc_target!=None:
            self.sess=tf.Session(target=grpc_target)
        else:
            self.sess=tf.Session()

    def _basic_block(self,input_data,filters,kernel_size,half=False,padding="same",downsample=False):
        '''
        Args:
        input_data:4-D Tesnor,[batches,height,width,channels]
        filters:the num of output channles
        kernel_size:tuple with 2 integers
        half:bool,true,use (2,2) stride to reduce half size of image,else,(1,1) keep original size
        padding:default is same,use 0 to padding to keep the original size
        downsample:use 1x1 convolution to make the channle of skip connection equal

        Returns:
        a 4-D Tensor with conv->bn->relu->conv->bn(downsample->bn)->skip connect->relu

        Raise:
        ValueError:if input_data missing
        '''
        if half:
            conv_1=tf.layers.conv2d(inputs=input_data,filters=filters,kernel_size=kernel_size,strides=(2,2),padding=padding,use_bias=False)
        else:
            conv_1=tf.layers.conv2d(inputs=input_data,filters=filters,kernel_size=kernel_size,strides=(1,1),padding=padding,use_bias=False)
        bn_1=tf.layers.batch_normalization(inputs=conv_1,epsilon=1e-5)
        activaiton_1=tf.nn.relu(bn_1)
        conv_2=tf.layers.conv2d(inputs=activaiton_1,filters=filters,kernel_size=kernel_size,strides=(1,1),padding=padding,use_bias=False)
        bn_2=tf.layers.batch_normalization(inputs=conv_2,epsilon=1e-5)
        if downsample:
            conv_3=tf.layers.conv2d(inputs=input_data,filters=filters,kernel_size=(1,1),strides=(2,2),use_bias=False)
            bn_3=tf.layers.batch_normalization(inputs=conv_3,epsilon=1e-5)
            activation_2=tf.nn.relu(bn_3+bn_2)
        else:
            activation_2=tf.nn.relu(bn_2+input_data)
        return activation_2
        


    def _build_net(self,input_data=None,num_classes:int=None,reuse=None):
        '''
        Args:
        input_data:4-D Tensor with [batches,height,width,channels]
        num_classes:the nums of the last units
        reuse:bool,if true,shared the variables

        Returns:
        2-D Tensor:linear ouput without activation,because tf.losses.softmax module will activate the layer

        Raise:
        ValueError:if num_classes is invalid,miss or not int 
        '''
        assert (num_classes!=None and isinstance(num_classes,int)),"be sure the num_classes is int value"
        
        conv_1=tf.layers.conv2d(inputs=input_data,filters=64,kernel_size=(7,7),strides=2,use_bias=False)
        bn_1=tf.layers.batch_normalization(inputs=conv_1,epsilon=1e-5)
        activation_1=tf.nn.relu(bn_1)
        pool_2=tf.layers.max_pooling2d(inputs=activation_1,pool_size=(3,3),strides=(2,2))

        with tf.variable_scope("layer_1",reuse=reuse):
            layer1_block1=self._basic_block(input_data=pool_2,filters=64,kernel_size=(3,3),half=False,padding="same")
            layer1_block2=self._basic_block(input_data=layer1_block1,filters=64,half=False,kernel_size=(3,3),padding="same")


        with tf.variable_scope("layer_2",reuse=reuse):
            layer2_block1=self._basic_block(input_data=layer1_block2,filters=128,kernel_size=(3,3),half=True,padding="same",downsample=True)
            layer2_block2=self._basic_block(input_data=layer2_block1,filters=128,kernel_size=(3,3),half=False,padding="same",downsample=False)
            

        with tf.variable_scope("layer_3",reuse=reuse):
            layer3_block1=self._basic_block(input_data=layer2_block2,filters=256,kernel_size=(3,3),half=True,padding="same",downsample=True)
            layer3_block2=self._basic_block(input_data=layer3_block1,filters=256,kernel_size=(3,3),half=False,padding="same",downsample=False)


        with tf.variable_scope("layer_4",reuse=reuse):
            layer4_block1=self._basic_block(input_data=layer3_block2,filters=512,kernel_size=(3,3),half=True,padding="same",downsample=True)
            layer4_block2=self._basic_block(input_data=layer4_block1,filters=512,kernel_size=(3,3),half=False,padding="same",downsample=False)
        
        flatten_1=tf.layers.flatten(inputs=layer4_block2)
        fc_1=tf.layers.dense(inputs=flatten_1,units=num_classes,activation=None,name="output",reuse=reuse)
        return fc_1


    
    def _loss(self,logits,labels,name=None):
        '''
        Args:
        logits:linear output without activation 
        labels:ture label of you dataset,one-hot coding
        name:name of summary,if None,do not add to summary

        Reruns:
        scala of loss
        '''
        loss=tf.losses.softmax_cross_entropy(labels,logits)
        if name!=None:
            loss_scalar=tf.summary.scalar(name,loss)
            return loss_scalar,loss
        return None,loss

    def _accuracy(self,logits,labels,name=None):
        '''
        Args:
        logits:linear output without activation 
        labels:ture label of you dataset,one-hot coding
        name:name of summary,if None,do not add to summary

        Reruns:
        scala of accuracy
        '''
        logits=tf.nn.softmax(logits)
        mask=tf.equal(tf.argmax(logits,axis=-1),tf.argmax(labels,axis=-1))
        mask=tf.cast(mask,tf.float32)
        accuracy=tf.reduce_mean(mask)
        if name!=None:
            accuracy_scalar=tf.summary.scalar(name,accuracy)
            return accuracy_scalar,accuracy
        return None,accuracy

    def train(self,x_train,y_train,x_val=None,y_val=None):
        '''
        Args:
        x_train:4-D Tensor
        y_train:one-hot conding,2-D Tensor
        x_val,y_val,same 

        Returns:
        None

        Raise:
        ValueError:
        if x_train,y_train is missing
        '''
        
        logits=self._build_net(x_train,num_classes=self.num_classes,reuse=False)
        loss_scalar,loss=self._loss(logits,y_train,name="train_loss")
        accuracy_scalar,accuracy=self._accuracy(logits,y_train,name="train_accuracy")
        global_step=tf.Variable(0,trainable=False)
        lr=tf.train.exponential_decay(self.lr,global_step=global_step,decay_steps=self.decay_steps,decay_rate=self.decay_rate)
        lr_scalar=tf.summary.scalar("learning_rate",lr)
        optimizer=tf.train.AdamOptimizer(lr)
        train_op=optimizer.minimize(loss,global_step=global_step)
        summary_writer=tf.summary.FileWriter(self.log_dir,self.sess.graph)
        image_summary=tf.summary.image("training-flowers",x_train,max_outputs=6)

        if loss_scalar!=None and accuracy_scalar!=None:
            merge_op=tf.summary.merge((loss_scalar,accuracy_scalar,lr_scalar))
        else:
            merge_op=tf.summary.merge((lr_scalar))

        init_op=tf.global_variables_initializer()
        saver=tf.train.Saver(max_to_keep=1)
        self.sess.run(init_op)
        if not self.is_training:
            print("restore varaibles from %s"%self.model_path)
            try:
                ckpt=tf.train.latest_checkpoint(self.model_path)
                saver.restore(self.sess,ckpt)
            except Exception as _:
                pass

        for step in range(self.epoches):
            if step%1000==0:
                _loss,_accuracy,_image_summary,train_summary,_=self.sess.run([loss,accuracy,merge_op,image_summary,train_op])
                summary_writer.add_summary(train_summary,step)
                summary_writer.add_summary(_image_summary,step)
            else:
                _loss,_accuracy,train_summary,_=self.sess.run([loss,accuracy,merge_op,train_op])
                summary_writer.add_summary(train_summary,step)

            if (step+1)%10==0:
                print("train step at %s,train_loss is %.4f,train_acc is %.4f time at %s"%(step+1,_loss,_accuracy,time.ctime()))
                
            if (step+1)%100==0:
                save_path=saver.save(self.sess,save_path="%s/ResNet_18.ckpt"%self.model_path,global_step=global_step)
                print("save model in %s"%save_path)
                
    def evalute(self,x_eval,y_eval):
        raise NotImplementedError("eval method not implemented!")

