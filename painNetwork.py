#

# network = mnistnet()
import tensorflow as tf

global NUM_CLASSES
NUM_CLASSES = 8

global IMGH
IMGH = 576

global IMGW
IMGW = 720

class PainNet:
  def __init__(self):
    with tf.Graph().as_default():
      self.sess        = tf.Session()
      self.global_step = tf.Variable(0,name='global_step',trainable = False)
      self.step        = 0

      self.input = tf.placeholder(tf.float32, [None, IMGH, IMGW, 3], name = 'Img_Input')
      self.label = tf.placeholder(tf.int64,   [None]               , name = 'Lab_Input')
      self.label_oh = tf.one_hot(self.label,int(NUM_CLASSES))

      self.inference()
      self.metrics()
      self.saver()

      self.sess.run(tf.local_variables_initializer())
      self.sess.run(tf.global_variables_initializer())


  def inference(self):
    net   = self.input / 255
    net   = tf.layers.conv2d(net,filters = 8,kernel_size = (3,3),strides = 2, padding = 'VALID')

    chan1 = tf.layers.conv2d(net  ,filters = 8,kernel_size = (3,3),strides = 2 , padding = 'VALID')
    chan2 = tf.layers.conv2d(net  ,filters = 8,kernel_size = (1,3),strides = (1,2), padding = 'VALID')
    chan2 = tf.layers.conv2d(chan2,filters = 16,kernel_size = (3,1),strides = (2,1), padding = 'VALID')
    chan3 = tf.layers.conv2d(net  ,filters = 8,kernel_size = (3,1),strides = (2,1) , padding = 'VALID')
    chan3 = tf.layers.conv2d(chan3,filters = 12,kernel_size = (1,3),strides = (1,2), padding = 'VALID')

    net = tf.concat([chan1,chan2,chan3],-1)

    net = tf.layers.conv2d(net    ,filters = 8 ,kernel_size = (1,1))
    net = tf.layers.max_pooling2d(net,pool_size = 2, strides = 2, padding = 'VALID')

    imgH = net.shape[1].value
    imgW = net.shape[2].value

    net = tf.reshape(net,[-1,imgH * imgW * 8])

    net = tf.layers.dense(net,676)
    net = tf.layers.dense(net,NUM_CLASSES,name='Logits')
    self.output = net

  def metrics(self):
    input(self.label_oh)
    cross_entropy = tf.losses.softmax_cross_entropy(labels=self.label_oh, logits=self.output)
    loss = tf.reduce_mean(cross_entropy)

    # The TF Metrics Accuracy wasn't working properly, so I used this instead
    correct_prediction = tf.equal(self.label, tf.argmax(self.output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    train = self.optimize(loss)

    tf.summary.scalar("Accuracy",accuracy)
    tf.summary.scalar("Loss",loss)

    self.metrics = [accuracy,train]
    self.summaries = tf.summary.merge_all()

    self.operation=[self.metrics,self.summaries,self.output]

  def optimize(self,loss):
    optimizer = tf.train.GradientDescentOptimizer(0.0015)
    train_op = optimizer.minimize(loss)
    return train_op

  def saver(self):
    self.saver  = tf.train.Saver()
    self.writer = tf.summary.FileWriter('./',self.sess.graph)

  def save(self):
    self.saver.save(self.sess,"./log/")

  def run(self,images,label):
    # self.step = tf.train.global_step(self.sess,self.global_step)
    self.step+=1
    _metrics,summary_out,_outputs = self.sess.run(self.operation, feed_dict = {self.input: images, self.label: label})
    step      = tf.train.global_step(self.sess,self.global_step)

    # Save and log the network every 10 steps.
    if self.step%10 == 0:
      self.writer.add_summary(summary_out,self.step)
      if self.step%100 == 0:
        self.save()

    return _metrics[0],_outputs
