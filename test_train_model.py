from tensorflow.python import pywrap_tensorflow
import os
import tensorflow as tf
from darwinutils.log import get_task_logger
logger = get_task_logger(__name__)
def train_model_darwinnet(FLAGS,x_batch,y_batch,tensor_dict):
    """
    :param x_batch:
    :param y_batch:
    :param our_output_ckpt_dir:
    :param ckpt_dir:
    :return:
    """
    #The function currently has bug, it cannot normally train. Gaoyang Tang will continuous improve.
    import tensorflow as tf
    import cv2
    tf.reset_default_graph()
    tf_graph = tf.get_default_graph()
    # Restore the graph
    if FLAGS.get("our_output_ckpt_dir") != None or os.path.exists(FLAGS.get("our_output_ckpt_dir")) is True:
        saver = tf.train.import_meta_graph(FLAGS.get("our_output_ckpt_dir")+'.meta')
    #x_image = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, 3), name='image')
    y_truth = tf_graph.get_tensor_by_name(tensor_dict.get("inputy_tensor", "input_1_1")+':0')
    input_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("inputx_tensor", "input_1")+':0')
    logits_tensor = tf_graph.get_tensor_by_name(tensor_dict.get("logit_tensor", "dense_1/BiasAdd")+':0')

    def compute_loss(y_truth,logits):
        y_truth = tf.squeeze(y_truth)#tf.one_hot(y_truth, 1001, on_value=1, off_value=None, axis=1)
        y_truth = tf.cast(y_truth,tf.int64)
        logits = tf.cast(logits, tf.float32)
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_truth, name = 'xentropy')
        xentropy = tf.reduce_mean(xentropy, name = 'xentropy_mean')
        return xentropy

    def compute_acc(y_truth, logits):
        logits = tf.cast(logits, tf.float32)
        y_truth = tf.cast(y_truth, tf.int64)
        acc = tf.equal(tf.argmax(logits, 1), y_truth)
        acc = tf.reduce_mean(tf.cast(acc, tf.float32))
        return acc

    proba_tensor = tf.nn.softmax(logits_tensor)
    pred_tensor = tf.argmax(logits_tensor, 1)
    loss = compute_loss(y_truth, logits_tensor)
    acc = compute_acc(y_truth, logits_tensor)
    optim = tf.train.AdamOptimizer(FLAGS.get("learning_rate"),name="optim").minimize(loss)
    itr = 0
    import time
    with tf_graph.as_default():
        # get y_pred
        with tf.Session(graph=tf_graph) as sess:
            # Restore the weights
            sess.run(tf.global_variables_initializer())
            if saver is not None:
                saver.restore(sess, FLAGS.get("our_output_ckpt_dir"))
                logger.info("restore {}".format(FLAGS.get("our_output_ckpt_dir")))
            if FLAGS.get("log_path") != None or os.path.exists(FLAGS.get("log_path")) is True:
                ckpt = tf.train.get_checkpoint_state(FLAGS.get("log_path"))
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    logger.info("restore {}".format(ckpt.model_checkpoint_path))
            else:
                saver = tf.train.Saver(save_relative_paths=True)
            while(1):
                try:
                    itr += 1
                    x = next(x_batch)
                    x = (x*255-128)/128
                    y = next(y_batch)
                    all_ops = tf.get_default_graph().get_operations()
                    all_ops_names = [op.name for op in all_ops]
                    if 'is_training' in all_ops_names:
                        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
                        #logger.info("Find existed is_training placeholder")
                    else:
                        is_training = tf.placeholder(dtype=bool, shape=(), name='is_training')

                    keras_learning_phase = None
                    for op in all_ops:
                        if op.name.find('keras_learning_phase') != -1 and op.type == 'PlaceholderWithDefault':
                            keras_learning_phase = tf.get_default_graph().get_tensor_by_name('{}:0'.format(op.name))
                            #logger.info("Find existed keras_learning_phase placeholder")
                    if keras_learning_phase is None:
                        keras_learning_phase = tf.placeholder(dtype=bool, shape=(), name='keras_learning_phase')
                    loss_res, acc_res = sess.run([loss,acc], feed_dict={input_tensor: x, y_truth: y,is_training: True,
                                                              keras_learning_phase: True})

                    if itr % FLAGS.get("display") == 0:
                        logger.info(" itr: {}, loss:{}, acc:{}".format(itr,loss_res,acc_res))
                    if itr % FLAGS.get("save_ckpt") == 0:
                        saver.save(sess,os.path.join(FLAGS.get("log_path"), 'model.ckpt'))
                        summary_op = tf.summary.merge_all()
                        summary_writer = tf.summary.FileWriter(FLAGS.get("log_path"), sess.graph)
                except StopIteration as e:
                    logger.info("End of input due to StopIteration.")
                    logger.info("Finish a epoch.")
                    break
            logger.info("Finish train.")


def main_store_ckpt(FLAGS,tensor_dict, valid_dict,adapt_for_darwinnet_order):
    """
    main function of this script. It will restore ckpt by resore_ckpt, diff assigned darwin ckpt with official ckpt by using
    valid_ckpt function.
    If you want print the darwin net classcification result, net_type need configure 'darwin'.
    evaluation_model will predict and evalute classfication result of two ckpt(official and assigned darwin ckpt).

    If you only do evaluations, you can annotation resore_ckpt and valid_ckpt
    :param adapt_for_darwinnet_order: if the tensor order are changed by xk.
    :param FLAGS.ckpt_dir: official ckpt path
    :param FLAGS.our_input_ckpt_dir: darwin ckpt path
    :param FLAGS.our_output_ckpt_dir: assign official weight to our darwin ckpt. The path is where to save after assigned the official weight.
    :param tensor_dict: The tensor dict is the the tensor that you what to get from ckpt. The first location of the value list is the darwin's ckpt tensor name.
    The second value is the official ckpt name. It uses to predict the data and check the weight whether to assign.
    :param valid_dict: valid_dict uses to valid tensor of official ckpt and your assigned darwin ckpt, and valid whether to equal.
    :return:
    """

    resore_ckpt(FLAGS,adapt_for_darwinnet_order)
    result1, result2 = valid_ckpt(FLAGS,valid_dict)
    yprdict, y_truth, eq_res = evaluation_model(FLAGS,tensor_dict)
    logger.info("The darwin net classfication result:{}".format(yprdict.get("class")))
    logger.info("The ground truth:{}".format(y_truth.get("class")))
    return result1, result2, eq_res


if __name__ == "__main__":

    '''
    mobilenet config:
    # ckpt_dir = "/home/gytang/ckpt/mobile_net/official/final"
    # our_input_ckpt_dir = "/home/gytang/ckpt/mobile_net/init_darwinnet/model.ckpt-5"
    # our_output_ckpt_dir = "/home/gytang/ckpt/mobile_net/darwinnet/final"
    # tensor_dict={"input_tensor":["input_1","Placeholder"],
    #              "logit_tensor":["dense_1/BiasAdd","MobileNet/fc_16/BiasAdd"],
    #              "check_tensor":["dense_1/BiasAdd","MobileNet/fc_16/BiasAdd"]}
    #valid_dict=[("1_Conv/kernel","MobileNet/conv_1/weights"),("4_Depthwise_conv2d/depthwise_kernel","MobileNet/conv_ds_2/depthwise_conv/depthwise_weights")]
    # full_connect_flag = False
    '''
    # The tensor dict is the the tensor that you what to get from ckpt. The first location of the value list is the darwin's ckpt tensor name.
    # The second value is the official ckpt name. It uses to predict the data and check the weight whether to assign.
    tensor_dict={"input_tensor":["input_1", "Placeholder"],
                 "logit_tensor":["flatten_1/Reshape", "resnet_v2_50/predictions/Reshape"],
                 "check_tensor":["21_Activation/Relu","resnet_v2_50/postnorm/Relu"]}#resnet_unit_19/add","resnet_v2_50/block4/unit_3/bottleneck_v2/add"]}
    #valid_dict uses to valid tensor of official ckpt and your assigned darwin ckpt, and valid whether to equal.
    valid_dict = [
        ("2_Conv/kernel", "resnet_v2_50/conv1/weights"),
        ("2_Conv/bias", "resnet_v2_50/conv1/biases")
    ]