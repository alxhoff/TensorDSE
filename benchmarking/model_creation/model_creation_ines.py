import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import numpy as np
import os 
import subprocess
import math 

'''
Python script to facilitate the automation of TensorFlow "one-operation" models creation.
'''

def empty_model():
    '''
    Function that creates an empty model, however durign run time Tensorflow benchmark did not recognize it. 
    '''
    w= tf.get_variable('w', initializer=0)
    bidule = tf.no_op(name="bidule")
    g = tf.get_default_graph()
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(bidule)
        saver.save(sess, 'graph.ckpt')
        gdef = g.as_graph_def()
        tf.train.write_graph(gdef, ".", "graph.pb", False)
    freeze_graph.freeze_graph('graph.pb', "", True, 'graph.ckpt', "bidule", "", "", "graph_f.pb", True, None )

def ops_models(operation_string, models_folder, target_shape, input_type):
    '''
    operation_string being evaluated 
    models_folder = path where the ops models would be saved 
    '''
    input_1 = tf.placeholder(dtype = input_type, shape=target_shape, name='input_1')
    #input_2 = tf.placeholder(dtype = float, shape=[1,224,224,3], name='input_2')
    #needed variable otherwise generates error no variables to save 
    w= tf.get_variable('w', initializer=0)
    x = eval(operation_string)
    print("name of x ", x.name)
    #output_1 = tf.placeholder_with_default(x, shape= None, name='output_1')
    output_1 = tf.identity(x, name="output_1")
    save_model_to = x.name[:len(x.name)-10]
    print("save model to ", save_model_to)
    subprocess.run("cd "+ models_folder + " && mkdir "+ save_model_to, shell=True)
    print("models_folder", models_folder)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        #sess.run(x, feed_dict={input_1:np.zeros([1,224,224,3]), input_2:np.zeros([1,224,224,3])})
        sess.run(x,feed_dict={input_1:np.zeros(target_shape, dtype = input_type)})
        saver.save(sess, models_folder + '/' + save_model_to + '/' + 'graph' + x.name[:-10] +'.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), models_folder + '/' + save_model_to + "/", "graph" + x.name[:-10] + ".pb", False)

    freeze_graph.freeze_graph(models_folder + '/' + save_model_to + "/" + "graph" + x.name[:-10] + ".pb", "", True, models_folder + '/' + save_model_to + "/" + 'graph' + x.name[:-10] +'.ckpt' , "output_1", "", "", models_folder + '/' + save_model_to+ "/" + "graph" + x.name[:-10] + "_f.pb", True, None )
    #if not(os.path.exists(models_folder + '/' + save_model_to + "/" + "non_frozen_model")):
    #    subprocess.run( "cd " + models_folder + '/' + save_model_to + "/" +  " && mkdir non_frozen_model", shell=True)
    
    #subprocess.run("cd " + models_folder + '/' + save_model_to + "/ && mv " + "graph" + x.name[:-2] + ".pb /non_frozen_model", shell=True)


    tf.reset_default_graph()

#def dense_layer(models_folder,target_shape, input_type):



def conv_models(models_folder,target_shape, input_type): 
    '''
    operation_string a evaluer 
    models_folder = path where the ops models would be saved 
    '''
    # variables 
    #num_maps =1
    padding = 'VALID'
    strides = [1,3,3,1]
    #img = np.random.rand(1,224,224,1)
    #img = np.random.rand(target_shape)
    print(target_shape[0])
    img = np.random.rand(target_shape[0],target_shape[1], target_shape[2], target_shape[3] )
    kernel = np.zeros([3,3,3,3], dtype=input_type)
    kernel[1, 1, :, :] = 5
    kernel[0, 1, :, :] = -1
    kernel[1, 0, :, :] = -1
    kernel[2, 1, :, :] = -1
    kernel[1, 2, :, :] = -1        

    #input_1 = tf.placeholder(dtype=float32, shape=[1, 224, 224, num_maps], name="input_1" )
    input_1 = tf.placeholder(input_type, target_shape, name="input_1")
    w = tf.get_variable('w', initializer=tf.to_float(kernel))
    #w = tf.get_variable('w', initializer=kernel)
    #w = tf.get_variable('w', initializer=tf.uint8(kernel))

    init = tf.initialize_all_variables()
    convx = tf.nn.conv2d(input_1, w, strides=strides, padding=padding, name="convx")
    print(convx.name)
    #output_1 = tf.placeholder_with_default(convx, shape= None, name='output_1')
    #output_1 = tf.placeholder(convx, shape= None, name = "output_1")
    output_1 = tf.identity(convx, name="output_1")
    save_model_to = convx.name[:len(convx.name)-2] + "_" + str(target_shape[1]) + "_" + str(input_type)
    #save_model_to = convx.name[:len(convx.name)-2] + "_" + str(target_shape[1])
    subprocess.run("cd "+ models_folder + " && mkdir "+ save_model_to, shell=True)

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        output_1 = sess.run(convx, feed_dict={input_1:img})
        saver.save(sess, models_folder + '/' + save_model_to + '/' + 'graph' + convx.name[:-2] +'.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), models_folder + '/' + save_model_to + "/", "graph" + convx.name[:-2] + ".pb", False)

    freeze_graph.freeze_graph(models_folder + '/' + save_model_to + "/" + "graph" + convx.name[:-2] + ".pb", "", True, models_folder + '/' + save_model_to + "/" + 'graph' + convx.name[:-2]+'.ckpt' , "output_1", "", "", models_folder + '/' + save_model_to+ "/" + "graph" + convx.name[:-2] + "_f.pb", True, None )
    
    '''
    cmd_toco = "toco --graph_def_file="  + "graph" + convx.name + "_f.pb" + " --output_file=" + "/" + models_folder + '/' + save_model_to + '/'+ "graph" + convx.name + ".tflite" + " --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=QUANTIZED_UINT8 --input_shape=" + "1,224,224,1" + " --input_array=input_1 --output_array=output_1 --std_dev_values=127 --mean_values=127 --default_ranges_min=0 --default_ranges_max=100" 
    subprocess.run(cmd_toco, shell=True, cwd='/' + models_folder + '/' + save_model_to+ "/")
    
    cmd_edge = "usr/bin/edgetpu_compiler -o "+"/" + models_folder + "/" + save_model_to+ "/" + "  "+ "/" + models_folder + "/" + save_model_to+ "/"  + "graph" + convx.name + ".tflite" 
    
    os.system(cmd_edge)
    '''
    if not(os.path.exists(models_folder + '/' + save_model_to + "/" + "non_frozen_model")):
        subprocess.run( "cd " + models_folder + '/' + save_model_to + "/" +  " && mkdir non_frozen_model", shell=True)
    subprocess.run("cd " + models_folder + '/' + save_model_to + "/ && mv " + "graph" + convx.name[:-2] + ".pb /non_frozen_model", shell=True)
    tf.reset_default_graph()


def depthwise_conv_models(models_folder, target_shape, input_type):
    '''
    operation_string a evaluer 
    models_folder = path where the ops models would be saved 
    '''
    # variables 
    #num_maps =1
    padding = 'SAME'
    strides = [1,3,3,1]
    #img = np.random.rand(1,224,224,1)
    img = np.random.rand(target_shape[0], target_shape[1], target_shape[2], target_shape[3])
    #img = np.random.rand(target_shape)
    kernel = np.zeros([3,3,3,3], dtype = input_type)
    kernel[1, 1, :, :] = 5
    kernel[0, 1, :, :] = -1
    kernel[1, 0, :, :] = -1
    kernel[2, 1, :, :] = -1
    kernel[1, 2, :, :] = -1        

    #input_1 = tf.placeholder(dtype=float32, shape=[1, 224, 224, num_maps], name="input_1" )
    input_1 = tf.placeholder(input_type, target_shape, name="input_1")
    #w = tf.get_variable('w', initializer=kernel)
    w = tf.get_variable('w', initializer=tf.to_float(kernel))
    #w = tf.get_variable('w', initializer=tf.uint8(kernel))

    init = tf.initialize_all_variables()
    convx = tf.nn.depthwise_conv2d(input_1, w, strides=strides, padding=padding, name="depthwiseconv2d")
    #print(convx.name)
    #output_1 = tf.placeholder_with_default(convx, shape= None, name='output_1')
    #output_1 = tf.placeholder(convx, shape= None, name = "output_1")
    output_1 = tf.identity(convx, name="output_1")

    save_model_to = convx.name[:len(convx.name)-2] + "_" + str(target_shape[1]) + "_" + str(input_type)
    subprocess.run("cd "+ models_folder + " && mkdir "+ save_model_to, shell=True)

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        output_1 = sess.run(convx, feed_dict={input_1:img})
        saver.save(sess, models_folder + '/' + save_model_to + '/' + 'graph' + convx.name[:-2] +'.ckpt')
        tf.train.write_graph(sess.graph.as_graph_def(), models_folder + '/' + save_model_to + "/", "graph" + convx.name[:-2] + ".pb", False)

    freeze_graph.freeze_graph(models_folder + '/' + save_model_to + "/" + "graph" + convx.name[:-2] + ".pb", "", True, models_folder + '/' + save_model_to + "/" + 'graph' + convx.name[:-2] +'.ckpt' , "output_1", "", "", models_folder + '/' + save_model_to+ "/" + "graph" + convx.name[:-2] + "_f.pb", True, None )
    
    '''
    cmd_toco = "toco --graph_def_file="  + "graph" + convx.name + "_f.pb" + " --output_file=" + "/" + models_folder + '/' + save_model_to + '/'+ "graph" + convx.name + ".tflite" + " --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --inference_type=QUANTIZED_UINT8 --input_shape=" + "1,224,224,1" + " --input_array=input_1 --output_array=output_1 --std_dev_values=127 --mean_values=127 --default_ranges_min=0 --default_ranges_max=100" 
    subprocess.run(cmd_toco, shell=True, cwd='/' + models_folder + '/' + save_model_to+ "/")
    
    cmd_edge = "usr/bin/edgetpu_compiler -o "+"/" + models_folder + "/" + save_model_to+ "/" + "  "+ "/" + models_folder + "/" + save_model_to+ "/"  + "graph" + convx.name + ".tflite" 
    
    os.system(cmd_edge)
    '''
    if not(os.path.exists(models_folder + '/' + save_model_to + "/" + "non_frozen_model")):
        subprocess.run( "cd " + models_folder + '/' + save_model_to + "/" +  " && mkdir non_frozen_model", shell=True)
    subprocess.run("cd " + models_folder + '/' + save_model_to + "/ && mv " + "graph" + convx.name[:-2] + ".pb /non_frozen_model", shell=True)
    tf.reset_default_graph()


def run_pgm():
    models_folder = input("Path of ops models folder: ") 
    if not(os.path.exists(models_folder)):
        subprocess.run("mkdir "+ models_folder, shell=True)
    #operations_list = [ "tf.add(input_1, input_2, name=\"add\") ", "tf.multiply(input_1, input_2, name=\"multiply_elementwise\")", "tf.maximum(input_1, input_2, name=\"maximum_elementwise\")", "tf.minimum(input_1, input_2, name=\"minimum_elementwise\")" ]
    #operations_list = [ "tf.add(input_1, input_1, name=\"add_", "tf.multiply(input_1, input_1, name=\"multiply_elementwise_" ]
    #operations_list = [ "tf.math.minimum(input_1, input_1, name=\"minimum_elementwise_", ]
    #operations_list = ["tf.nn.relu(input_1, name=\"relu_", "tf.nn.relu6(input_1, name=\"relu6_" , "tf.add(input_1, input_1, name=\"add_", "tf.subtract(input_1, input_1, name=\"sub_", "tf.math.tanh(input_1, name=\"tanh_", "tf.multiply(input_1, input_1, name=\"multiply_elementwise_", "tf.maximum(input_1, input_1, name=\"maximum_elementwise_", "tf.minimum(input_1, input_1, name=\"minimum_elementwise_",  "tf.math.l2_normalize(input_1, axis = None, epsilon= 1e-12, name=\"L2Normalization_" ]
    #operations_list = ["tf.identity(input_1, name=\"Identity_", "tf.layers.dense(input_1, units, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regulizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True,  name=\"Dense_"]
    operations_list = ["tf.layers.dense(input_1, units=3, activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True,  name=\"Dense\", reuse=None )"]
    target_shape_list= [[1,10,10,3], [1,20,20,3], [1,30,30,3], [1,40,40,3], [1,50,50,3], [1,60,60,3], [1,70,70,3], [1,80,80,3],[1,90,90,3] ,[1,100,100,3], [1,192,192,3], [1,224,224,3], [1,300,300,3], [1,400,400,3], [1,500,500,3], [1,600,600,3], [1,700,700,3], [1,800,800,3], [1,900,900,3], [1,1000,1000,3]]
    input_type_list = ["float", "float16", "float32", "int32"]
    #input_type_list = ["uint8"]
    for operation_string_iter in operations_list: 
        
        for input_type in input_type_list:
            #try:
            for target_shape in target_shape_list:
                    #try: 
                        #operation_string = operation_string_iter + str(target_shape[1]) + "_" + input_type + "\" )"
                                    #try:
                operation_string1 = operation_string_iter
                ops_models(operation_string1, models_folder, target_shape, input_type)
                        #ops_models(operation_string, models_folder, target_shape, input_type)
                        #conv_models(models_folder, target_shape, input_type)
                        #depthwise_conv_models(models_folder, target_shape, input_type)

                    #except:
                    #    print("no" + operation_string1)
                        #print(operation_string)
                    #    break
            #except:
                #continue                #ops_models(operation_string, models_folder)
                       
run_pgm()
