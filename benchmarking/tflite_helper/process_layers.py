

def process_CONV_2D(operator_name, out_dir, options, input_tensors, output_tensors):
    """Processes the overloaded arguments to recreate the wished Conv 2D model."""
    import numpy as np
    import tensorflow as tf
    from utils.convert import TFLiteConverter, SaveSession
    import tflite_helper.conv_helper as ch
    import tflite_helper as tfh

    # The input tensors for a Conv2D layer are as follows:
    # [0] : The input shape to the Conv2D layer
    # [1] : The filter shape, ie. no of filters x kernel width x kernel height x kernel depth
    # [2] : The number of filters

    input_shape = ch.GetInputShape(input_tensors=input_tensors)
    input_type = ch.GetInputType(input_tensors=input_tensors)
    padding = tfh.GetPadding(options=options)
    strides = tfh.GetStrides(options=options)
    filter_count = ch.GetFilterCount(input_tensors=input_tensors)
    filter_type = ch.GetFilterType(input_tensors=input_tensors)
    kernel_shape = ch.GetKernelShape(input_tensors=input_tensors)

    # TODO the activation function is not used
    activ_func = tfh.GetActivationFunction(options=options)

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:

        # A Tensor. Must have the same type as input. A 4-D tensor of shape
        # [filter_height, filter_width, in_channels, out_channels]
        filter_placeholder = tf.Variable(
            tf.random.normal(np.append(kernel_shape, filter_count), dtype=filter_type),
            dtype=tf.float32,
        )

        input_placeholder = tf.raw_ops.Placeholder(
            dtype=input_type, shape=input_shape, name=operator_name + "_input"
        )

        conv_2d_layer = tf.nn.conv2d(
            input=input_placeholder,
            filters=filter_placeholder,
            strides=strides,
            padding=padding,
            name=operator_name + "_op",
        )

        test_input = np.array(np.random.random_sample(input_shape))
        session_init = tf.compat.v1.global_variables_initializer()
        session.run(session_init)
        session.run(conv_2d_layer, feed_dict={input_placeholder: test_input})
        export_dir = SaveSession(
            session, operator_name, conv_2d_layer, out_dir, input_placeholder
        )

    # Convert .pb into .tflite
    TFLiteConverter(out_dir, export_dir, operator_name, input_placeholder)


def process_MAX_POOL_2D(operator_name, out_dir, options, input_tensors, output_tensors):
    """Processes the overloaded arguments to recreate the wished Max Pool 2D model."""
    import numpy as np
    import tensorflow as tf
    from utils.convert import TFLiteConverter, SaveSession
    import tflite_helper as tfh
    import tflite_helper.pool_helper as ph

    input_shape = tfh.GetInputShape(input_tensors=input_tensors)
    input_type = tfh.GetInputType(input_tensors=input_tensors)
    padding = tfh.GetPadding(options)
    strides = tfh.GetStrides(options)
    pool_size = ph.GetPoolSize(options)
    activ_func = tfh.GetActivationFunction(options)

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:

        input_placeholder = tf.raw_ops.Placeholder(
            dtype=input_type, shape=input_shape, name=operator_name + "_input"
        )

        pool_2d_layer = tf.nn.max_pool2d(
            input=input_placeholder,
            ksize=pool_size,
            strides=strides,
            padding=padding,
            data_format="NHWC",
            name=operator_name + "_op",
        )

        test_input = np.array(np.random.random_sample(input_shape))
        session_init = tf.compat.v1.global_variables_initializer()
        session.run(session_init)
        session.run(pool_2d_layer, feed_dict={input_placeholder: test_input})
        export_dir = SaveSession(
            session, operator_name, pool_2d_layer, out_dir, input_placeholder
        )

    TFLiteConverter(out_dir, export_dir, operator_name, input_placeholder)


def process_RESHAPE(operator_name, out_dir, options, input_tensors, output_tensors):
    """Processes the overloaded arguments to recreate the wished Reshape model."""
    import numpy as np
    import tensorflow as tf
    from utils.convert import TFLiteConverter, SaveSession
    import tflite_helper as tfh

    input_shape = tfh.GetInputShape(input_tensors=input_tensors)
    input_type = tfh.GetInputType(input_tensors=input_tensors)
    output_shape = tfh.GetOutputShape(output_tensors=output_tensors)

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:

        input_placeholder = tf.raw_ops.Placeholder(
            dtype=input_type, shape=input_shape, name=operator_name + "_input"
        )

        reshape_layer = tf.reshape(
            input_placeholder,
            output_shape,
            name=operator_name + "_op",
        )

        test_input = np.array(np.random.random_sample(input_shape), dtype=np.int32)
        session_init = tf.compat.v1.global_variables_initializer()
        session.run(session_init)
        session.run(reshape_layer, feed_dict={input_placeholder: test_input})
        export_dir = SaveSession(
            session, operator_name, reshape_layer, out_dir, input_placeholder
        )

    TFLiteConverter(out_dir, export_dir, operator_name, input_placeholder)


def process_FULLY_CONNECTED(
    operator_name, out_dir, options, input_tensors, output_tensors
):
    """Processes the overloaded arguments to recreate the wished FC model."""
    import numpy as np
    import tensorflow as tf
    from utils.convert import TFLiteConverter, SaveSession
    import tflite_helper as tfh
    import tflite_helper.fully_connected_helper as fch

    activ_func = tfh.GetActivationFunction(options)

    op_name = operator_name + "_" + activ_func if activ_func else operator_name

    input_shape = tfh.GetInputShape(input_tensors=input_tensors)
    input_type = tfh.GetInputType(input_tensors=input_tensors)
    keep_num_dim = fch.GetNumDims(options=options)
    units = fch.GetUnits(output_tensors=output_tensors)

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:

        input_placeholder = tf.raw_ops.Placeholder(
            dtype=input_type, shape=input_shape, name=op_name + "_input"
        )

        fcl = tf.compat.v1.layers.dense(
            inputs=input_placeholder,
            units=units,
            activation=activ_func,
            use_bias=keep_num_dim,
            name=op_name + "_op",
        )

        test_input = np.array(np.random.random_sample(input_shape))
        session_init = tf.compat.v1.global_variables_initializer()
        session.run(session_init)
        session.run(fcl, feed_dict={input_placeholder: test_input})

        export_dir = SaveSession(session, op_name, fcl, out_dir, input_placeholder)

    TFLiteConverter(out_dir, export_dir, op_name, input_placeholder)


def process_SOFTMAX(operator_name, out_dir, options, input_tensors, output_tensors):
    """Processes the overloaded arguments to recreate the wished softmax model."""
    import numpy as np
    import tensorflow as tf
    from utils.convert import TFLiteConverter, SaveSession
    import tflite_helper as tfh
    import tflite_helper.fully_connected_helper as fch

    input_shape = tfh.GetInputShape(input_tensors=input_tensors)
    input_type = tfh.GetInputType(input_tensors=input_tensors)
    test_input = np.array(np.random.random_sample(input_shape))

    graph = tf.Graph()
    with graph.as_default(), tf.compat.v1.Session() as session:

        input_placeholder = tf.raw_ops.Placeholder(
            dtype=input_type, shape=input_shape, name=operator_name + "_input"
        )

        Soft = tf.nn.softmax(input_placeholder, None, operator_name + "_op")

        session_init = tf.compat.v1.global_variables_initializer()
        session.run(session_init)
        session.run(Soft, feed_dict={input_placeholder: test_input})
        export_dir = SaveSession(
            session, operator_name, Soft, out_dir, input_placeholder
        )

    TFLiteConverter(out_dir, export_dir, operator_name, input_placeholder)
