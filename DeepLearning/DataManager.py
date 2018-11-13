import tensorflow as tf
import os


def produce_images(images, labels, n_input, n_classes, batch_size):
    '''build datasets from images and labels
    '''
    sess = tf.Session()
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    _data = tf.placeholder(tf.float32, [None, n_input])
    _labels = tf.placeholder(tf.float32, [None, n_classes])
    sess.run(iterator.initializer, feed_dict={_data: images, _labels: labels})
    X, y = iterator.get_next()
    return X, y


def read_images(dataset_path, target_shape, batch_size, mode='folder'):
    '''build datasets from files or directory
    Args:
        dataset_path: datasets' directory
        target_shape: the image shape to be resized to 
        mode: 'folder', 'file'
    '''
    imagepaths, labels = list(), list()
    HEIGHT, WIDTH, CHANNELS = target_shape[0], target_shape[1], target_shape[2]
    if mode == 'file':
        data = open(dataset_path, 'r').read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        label = 0
        classes = sorted(os.walk(dataset_path).__next__()[1])

        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception("Unknown mode")

    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)

    image, label = tf.train.slice_input_producer([imagepaths, labels],
                                                 shuffle=True)

    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)

    image = tf.image.resize_images(image, [HEIGHT, WIDTH])

    image = image * 1.0 / 127.5 - 1.0

    X, y = tf.train.batch([image, label],
                          batch_size=batch_size,
                          capacity=batch_size * 8,
                          num_threads=4)

    return X, y
