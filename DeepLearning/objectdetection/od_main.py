from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import argparse

import tensorflow as tf
import sys
import os

from object_detection import model_hparams
from object_detection import model_lib

FLAGS = None


def main(_):
    #flags.mark_flag_as_required('model_dir')
    #flags.mark_flag_as_required('pipeline_config_path')
    config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir)

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples))
    estimator = train_and_eval_dict['estimator']
    train_input_fn = train_and_eval_dict['train_input_fn']
    eval_input_fns = train_and_eval_dict['eval_input_fns']
    eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
    predict_input_fn = train_and_eval_dict['predict_input_fn']
    train_steps = train_and_eval_dict['train_steps']

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = 'training_data'
            input_fn = eval_on_train_input_fn
        else:
            name = 'validation_data'
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]
        if FLAGS.run_once:
            estimator.evaluate(
                input_fn,
                num_eval_steps=None,
                checkpoint_path=tf.train.latest_checkpoint(
                    FLAGS.checkpoint_dir))
        else:
            model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir,
                                      input_fn, train_steps, name)
    else:
        train_spec, eval_specs = model_lib.create_train_and_eval_specs(
            train_input_fn,
            eval_input_fns,
            eval_on_train_input_fn,
            predict_input_fn,
            train_steps,
            eval_on_train_data=False)

        # Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_dir',
        type=str,
        default=os.path.join(os.getcwd(),
                             'models\\faster_rcnn_resnet101_coco_11_06_2017'),
        help=
        'Path to output model directory where event and checkpoint files will be written.'
    )
    parser.add_argument(
        '--pipeline_config_path',
        type=str,
        default=os.path.join(
            os.getcwd(),
            'models\\faster_rcnn_resnet101_coco_11_06_2017\\pipeline.config'),
        help='Path to pipeline config file.')
    parser.add_argument(
        '--num_train_steps',
        type=int,
        default=50000,
        help=
        'Path to output model directory where event and checkpoint files will be written.'
    )
    parser.add_argument(
        '--eval_training_data',
        type=bool,
        default=False,
        help=
        'If training data should be evaluated for this job. Note that one call only use this in eval-only mode, and `checkpoint_dir` must be supplied.'
    )
    parser.add_argument(
        '--sample_1_of_n_eval_examples',
        type=int,
        default=1,
        help=
        'Will sample one of every n eval input examples, where n is provided.')
    parser.add_argument(
        '--sample_1_of_n_eval_on_train_examples',
        type=int,
        default=5,
        help=
        'Will sample one of every n train input examples for evaluation, where n is provided. This is only used if `eval_training_data` is True.'
    )
    parser.add_argument(
        '--hparams_overrides',
        type=str,
        default=None,
        help=
        'Hyperparameter overrides, represented as a string containing comma-separated hparam_name=value pairs.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default=None,
        #default=os.path.join(
        #    os.getcwd(),
        #    'models\\faster_rcnn_resnet101_coco_11_06_2017'),
        help=
        'Path to directory holding a checkpoint.  If `checkpoint_dir` is provided, this binary operates in eval-only mode, writing resulting metrics to `model_dir`.'
    )
    parser.add_argument(
        '--run_once',
        type=bool,
        default=False,
        help=
        'If running in eval-only mode, whether to run just one round of eval vs running continuously (default).'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
