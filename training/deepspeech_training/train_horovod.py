#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import os
import sys

LOG_LEVEL_INDEX = sys.argv.index('--log_level') + 1 if '--log_level' in sys.argv else 0
DESIRED_LOG_LEVEL = sys.argv[LOG_LEVEL_INDEX] if 0 < LOG_LEVEL_INDEX < len(sys.argv) else '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = DESIRED_LOG_LEVEL

import absl.app
import progressbar
import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import time

tfv1.logging.set_verbosity({
                               '0': tfv1.logging.DEBUG,
                               '1': tfv1.logging.INFO,
                               '2': tfv1.logging.WARN,
                               '3': tfv1.logging.ERROR
                           }.get(DESIRED_LOG_LEVEL))

from datetime import datetime
from .util.config import Config, initialize_globals
from .util.checkpoints import load_or_init_graph_for_training, reload_best_checkpoint
from .util.feeding import create_dataset, audio_to_features
from .util.flags import create_flags, FLAGS
from .util.helpers import check_ctcdecoder_version, ExceptionBox
from .util.logging import create_progressbar, log_debug, log_error, log_info, log_progress

from .train import create_optimizer, get_tower_results, average_gradients, log_grads_and_vars, rnn_impl_static_rnn, \
    rnn_impl_lstmblockfusedcell, create_model, export, test, do_single_file_inference, package_zip, \
    early_training_checks

import horovod.tensorflow as hvd

check_ctcdecoder_version()


def initialize_horovod():
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    #Config._config.gpu_options.visible_device_list = str(hvd.local_rank())
    Config.available_devices = str(hvd.local_rank())


def train():
    exception_box = ExceptionBox()

    # Create training and validation datasets
    train_set = create_dataset(FLAGS.train_files.split(','),
                               batch_size=FLAGS.train_batch_size,
                               epochs=FLAGS.epochs,
                               augmentations=Config.augmentations,
                               cache_path=FLAGS.feature_cache,
                               train_phase=True,
                               exception_box=exception_box,
                               process_ahead=len(Config.available_devices) * FLAGS.train_batch_size * 2,
                               reverse=FLAGS.reverse_train,
                               limit=FLAGS.limit_train,
                               buffering=FLAGS.read_buffer)

    iterator = tfv1.data.Iterator.from_structure(tfv1.data.get_output_types(train_set),
                                                 tfv1.data.get_output_shapes(train_set),
                                                 output_classes=tfv1.data.get_output_classes(train_set))

    # Make initialization ops for switching between the two sets
    train_init_op = iterator.make_initializer(train_set)

    if FLAGS.dev_files:
        dev_sources = FLAGS.dev_files.split(',')
        dev_sets = [create_dataset([source],
                                   batch_size=FLAGS.dev_batch_size,
                                   train_phase=False,
                                   exception_box=exception_box,
                                   process_ahead=len(Config.available_devices) * FLAGS.dev_batch_size * 2,
                                   reverse=FLAGS.reverse_dev,
                                   limit=FLAGS.limit_dev,
                                   buffering=FLAGS.read_buffer) for source in dev_sources]
        dev_init_ops = [iterator.make_initializer(dev_set) for dev_set in dev_sets]

    if FLAGS.metrics_files:
        metrics_sources = FLAGS.metrics_files.split(',')
        metrics_sets = [create_dataset([source],
                                       batch_size=FLAGS.dev_batch_size,
                                       train_phase=False,
                                       exception_box=exception_box,
                                       process_ahead=len(Config.available_devices) * FLAGS.dev_batch_size * 2,
                                       reverse=FLAGS.reverse_dev,
                                       limit=FLAGS.limit_dev,
                                       buffering=FLAGS.read_buffer) for source in metrics_sources]
        metrics_init_ops = [iterator.make_initializer(metrics_set) for metrics_set in metrics_sets]

    # Dropout
    dropout_rates = [tfv1.placeholder(tf.float32, name='dropout_{}'.format(i)) for i in range(6)]
    dropout_feed_dict = {
        dropout_rates[0]: FLAGS.dropout_rate,
        dropout_rates[1]: FLAGS.dropout_rate2,
        dropout_rates[2]: FLAGS.dropout_rate3,
        dropout_rates[3]: FLAGS.dropout_rate4,
        dropout_rates[4]: FLAGS.dropout_rate5,
        dropout_rates[5]: FLAGS.dropout_rate6,
    }
    no_dropout_feed_dict = {
        rate: 0. for rate in dropout_rates
    }

    # Building the graph
    learning_rate_var = tfv1.get_variable('learning_rate', initializer=FLAGS.learning_rate, trainable=False)
    reduce_learning_rate_op = learning_rate_var.assign(tf.multiply(learning_rate_var, FLAGS.plateau_reduction))

    # Effective batch size in synchronous distributed training is scaled by the number of workers. An increase in learning rate compensates for the increased batch size.
    optimizer = create_optimizer(learning_rate_var * hvd.size())
    optimizer = hvd.DistributedOptimizer(optimizer)

    # Enable mixed precision training
    if FLAGS.automatic_mixed_precision:
        log_info('Enabling automatic mixed precision training.')
        optimizer = tfv1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

    gradients, loss, non_finite_files = get_tower_results(iterator, optimizer, dropout_rates)

    # Average tower gradients across GPUs
    avg_tower_gradients = average_gradients(gradients)
    log_grads_and_vars(avg_tower_gradients)

    # Add hook to broadcast variables from rank 0 to all other processes during
    # initialization.
    hooks = [hvd.BroadcastGlobalVariablesHook(0)]

    # global_step is automagically incremented by the optimizer
    global_step = tfv1.train.get_or_create_global_step()
    apply_gradient_op = optimizer.apply_gradients(avg_tower_gradients, global_step=global_step)

    # Summaries
    step_summaries_op = tfv1.summary.merge_all('step_summaries')
    step_summary_writers = {
        'train': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), max_queue=120),
        'dev': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'dev'), max_queue=120),
        'metrics': tfv1.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'metrics'), max_queue=120),
    }

    human_readable_set_names = {
        'train': 'Training',
        'dev': 'Validation',
        'metrics': 'Metrics',
    }

    # # Checkpointing
    # checkpoint_saver = tfv1.train.Saver(max_to_keep=FLAGS.max_to_keep)
    # checkpoint_path = os.path.join(FLAGS.save_checkpoint_dir, 'train')
    #
    # best_dev_saver = tfv1.train.Saver(max_to_keep=1)
    # best_dev_path = os.path.join(FLAGS.save_checkpoint_dir, 'best_dev')
    #
    # # Save flags next to checkpoints
    # os.makedirs(FLAGS.save_checkpoint_dir, exist_ok=True)
    # flags_file = os.path.join(FLAGS.save_checkpoint_dir, 'flags.txt')
    # with open(flags_file, 'w') as fout:
    #     fout.write(FLAGS.flags_into_string())

    # Save checkpoints only on worker 0 to prevent other workers from corrupting them.
    checkpoint_dir = os.path.join(FLAGS.save_checkpoint_dir, 'train') if hvd.rank() == 0 else None

    # with tfv1.Session(config=Config.session_config) as session:
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           config=Config.session_config,
                                           hooks=hooks) as session:
        log_debug('Session opened.')

        # Prevent further graph changes
        tfv1.get_default_graph().finalize()

        # Load checkpoint or initialize variables
        load_or_init_graph_for_training(session)

        def run_set(set_name, epoch, init_op, dataset=None):
            is_train = set_name == 'train'
            train_op = apply_gradient_op if is_train else []
            feed_dict = dropout_feed_dict if is_train else no_dropout_feed_dict

            total_loss = 0.0
            step_count = 0

            step_summary_writer = step_summary_writers.get(set_name)
            checkpoint_time = time.time()

            if is_train and FLAGS.cache_for_epochs > 0 and FLAGS.feature_cache:
                feature_cache_index = FLAGS.feature_cache + '.index'
                if epoch % FLAGS.cache_for_epochs == 0 and os.path.isfile(feature_cache_index):
                    log_info('Invalidating feature cache')
                    os.remove(feature_cache_index)  # this will let TF also overwrite the related cache data files

            # Setup progress bar
            class LossWidget(progressbar.widgets.FormatLabel):
                def __init__(self):
                    progressbar.widgets.FormatLabel.__init__(self, format='Loss: %(mean_loss)f')

                def __call__(self, progress, data, **kwargs):
                    data['mean_loss'] = total_loss / step_count if step_count else 0.0
                    return progressbar.widgets.FormatLabel.__call__(self, progress, data, **kwargs)

            prefix = 'Epoch {} | {:>10}'.format(epoch, human_readable_set_names[set_name])
            widgets = [' | ', progressbar.widgets.Timer(),
                       ' | Steps: ', progressbar.widgets.Counter(),
                       ' | ', LossWidget()]
            suffix = ' | Dataset: {}'.format(dataset) if dataset else None
            pbar = create_progressbar(prefix=prefix, widgets=widgets, suffix=suffix).start()

            # Initialize iterator to the appropriate dataset
            session.run(init_op)

            # Batch loop
            while True:
                try:
                    _, current_step, batch_loss, problem_files, step_summary = \
                        session.run([train_op, global_step, loss, non_finite_files, step_summaries_op],
                                    feed_dict=feed_dict)
                    exception_box.raise_if_set()
                except tf.errors.OutOfRangeError:
                    exception_box.raise_if_set()
                    break

                if problem_files.size > 0:
                    problem_files = [f.decode('utf8') for f in problem_files[..., 0]]
                    log_error('The following files caused an infinite (or NaN) '
                              'loss: {}'.format(','.join(problem_files)))

                total_loss += batch_loss
                step_count += 1

                pbar.update(step_count)

                step_summary_writer.add_summary(step_summary, current_step)

                # if is_train and FLAGS.checkpoint_secs > 0 and time.time() - checkpoint_time > FLAGS.checkpoint_secs:
                #     checkpoint_saver.save(session, checkpoint_path, global_step=current_step)
                #     checkpoint_time = time.time()

            pbar.finish()
            mean_loss = total_loss / step_count if step_count > 0 else 0.0
            return mean_loss, step_count

        log_info('STARTING Optimization')
        train_start_time = datetime.utcnow()
        best_dev_loss = float('inf')
        dev_losses = []
        epochs_without_improvement = 0
        try:
            for epoch in range(FLAGS.epochs):
                # Training
                log_progress('Training epoch %d...' % epoch)
                train_loss, _ = run_set('train', epoch, train_init_op)
                log_progress('Finished training epoch %d - loss: %f' % (epoch, train_loss))
                # checkpoint_saver.save(session, checkpoint_path, global_step=global_step)

                if FLAGS.dev_files:
                    # Validation
                    dev_loss = 0.0
                    total_steps = 0
                    for source, init_op in zip(dev_sources, dev_init_ops):
                        log_progress('Validating epoch %d on %s...' % (epoch, source))
                        set_loss, steps = run_set('dev', epoch, init_op, dataset=source)
                        dev_loss += set_loss * steps
                        total_steps += steps
                        log_progress('Finished validating epoch %d on %s - loss: %f' % (epoch, source, set_loss))

                    dev_loss = dev_loss / total_steps
                    dev_losses.append(dev_loss)

                    # Count epochs without an improvement for early stopping and reduction of learning rate on a plateau
                    # the improvement has to be greater than FLAGS.es_min_delta
                    if dev_loss > best_dev_loss - FLAGS.es_min_delta:
                        epochs_without_improvement += 1
                    else:
                        epochs_without_improvement = 0

                    # # Save new best model
                    # if dev_loss < best_dev_loss:
                    #     best_dev_loss = dev_loss
                    #     save_path = best_dev_saver.save(session, best_dev_path, global_step=global_step,
                    #                                     latest_filename='best_dev_checkpoint')
                    #     log_info("Saved new best validating model with loss %f to: %s" % (best_dev_loss, save_path))

                    # Early stopping
                    if FLAGS.early_stop and epochs_without_improvement == FLAGS.es_epochs:
                        log_info('Early stop triggered as the loss did not improve the last {} epochs'.format(
                            epochs_without_improvement))
                        break

                    # Reduce learning rate on plateau
                    # If the learning rate was reduced and there is still no improvement
                    # wait FLAGS.plateau_epochs before the learning rate is reduced again
                    if (
                            FLAGS.reduce_lr_on_plateau
                            and epochs_without_improvement > 0
                            and epochs_without_improvement % FLAGS.plateau_epochs == 0
                    ):
                        # Reload checkpoint that we use the best_dev weights again
                        reload_best_checkpoint(session)

                        # Reduce learning rate
                        session.run(reduce_learning_rate_op)
                        current_learning_rate = learning_rate_var.eval()
                        log_info('Encountered a plateau, reducing learning rate to {}'.format(
                            current_learning_rate))

                        # Overwrite best checkpoint with new learning rate value
                        # save_path = best_dev_saver.save(session, best_dev_path, global_step=global_step,
                        #                                 latest_filename='best_dev_checkpoint')
                        # log_info("Saved best validating model with reduced learning rate to: %s" % (save_path))

                if FLAGS.metrics_files:
                    # Read only metrics, not affecting best validation loss tracking
                    for source, init_op in zip(metrics_sources, metrics_init_ops):
                        log_progress('Metrics for epoch %d on %s...' % (epoch, source))
                        set_loss, _ = run_set('metrics', epoch, init_op, dataset=source)
                        log_progress('Metrics for epoch %d on %s - loss: %f' % (epoch, source, set_loss))

                print('-' * 80)


        except KeyboardInterrupt:
            pass
        log_info('FINISHED optimization in {}'.format(datetime.utcnow() - train_start_time))
    log_debug('Session closed.')




def main(_):
    initialize_globals()
    initialize_horovod()
    early_training_checks()

    if FLAGS.train_files:
        tfv1.reset_default_graph()
        tfv1.set_random_seed(FLAGS.random_seed)
        train()

    if hvd.rank() == 0:
        if FLAGS.test_files:
            tfv1.reset_default_graph()
            test()

        if FLAGS.export_dir and not FLAGS.export_zip:
            tfv1.reset_default_graph()
            export()

        if FLAGS.export_zip:
            tfv1.reset_default_graph()
            FLAGS.export_tflite = True

            if os.listdir(FLAGS.export_dir):
                log_error('Directory {} is not empty, please fix this.'.format(FLAGS.export_dir))
                sys.exit(1)

            export()
            package_zip()

        if FLAGS.one_shot_infer:
            tfv1.reset_default_graph()
            do_single_file_inference(FLAGS.one_shot_infer)


def run_script():
    create_flags()
    absl.app.run(main)


if __name__ == '__main__':
    run_script()
