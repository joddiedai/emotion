import argparse
import pickle
import sys
import numpy as np
import try_ql

from data_prep import batch_iter,get_raw_data

seed = 1234

np.random.seed(seed)
import tensorflow as tf
from tqdm import tqdm

from MODEL_ import LSTM_Model

from sklearn.metrics import f1_score

tf.set_random_seed(seed)

unimodal_activations = {}


def multimodal(unimodal_activations, data, classes, attn_fusion=True, enable_attn_2=False, use_raw=True):
    q_table = try_ql.rl()[2]
    if use_raw:
        if attn_fusion:
            attn_fusion = False

        train_data, test_data, audio_train, audio_test, text_train, text_test, video_train, video_test, train_label, test_label, seqlen_train, seqlen_test, train_mask, test_mask = get_raw_data(
            data, classes)
        print("classes", classes)


    else:
        print("starting multimodal")
        # Fusion (appending) of features

        text_train = unimodal_activations['text_train']
        audio_train = unimodal_activations['audio_train']
        video_train = unimodal_activations['video_train']
        # print(video_train.shape)

        text_test = unimodal_activations['text_test']
        audio_test = unimodal_activations['audio_test']
        video_test = unimodal_activations['video_test']

        train_mask = unimodal_activations['train_mask']
        test_mask = unimodal_activations['test_mask']
        print("test_mask", test_mask.shape)

        print('train_mask', train_mask.shape)

        train_label = unimodal_activations['train_label']
        print('train_label', train_label.shape)
        test_label = unimodal_activations['test_label']
        print(test_label.dtype)

        # print(train_mask_bool)
        seqlen_train = np.sum(train_mask, axis=-1)
        print('seqlen_train', seqlen_train.shape)
        seqlen_test = np.sum(test_mask, axis=-1)
        print('seqlen_test', seqlen_test.shape)

    a_dim = audio_train.shape[-1]
    v_dim = video_train.shape[-1]
    t_dim = text_train.shape[-1]
    if attn_fusion:
        print('With attention fusion')
    allow_soft_placement = True
    log_device_placement = False

    # Multimodal model
    session_conf = tf.ConfigProto(
        # device_count={'GPU': gpu_count},
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement,
        gpu_options=tf.GPUOptions(allow_growth=True))
    gpu_device = 0
    best_acc = 0
    best_loss_accuracy = 0
    best_loss = 10000000.0
    best_epoch = 0
    best_epoch_loss = 0
    with tf.device('/device:GPU:%d' % gpu_device):
        print('Using GPU - ', '/device:GPU:%d' % gpu_device)
        with tf.Graph().as_default():
            tf.set_random_seed(seed)
            sess = tf.Session(config=session_conf)
            with sess.as_default():

                model = LSTM_Model(text_train.shape[1:], 0.0001,
                                   a_dim=a_dim,
                                   v_dim=v_dim,
                                   t_dim=t_dim,
                                   emotions=classes, attn_fusion=attn_fusion,
                                   unimodal=False, enable_attn_2=enable_attn_2,
                                   seed=seed)
                sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

                test_feed_dict = {
                    model.a_input: audio_test,  # 31,63,100
                    model.v_input: video_test,  # 31,63,100
                    model.t_input: text_test,  # 31,63,100
                    model.chosen_layer: np.array([2]),
                    model.y: test_label,  # 31,63,2
                    model.seq_len: seqlen_test,  # 31
                    model.mask: test_mask,  # 31,63
                    model.lstm_dropout: 0.0,
                    model.lstm_inp_dropout: 0.0,
                    model.dropout: 0.0,
                    model.dropout_lstm_out: 0.0
                }

                print()
                print(test_mask.shape)
                print("\nEvaluation before training:")
                # Evaluation after epoch
                step, loss, accuracy = sess.run(
                    [model.global_step, model.loss, model.accuracy],
                    test_feed_dict)
                print("EVAL: epoch {}: step {}, loss {:g}, acc {:g}".format(0, step, loss, accuracy))

                for epoch in range(epochs):
                    epoch += 1

                    batches = batch_iter(list(
                        zip(audio_train, video_train, text_train, train_mask, seqlen_train, train_label)),
                        batch_size)


                    # Training loop. For each batch...
                    print('\nTraining epoch {}'.format(epoch))
                    l = []
                    a = []
                    for i, batch in tqdm(enumerate(batches)):
                        chosen = []
                        chosen.append(q_table[epoch-1])
                        b_audio_train, b_video_train, b_text_train, b_train_mask, b_seqlen_train, b_train_label = zip(
                            *batch)
                        # print('batch_hist_v', len(batch_utt_v))
                        feed_dict = {
                            model.a_input: b_audio_train,
                            model.v_input: b_video_train,
                            model.t_input: b_text_train,
                            model.chosen_layer: np.array(chosen),
                            model.y: b_train_label,
                            model.seq_len: b_seqlen_train,
                            model.mask: b_train_mask,
                            model.lstm_dropout: 0.4,
                            model.lstm_inp_dropout: 0.0,
                            model.dropout: 0.2,
                            model.dropout_lstm_out: 0.2
                        }

                        _, step, loss, accuracy = sess.run(
                            [model.train_op, model.global_step, model.loss, model.accuracy],
                            feed_dict)
                        l.append(loss)
                        a.append(accuracy)

                    print("\t \tEpoch {}:, loss {:g}, accuracy {:g}".format(epoch, np.average(l), np.average(a)))
                    # Evaluation after epoch
                    step, loss, accuracy, preds, y, mask = sess.run(
                        [model.global_step, model.loss, model.accuracy, model.preds, model.y, model.mask],
                        test_feed_dict)
                    f1 = f1_score(np.ndarray.flatten(tf.argmax(y, -1, output_type=tf.int32).eval()),
                                  np.ndarray.flatten(tf.argmax(preds, -1, output_type=tf.int32).eval()),
                                  sample_weight=np.ndarray.flatten(tf.cast(mask, tf.int32).eval()), average="weighted")
                    print("EVAL: After epoch {}: step {}, loss {:g}, acc {:g}, f1 {:g}".format(epoch, step,
                                                                                               loss / test_label.shape[
                                                                                                   0],
                                                                                               accuracy, f1))

                    if accuracy > best_acc:
                        best_epoch = epoch
                        best_acc = accuracy
                        # saver.save(sess, "./checkpoint_dir/",global_step=i+1)

                    if loss < best_loss:
                        best_loss = loss
                        best_loss_accuracy = accuracy
                        best_epoch_loss = epoch

                print(
                    "\n\nBest epoch: {}\nBest test accuracy: {}\nBest epoch loss: {}\nBest test accuracy when loss is least: {}".format(
                        best_epoch, best_acc, best_epoch_loss, best_loss_accuracy))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--unimodal", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--fusion", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument("--attention_2", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--use_raw", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--data", type=str, default='mosi')
    parser.add_argument("--classes", type=str, default='2')
    args, _ = parser.parse_known_args(argv)

    print(args)

    batch_size = 62
    emotions = args.classes
    assert args.data in ['mosi', 'mosei', 'iemocap']

    if not args.use_raw:
        with open('unimodal_{0}_{1}way.pickle'.format(args.data, args.classes), 'rb') as handle:
            u = pickle._Unpickler(handle)
            u.encoding = 'latin1'
            unimodal_activations = u.load()
    epochs = 5
    multimodal(unimodal_activations, args.data, args.classes, args.fusion, args.attention_2, use_raw=args.use_raw)
