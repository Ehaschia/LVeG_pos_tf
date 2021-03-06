from __future__ import print_function
import sys

sys.path.append(".")
sys.path.append("..")

import time
import argparse
import tensorflow as tf
import numpy as np
from neuronlp2.io import get_logger, conllx_data
from neuronlp2.models import BiRecurrentConvCRF, BiRecurrentConvLVeG
from neuronlp2 import utils
import tensorboard
import os
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#
def main():
    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN-CRF')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    # may this is useless
    parser.add_argument('--p_rnn', type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_in', type=float, default=0.5, help='lstm input dropout rate')
    parser.add_argument('--p_out', type=float, default=0.5, help='dense output dropout rate')

    parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot', 'random'],
                        help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--lveg', default=False, action='store_true')
    parser.add_argument('--language', type=str, default='wsj')
    parser.add_argument('--spherical', default=False, action='store_true')
    parser.add_argument('--gaussian-dim', type=int, default=1)
    parser.add_argument('--t-comp', type=int, default=1)
    parser.add_argument('--e-comp', type=int, default=1)
    parser.add_argument('--use-tensorboard', default=False, action='store_true')
    parser.add_argument('--log-dir', type=str, default='./tensorboard/')
    parser.add_argument('--lveg-clip', type=float, default=1.0)

    args = parser.parse_args()

    logger = get_logger("POSCRFTagger")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    test_path = args.test
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = args.p_rnn
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    bigram = args.bigram
    embedding = args.embedding
    embedding_path = args.embedding_dict

    if args.use_tensorboard:
        boardWriter = tf.Summary.FileWriter(args.log_dir)

    def add_loss_summary(step, loss):
        if args.use_tensorboard:
            boardWriter.add_summary('loss',loss,  global_step=step)

    def add_acc_summary(step, acc):
        if args.use_tensorboard:
            boardWriter.add_summary('acc', acc, global_step=step)
    if embedding == 'random':
        embedd_dim = args.dim
        embedd_dict = None
    else:
        embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)
    # embedd_dim = 100
    logger.info("Creating Alphabets")
    word_alphabet, char_alphabet, pos_alphabet, \
    type_alphabet = conllx_data.create_alphabets("data/alphabets/pos_crf/" + args.language + '/',
                                                 train_path, data_paths=[dev_path, test_path],
                                                 max_vocabulary_size=50000, embedd_dict=None)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    logger.info("Reading Data")
    # use_gpu = torch.cuda.is_available()

    data_train = conllx_data.read_data_to_numpy(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                type_alphabet)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])
    num_labels = pos_alphabet.size()

    data_dev = conllx_data.read_data_to_numpy(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    data_test = conllx_data.read_data_to_numpy(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if embedding == 'random':
                return None
            else:
                if word in embedd_dict:
                    a_embedding = embedd_dict[word]
                elif word.lower() in embedd_dict:
                    a_embedding = embedd_dict[word.lower()]
                else:
                    a_embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                    oov += 1
            table[index, :] = a_embedding
        print('oov: %d' % oov)
        return table

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    window = 3
    num_layers = 1
    if args.lveg:
        network = BiRecurrentConvLVeG(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters,
                                      window, mode, hidden_size, num_layers, num_labels,
                                      embedd_word=None, bigram=bigram,
                                      p_rnn=p_rnn, t_comp=args.t_comp, tag_space=256,
                                      e_comp=args.e_comp, gaussian_dim=args.gaussian_dim)
    else:
        network = BiRecurrentConvCRF(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters,
                                     window, mode, hidden_size, num_layers, num_labels, tag_space=256,
                                     embedd_word=word_table, bigram=bigram, p_rnn=p_rnn, p_in=p_in, p_out=p_out)

    network_name = type(network).__name__
    logger.info("Bulid network:" + network_name)

    lr = learning_rate
    input_word_ph = tf.placeholder(dtype=tf.int32, shape=(None, None), name='word_dim')
    input_char_ph = tf.placeholder(dtype=tf.int32, shape=(None, None, None), name='char_dim')
    target_ph = tf.placeholder(dtype=tf.int32, shape=(None, None), name='golden_sequence')
    mask_ph = tf.placeholder(dtype=tf.float32, shape=(None, None), name='mask')
    length_ph = tf.placeholder(dtype=tf.int32, shape=(None), name='length')

    loss = network.loss(input_word_ph, input_char_ph, target_ph, mask=mask_ph, length=length_ph)
    preds, corr = network.decode(input_word_ph, input_char_ph, target=target_ph, mask=mask_ph,
                                 length=length_ph, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

    # optim = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum,
    #                                    use_nesterov=True).minimize(loss)

    optims = []
    tmp_lr = lr
    for i in range(0, num_epochs + 1):
        if i % schedule == 0:
            tmp_lr = tmp_lr / (1.0 + i * decay_rate)

            optims.append(tf.train.MomentumOptimizer(learning_rate=tmp_lr,
                                                     momentum=momentum,
                                                     use_nesterov=True).minimize(loss))
    optim = optims[0]
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, crf=%s" % (
        mode, num_layers, hidden_size, num_filters, 'bigram' if bigram else 'unigram'))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, dropout: %.2f, unk replace: %.2f)" % (
        gamma, num_data, batch_size, p_in, unk_replace))

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0
    for epoch in range(1, num_epochs + 1):
        print('Epoch %d (%s, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
            epoch, mode, lr, decay_rate, schedule))
        train_err = 0.
        train_total = 0.

        start_time = time.time()
        num_back = 0
        for batch in range(1, num_batches + 1):
            word, char, labels, _, _, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size,
                                                                                      unk_replace=unk_replace)
            max_len = np.max(lengths) + 1
            word = word[:, :max_len]
            char = char[:, :max_len]
            labels = labels[:, :max_len]
            masks = masks[:, :max_len]
            feed_dict = {input_word_ph: word,
                         input_char_ph: char,
                         target_ph: labels,
                         length_ph: lengths,
                         mask_ph: masks}
            loss_batch = sess.run([optim, loss], feed_dict=feed_dict)[1]

            num_inst = word.shape[0]
            train_err += loss_batch * num_inst
            train_total += num_inst

            time_ave = (time.time() - start_time) / batch
            time_left = (num_batches - batch) * time_ave

            # update log
            if batch % 100 == 0:
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                    batch, num_batches, train_err / train_total, time_left)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('train: %d loss: %.4f, time: %.2fs' % (num_batches, train_err / train_total, time.time() - start_time))

        # evaluate performance on dev data
        dev_corr = 0.0
        dev_total = 0
        for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
            word, char, labels, _, _, masks, lengths = batch
            feed_dict = {input_word_ph: word,
                         input_char_ph: char,
                         target_ph: labels,
                         length_ph: lengths,
                         mask_ph: masks}
            pred, cor = sess.run([preds, corr], feed_dict=feed_dict)
            num_tokens = np.sum(lengths)
            dev_corr += cor
            dev_total += num_tokens
        print('dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total))

        if dev_correct < dev_corr:
            dev_correct = dev_corr
            best_epoch = epoch

            # evaluate on test data when better performance detected
            test_corr = 0.0
            test_total = 0
            for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                word, char, labels, _, _, masks, lengths = batch
                feed_dict = {input_word_ph: word,
                             input_char_ph: char,
                             target_ph: labels,
                             length_ph: lengths,
                             mask_ph: masks}
                pred, cor = sess.run([preds, corr], feed_dict=feed_dict)
                num_tokens = np.sum(lengths)
                test_corr += cor
                test_total += num_tokens
            test_correct = test_corr
        if dev_total != 0:
            print("best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch))
        if test_total != 0:
            print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (
                test_correct, test_total, test_correct * 100 / test_total, best_epoch))

        # if epoch % schedule == 0:
        #     lr = learning_rate / (1.0 + epoch * decay_rate)
        #     optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)

        if epoch % schedule == 0:
            tmp = epoch / schedule

            optim = optims[tmp]


if __name__ == '__main__':
    tf.set_random_seed(48)
    np.random.seed(48)
    main()
