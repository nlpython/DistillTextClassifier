import argparse
import six

def print_arguments(args, log):
    log.info('-----------  Configuration Arguments -----------')
    for arg, value in sorted(six.iteritems(vars(args))):
        log.info('%s: %s' % (arg, value))
    log.info('------------------------------------------------')

def get_parser():

    parser = argparse.ArgumentParser(description='Hyperparameters')

    parser.add_argument('--bert_path', type=str, default='./bert-base-chinese',
                        help='path of bert model')
    parser.add_argument('--data_dir', type=str, default='./data/hotel',
                        help='path of data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of train and evaluate')
    parser.add_argument('--student_num_epochs', type=int, default=100,
                        help='number of student epochs')
    parser.add_argument('--teacher_num_epochs', type=int, default=3,
                        help='number of teacher epochs')
    parser.add_argument('--bert_lr', type=float, default=2e-5,
                        help='learning rate of bert')
    parser.add_argument('--other_lr', type=float, default=1e-3,
                        help='learning rate of other layers')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='weight decay of all layers')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--warmup_proportion', type=float, default=0.1,
                        help='proportion of training steps to perform linear learning rate warmup for')
    parser.add_argument('--adam_epsilon', type=float, default=1e-8,
                        help='epsilon of adam')

    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='embedding dimension of LstmClassifier')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='hidden dimension of LstmClassifier')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--labels', type=list, default=[0, 1],
                        help='labels of classification')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes')
    parser.add_argument('--teacher_hidden_size', type=int, default=768,
                        help='hidden size of teacher model')
    parser.add_argument('--student_hidden_size', type=int, default=512,
                        help='hidden size of student model')
    parser.add_argument('--max_seq_length', type=int, default=128,
                        help='max length of sentence')

    parser.add_argument('--temperature', type=int, default=9,
                        help='temperature of teacher model')
    parser.add_argument('--alpha', type=float, default=0.9,
                        help='alpha of teacher model')

    return parser.parse_args()

if __name__ == '__main__':
    get_parser()

