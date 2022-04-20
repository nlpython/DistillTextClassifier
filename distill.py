import torch
from torch import nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from utils.tools import seed_everything, save_model, LstmDataset
from utils.hyperParams import get_parser, print_arguments
from utils.processer import Processer
from models.bert import BertClassifier
from models.lstm import LstmClassifier
from models.loss import DistillKL
from loguru import logger
from evaluate import evaluate


def train():
    logger.add("logs/distill_train.log")

    args = get_parser()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    print_arguments(args, logger)

    processer = Processer(args)

    # load data
    train_loader = DataLoader(
        dataset=LstmDataset(processer.convert_file_to_lstm('train')),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=LstmDataset.collate_fn
    )
    valid_loader = DataLoader(
        dataset=LstmDataset(processer.convert_file_to_lstm('test')),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LstmDataset.collate_fn
    )

    args.vocab_size = len(processer.word2idx)
    t_total = len(train_loader) * args.student_num_epochs

    # define model
    student_model = LstmClassifier(args).to(device)

    teacher_model = BertClassifier.from_pretrained(args.bert_path, processer.tokenizer).eval().to(device)
    teacher_model.load_state_dict(torch.load('./checkpoints/teacher_model.pth'))
    logger.info('Teacher model loaded.')

    # define the optimizer
    parameters = student_model.named_parameters()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.other_lr},
        {'params': [p for n, p in parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.other_lr},
    ]

    warmup_step = int(t_total * args.warmup_proportion)
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-3, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                num_training_steps=t_total)

    logger.info('\n')
    logger.info("***** Running training *****")
    logger.info("  Num Epochs = {}".format(args.student_num_epochs))
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = {}".format(
        args.batch_size))
    logger.info("  Total optimization steps = {}".format(t_total))
    logger.info("  lr of encoder = {}, lr of task_layer = {}".format(
        args.bert_lr, args.other_lr))
    logger.info('\n')

    # train
    best_result = 0.0
    temperature = args.temperature
    alpha = args.alpha
    hard_loss = nn.CrossEntropyLoss()
    soft_loss = nn.MSELoss()

    for epoch in range(args.student_num_epochs):
        student_model.train()
        logger.info('\n')
        for step, batch in enumerate(train_loader):
            student_model.zero_grad()

            input, label, text = batch
            # to device
            for key in input.keys():
                if key not in ['label']:
                    input[key] = input[key].to(device)
            label = label.to(device)

            # 教师模型预测
            with torch.no_grad():
                teacher_logits = teacher_model.predict(text, device)

            # 学生模型预测
            student_logits = student_model(**input)

            # 计算损失
            student_loss = hard_loss(student_logits, label)
            p_s = F.log_softmax(student_logits / temperature, dim=1)
            p_t = F.softmax(teacher_logits / temperature, dim=1)
            distillation_loss = F.kl_div(p_s, p_t, size_average=False) * (temperature ** 2) / student_logits.shape[0]
            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 50 == 0:
                logger.info('epoch: %2d, step: %4d, loss: %6.4f' % (epoch, step, loss.item()))

        valid_loss, valid_acc = evaluate(student_model, valid_loader, nn.CrossEntropyLoss(), device)
        logger.info('epoch: %2d, valid_loss: %4f, valid_acc: %6.4f' % (epoch, valid_loss, valid_acc * 100))

        if valid_acc > best_result:
            best_result = valid_acc
            save_model(student_model, logger, './checkpoints/distill_model.pth')
        logger.info('best_result: %6.4f' % (best_result * 100))


if __name__ == '__main__':
    train()
