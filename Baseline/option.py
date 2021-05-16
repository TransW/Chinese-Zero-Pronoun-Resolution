# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass
from datetime import datetime

@dataclass
class ModelArgs:
    debug: bool = True
    seed: int = 2020

    # 数据路径
    train_data_path: str = './dataset/train_examples.pkl'
    test_data_path: str = './dataset/test_examples.pkl'

    # 词表路径
    vocab_path: str = './saved/vocab.pt'
    ptm_vocab_path: str = './saved/bert_base_ch/vocab.txt'


    # 词向量路径
    use_emb_matrix: bool = True
    emb_matrix_path: str = './saved/emb_matrix.pt'

    # 模型路径
    model_path: str = './saved/models/{}/'.format(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
    trained_model_path: str = None
    ptm_dir_path: str = './saved/bert_base_ch/'

    # 训练参数
    optimizer_type: str = 'adamw'
    scheduler_type: str = 'decay'


    num_epochs: int = 5
    max_steps: int = 1000
    max_steps_before_stop: int = 1000
    log_steps: int = 200
    eval_steps: int = 500

    warmup_steps: int = 0
    decay_steps: int = 10000

    gradient_accumulation_steps: int = 1

    lr: float = 1e-3
    weight_decay: float = 0
    max_grad_norm: float = 1.0
    adam_epsilon: float = 1e-8

    use_early_stop: bool = False
    use_cuda: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class CZPRArgs(ModelArgs):

    # 数据参数
    batch_size: int = 32
    max_seq_len: int = 256

    # 模型参数
    word_vocab_size: int = None

    word_emb_size: int = 300
    word_rnn_size: int = 150
    num_word_rnn_layers: int = 1
    word_attn_size: int = 150

    key_size: int = 300

    feature_size: int = 300
    rep_size: int = 512

    dp_rate: float = 0.5




if __name__ == '__main__':
    args = CZPRArgs()
    args = args.to_dict()
    print(args)