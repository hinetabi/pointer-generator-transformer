''' Translate input text with trained model. '''

import torch
import argparse
import dill as pickle
from tqdm import tqdm

import transformer.Constants as Constants
from torchtext.legacy.data import Dataset
from torchtext.legacy.datasets import TranslationDataset
from torchtext.legacy.data import Field, Dataset, BucketIterator

from transformer.Models import Transformer
from transformer.Translator import Translator


def load_model(opt, device):

    checkpoint = torch.load(opt.model, map_location=device)
    model_opt = checkpoint['settings']

    model = Transformer(
        model_opt.src_vocab_size,
        model_opt.trg_vocab_size,

        model_opt.src_pad_idx,
        model_opt.trg_pad_idx,

        trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
        emb_src_trg_weight_sharing=model_opt.embs_share_weight,
        d_k=model_opt.d_k,
        d_v=model_opt.d_v,
        d_model=model_opt.d_model,
        d_word_vec=model_opt.d_word_vec,
        d_inner=model_opt.d_inner_hid,
        n_layers=model_opt.n_layers,
        n_head=model_opt.n_head,
        dropout=model_opt.dropout).to(device)

    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model 

# def prepare_dataloaders_from_bpe_files(opt, device):
#     batch_size = opt.batch_size
#     MIN_FREQ = 2
#     # if not opt.embs_share_weight:
#     #     raise

#     data = pickle.load(open(opt.vocab, 'rb'))
#     MAX_LEN = data['settings'].max_len
#     field = data['vocab']
#     fields = (field, field)

#     def filter_examples_with_length(x):
#         return len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN

#     test = TranslationDataset(
#         fields=fields,
#         path=opt.src, 
#         exts=('.src', '.trg'),
#         filter_pred=filter_examples_with_length)

#     opt.max_token_seq_len = MAX_LEN + 2
#     opt.src_pad_idx = opt.trg_pad_idx = field.vocab.stoi[Constants.PAD_WORD]
#     opt.src_bos_idx = opt.trg_bos_idx = field.vocab.stoi[Constants.BOS_WORD]
#     opt.src_vocab_size = opt.trg_vocab_size = len(field.vocab)
#     opt.unk_idx = field.vocab.stoi[Constants.UNK_WORD]

#     test_iterator = BucketIterator(test, batch_size=batch_size, device=device, train=True)
#     return test_iterator, data

def load_dataset(opt, SRC, TRG, MAX_LEN):    
    data = []
    with open(opt.src, 'r') as f:
        filedata = f.readlines()
        for sent in filedata:
            if len(sent) < MAX_LEN:
                sent = sent.replace("@", "")
                data.append(sent.split())
                
    test_loader = Dataset(examples=data, fields={'src': SRC, 'trg': TRG})
    return test_loader

def main():
    '''
    Usage: python translate.py -src '/home/hinetabi/Documents/university/data/sample/processed/v1-train.src' -model output/model.chkpt -vocab /home/hinetabi/Documents/university/data/sample/processed/multi30k_de_en.pkl
    '''
    parser = argparse.ArgumentParser(description='translate.py')

    parser.add_argument('-model', required=True,
                        help='Path to model weight file')
    # parser.add_argument('-data_pkl', required=True,
    #                     help='Pickle file with both instances and vocabulary.')
    parser.add_argument('-output', default='pred.txt',
                        help="""Path to output the predictions (each line will
                        be the decoded sequence""")
    parser.add_argument('-beam_size', type=int, default=5)
    parser.add_argument('-max_seq_len', type=int, default=100)

    parser.add_argument('-src', required=True,
                       help='Source sequence to decode (one line per sequence)')
    parser.add_argument('-vocab', required=True,
                       help='vocab path')

    # TODO: Translate bpe encoded files 
    # parser.add_argument('-enc_src', required=True,
    #                    help='Source sequence to decode (one line per sequence)')
    # TODO: Batch translation
    parser.add_argument('-batch_size', type=int, default=4,
                       help='Batch size')
    #parser.add_argument('-n_best', type=int, default=1,
    #                    help="""If verbose is set, will output the n_best
    #                    decoded sentences""")

    opt = parser.parse_args()

    data = pickle.load(open(opt.vocab, 'rb'))
    
    SRC, TRG = data['vocab'], data['vocab']
    MAX_LEN = data['settings'].max_len
    opt.src_pad_idx = SRC.vocab.stoi[Constants.PAD_WORD]
    opt.trg_pad_idx = TRG.vocab.stoi[Constants.PAD_WORD]
    opt.trg_bos_idx = TRG.vocab.stoi[Constants.BOS_WORD]
    opt.trg_eos_idx = TRG.vocab.stoi[Constants.EOS_WORD]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # test_loader = Dataset(examples=data['test'], fields={'src': SRC, 'trg': TRG})
    test_loader = load_dataset(opt, SRC, TRG, MAX_LEN)

    translator = Translator(
        model=load_model(opt, device),
        beam_size=opt.beam_size,
        max_seq_len=opt.max_seq_len,
        src_pad_idx=opt.src_pad_idx,
        trg_pad_idx=opt.trg_pad_idx,
        trg_bos_idx=opt.trg_bos_idx,
        trg_eos_idx=opt.trg_eos_idx).to(device)

    unk_idx = SRC.vocab.stoi[SRC.unk_token]
    print(f"unk idx = {unk_idx}")
    with open(opt.output, 'w') as f:
        for example in tqdm(test_loader, mininterval=2, desc='  - (Test)', leave=False):
            #print(' '.join(example.src))
            src_seq = [SRC.vocab.stoi.get(word, unk_idx) for word in example]
            print("source tensor === ", src_seq)
            pred_seq = translator.translate_sentence(torch.LongTensor([src_seq]).to(device))
            pred_line = ' '.join(TRG.vocab.itos[idx] for idx in pred_seq)
            print(pred_line)
            pred_line = pred_line.replace(Constants.BOS_WORD, '').replace(Constants.EOS_WORD, '')

            f.write(pred_line.strip() + '\n')

    print('[Info] Finished.')

if __name__ == "__main__":
    main()
