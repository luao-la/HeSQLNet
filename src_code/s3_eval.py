import json

from sacrebleu import corpus_bleu

from my_lib.util.eval.translate_metric import get_nltk33_sent_bleu1 as get_sent_bleu1, \
    get_nltk33_sent_bleu2 as get_sent_bleu2, \
    get_nltk33_sent_bleu3 as get_sent_bleu3, \
    get_nltk33_sent_bleu4 as get_sent_bleu4, \
    get_nltk33_sent_bleu as get_sent_bleu
# from my_lib.util.eval.translate_metric import get_sent_bleu1,get_sent_bleu2,get_sent_bleu3,get_sent_bleu4,get_sent_bleu
from my_lib.util.eval.translate_metric import get_corp_bleu1, get_corp_bleu2, get_corp_bleu3, get_corp_bleu4, \
    get_corp_bleu
from my_lib.util.eval.translate_metric import get_meteor, get_rouge, get_cider



if __name__ == '__main__':

    res_path = '../data/Spider/result/codescriber_v41a1_6_8_512.json'
    ref_path = '../data/SpiderCogent/s2t/RGT/ref.txt'
    pred_path = '../src_code/java/code_sum_41/results/predictions-SpiderCogent-2.txt'


    gold_texts = []
    pred_texts = []

    with open(ref_path, 'r') as f:
        for line in f.readlines():
            gold_text = line
            gold_texts.append(gold_text)

    with open(pred_path, 'r') as f:
        for line in f.readlines():
            pred_text = line
            pred_texts.append(pred_text)

    print(len(gold_texts))

    # with open(res_path, 'r') as f:
    #     res_data = json.load(f)
    #
    # for i, item in enumerate(res_data):
    #     gold_text = item['gold_text'][:-1]
    #     pred_text = item['pred_text'][:-1]
    #     gold_texts.append(gold_text)
    #     pred_texts.append(pred_text)

    preds = []
    refs = []

    pre = ""

    for index, pred in enumerate(pred_texts):
        if pred == pre:
            refs[-1].append(gold_texts[index])
        else:
            preds.append(pred)
            pre = pred
            refs.append([gold_texts[index]])

    ref1 = []
    ref2 = []

    for ref in refs:
        ref1.append(ref[0])
        if len(ref) > 1:
            ref2.append(ref[1])
        else:
            ref2.append(ref[0])

    pred_texts = preds
    gold_texts = [ref1, ref2]

    print(corpus_bleu(pred_texts, gold_texts, force=True,
                       lowercase=True).score)

    # print(get_corp_bleu.__name__,':',get_corp_bleu(pred_texts,gold_texts))
    # print(get_sent_bleu.__name__,':',get_sent_bleu(pred_texts,gold_texts))
    # print(get_meteor.__name__,':',get_meteor(pred_texts,gold_texts))
    # print(get_rouge.__name__,':',get_rouge(pred_texts,gold_texts))
    # print(get_sent_bleu1.__name__,':',get_sent_bleu1(pred_texts,gold_texts))
    # print(get_sent_bleu2.__name__,':',get_sent_bleu2(pred_texts,gold_texts))
    # print(get_sent_bleu3.__name__,':',get_sent_bleu3(pred_texts,gold_texts))
    # print(get_sent_bleu4.__name__,':',get_sent_bleu4(pred_texts,gold_texts))
