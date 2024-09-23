#coding=utf-8
from .base_model import BaseModel
# sys.path.append(os.path.abspath('my_lib/util'))
from my_lib.util.eval.translate_metric import get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_corp_bleu,get_meteor,get_rouge,get_cider
import torch
from torch.utils.data import DataLoader
import os
import logging
# from nltk.translate import meteor_score
import pandas as pd
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class TransSeq2Seq(BaseModel):
    def _get_log_fit_eval(self,loss,pred_tgt, gold_tgt,tgt_i2w):
        '''
        打印性能日志
        :param loss:
        :param big_step:
        :param batch_step:
        :param big_epochs:
        :param batch_epochs:
        :param pred_tgt:
        :param gold_tgt:
        :return:
        '''
        # if 'avg_v2_loss_val' not in self.__dict__:
        #     self.avg_v2_loss_val = 0.  # 一个big epoch中平均的loss
        #     self.avg_v2_eval_np = np.zeros(len(self.train_metrics), dtype=np.float32)  # 一个big epoch中平均的测试结果
        # if verbose == 1 and 'avg_v1_loss_val' not in self.__dict__:
        #     self.avg_v1_loss_val = 0.
        #     self.avg_v1_eval_np = np.zeros(len(self.train_metrics), dtype=np.float32)  # 100个batch epoch中平均的测试结果

        pred_tgt = torch.argmax(pred_tgt, dim=1)
        pred_tgt = pred_tgt.to('cpu').data.numpy()
        # out_lens = gold_tgt.sign().sum(1)  # (B) 每个序列的长度
        gold_tgt = gold_tgt.to('cpu').data.numpy()

        pred_tgts=self._tgt_ids2tokens(pred_tgt, tgt_i2w, self.tgt_end_idx)
        # for i, pred_tgt in enumerate(pred_tgts):
        #     if len(pred_tgt) == 0:
        #         pred_tgts[i] = ['.']
        #     assert len(pred_tgts[i])>0
        gold_tgts=self._tgt_ids2tokens(gold_tgt, tgt_i2w, self.tgt_end_idx)
        if not isinstance(gold_tgts[0][0], list):   #必不可少
            gold_tgts=[[seq] for seq in gold_tgts]

        eval_np=np.array([metric(pred_tgts,gold_tgts) for metric in self.train_metrics])

        # self.avg_v2_eval_np += eval_np
        # self.avg_v2_loss_val += loss.item()

        # if verbose == 1:
        #     self.avg_v1_eval_np += eval_np
        #     self.avg_v1_loss_val += loss.item()
        #
        # if batch_step == batch_epochs:
        #     log_info = '>>>>>>>>> Big epoch step:{}/{}, train loss: {}'. \
        #         format(big_step, big_epochs, self.avg_v2_loss_val / batch_epochs)
        #     self.avg_v2_eval_np /= batch_epochs
        #     for avg_eval, metric in zip(self.avg_v2_eval_np, self.train_metrics):
        #         log_info += ', average {}: {}'.format(metric.__name__, avg_eval)
        #     log_info += ' <<<<<<<<<'
        #     # logging.info(log_info)
        #     del self.avg_v2_loss_val
        #     del self.avg_v2_eval_np
        #
        # if verbose == 1 and (batch_step % 100 == 0 or batch_step == batch_epochs):
        #     epochs = 100 if batch_step % 100 == 0 else batch_epochs % 100
        #     log_info = 'Big epoch step:{}/{}, batch epoch step:{}/{}, train loss: {}'. \
        #         format(big_step, big_epochs, batch_step, batch_epochs, self.avg_v1_loss_val / epochs)
        #     self.avg_v1_eval_np /= epochs
        #     for avg_v1_eval, metric in zip(self.avg_v1_eval_np, self.train_metrics):
        #         log_info += ', (average) {}: {}'.format(metric.__name__, avg_v1_eval)
        #     # logging.info(log_info)
        #     del self.avg_v1_loss_val
        #     del self.avg_v1_eval_np
        #
        # if verbose == 0:
        #     log_info = 'Big epoch step:{}/{}, batch epoch step:{}/{}, train Loss:{}'. \
        #         format(big_step, big_epochs, batch_step, batch_epochs, loss.item())  # 日志信息
        #     for eval_val, metric in zip(eval_np, self.train_metrics):
        #         log_info += ', {}: {}'.format(metric.__name__, eval_val)
        #     # logging.info(log_info)

        log_info='train loss:{0:.5f}'.format(loss.item())
        for eval_val, metric in zip(eval_np, self.train_metrics):
            log_info += ',{0}:{1:.5f}'.format(metric.__name__, eval_val)
        return log_info

    def _tgt_ids2tokens(self,tgts, tgt_i2w, end_idx,**kwargs):
        raise NotImplementedError

    # def _src_ids2tokens(self,srcs,src_i2w,end_idx,**kwargs):
    #     raise NotImplementedError

    def _do_validation(self,
                       valid_srcs=None,
                       valid_tgts=None,
                       tgt_i2w=None,
                       last=False,
                       increase_better=True):
        '''
        根据验证集选择最好模型
        :param criterion: func，计算loss的评价函数
        # :param Dataset: class，torch的定制Dataset类
        :param valid_srcs: lists, 验证集特征
        :param valid_tgts: list, 验证集输出
        :param last: boolen, 是否为最后一次，用在训练轮次结束后，用于选出在验证集表现最好的模型做为最终模型
        :param increase_better: boolen, 根据验证集选择更好模型时，指标是越大越好还是越小越好
        :param seq_mode:序列模式，'POS'或者None为普通序列分类问题（如词性标注），'NER'为序列标注问题（可能多个span label合并），
                'WHOLE'为整个序列是否全对的分类问题,'BLEU'为文本翻译评价
        :return:
        '''
        # self.net.eval()
        # best_net_name='{}_{}_best_net.net'.format(self.model_name,self.model_id) if self.model_id is not None \
        #     else '{}_best_net.net'.format(self.model_name)
        best_net_path=os.path.join(self.model_dir, '{}_best_net.net'.format(self.model_name))
        if not last and valid_srcs is not None and valid_tgts is not None:  # 如果有验证集
            if 'best_net' not in self.__dict__:
                # self.best_net = copy.deepcopy(self.net)
                self.best_net=None
                self.worse_epochs=0
                if increase_better:
                    self.valid_eval = -np.inf
                else:
                    self.valid_eval = np.inf
            pred_tgts=self.predict(valid_srcs,tgt_i2w)   #(B,L)
            for i, pred_tgt in enumerate(pred_tgts):
                if len(pred_tgt) == 0:
                    pred_tgts[i] = ['.']
                assert len(pred_tgts[i]) > 0
            gold_tgts = self._tgt_ids2tokens(valid_tgts, tgt_i2w, self.pad_idx)
            if not isinstance(gold_tgts[0][0], list):  # 必不可少
                gold_tgts = [[seq] for seq in gold_tgts]
            #计算指标
            log_info = 'Comparison of previous and current '

            valid_eval = self.valid_metric(pred_tgts, gold_tgts)

            log_info += '{}: ({},{}) # '.format(self.valid_metric.__name__, self.valid_eval, valid_eval)

            logging.info(log_info)
            is_better = False
            if increase_better and valid_eval >= self.valid_eval:
                is_better = True
            elif not increase_better and valid_eval<= self.valid_eval:
                is_better = True

            if is_better:  # 如果表现更好
                self.worse_epochs = 0
                self.valid_eval = valid_eval
                torch.save(self.net.state_dict(),best_net_path)
                
            else:
                self.worse_epochs+=1
            return self.worse_epochs
        elif last:
            self.net.load_state_dict(torch.load(best_net_path))
            self.net.train()


    def eval(self,
             test_srcs,
             test_tgts,
             tgt_i2w,
             ):
        '''
        序列标注的模型评价
        :param test_srcs: 测试样本特征集二维数组
        :param test_tgts: 类别标签一维数组
        :param unique_outs: 不同种类的标记列表
        :param focus_labels: 关注类别列表
        :return:
        '''
        # logging.info('Evaluate %s' % self.model_name)
        pred_tgts = self.predict(test_srcs,tgt_i2w)  # 预测出的标记二维数组
        for i, pred_tgt in enumerate(pred_tgts):
            if len(pred_tgt) == 0:
                pred_tgts[i] = ['.']
            assert len(pred_tgts[i])>0
        gold_tgts = self._tgt_ids2tokens(test_tgts, tgt_i2w, self.pad_idx)
        if not isinstance(gold_tgts[0][0], list):  # 必不可少
            gold_tgts = [[seq] for seq in gold_tgts]

        eval_dic = dict()
        logging.info('Evaluate %s' % self.model_name)
        if not self.test_metrics:
            for metric in [get_meteor,get_rouge,get_corp_bleu,get_corp_bleu1,get_corp_bleu2,get_corp_bleu3,get_corp_bleu4,get_cider]:
                eval_dic[metric.__name__]={'OVERALL':metric(pred_tgts, gold_tgts)}
        else:
            for metric in self.test_metrics:
                eval_res = metric(pred_tgts,gold_tgts)
                eval_dic[metric.__name__] = dict()
                if isinstance(eval_res, float) or isinstance(eval_res, int):
                    eval_dic[metric.__name__]['OVERALL'] = eval_res
                elif isinstance(eval_res, pd.Series):  # 如果评价结果是一个Series
                    eval_dic[metric.__name__] = dict(eval_res)
        eval_df = pd.DataFrame(eval_dic)
        return eval_df

    # def eval_seq_by_lens(self,
    #                      test_srcs,
    #                      test_tgts,
    #                      eval_dir,
    #                      decrease_in=0,
    #                      src_sec_num=10,
    #                      src_min_len=0,
    #                      tgt_sec_num=10,
    #                      tgt_min_len=0,
    #                      ):
    #     logging.info('Predict the outputs of %s' % self.model_name)
    #     pred_tgts, _ = self.predict(test_srcs)  # 预测出的标记二维数组
    #     if not isinstance(test_tgts[0][0],list):
    #         test_tgts=[[test_out] for test_out in test_tgts]
    #     # if isinstance(test_tgts[0][0], list):
    #     true_out_list = [[[self.tgt_i2w[idx] for idx in (test_out_item[:test_out_item.tolist().index(0)]
    #                                                      if 0 in test_out_item else test_out_item)]
    #                       for test_out_item in test_out] for test_out in test_tgts]  # (BL-,)
    #     # else:
    #     #     true_out_list = [[[self.tgt_i2w[idx] for idx in (true_out[:true_out.tolist().index(0)]
    #     #                                                      if 0 in true_out else true_out)]]
    #     #                      for true_out in test_tgts]
    #     pred_tgt_list = [[self.tgt_i2w[idx] for idx in (pred_tgt[:pred_tgt.tolist().index(0)]
    #                                                     if 0 in pred_tgt else pred_tgt)] for pred_tgt in pred_tgts]
    #     # in_max_len = max([len(seqs[0]) for seqs in train_ins])  # 最大输入长度
    #     # out_max_len = max([len(seq) for seq in train_outs])  # 最大输出长度
    #     data_num = len(true_out_list)  # 数据数量
    #
    #     logging.info('--- Count the input lengths')
    #
    #     in_lens = [len(seqs[0]) - decrease_in for seqs in test_srcs]
    #     max_in_len=max(in_lens)
    #     interval = math.ceil(1.0*max_in_len/src_sec_num)
    #     in_lens=['({},{}]'.format(((in_len-1)//interval)*interval,((in_len-1)//interval+1)*interval) for in_len in in_lens]
    #     # in_lens=[((in_len-1)//interval+1)*interval for in_len in in_lens]
    #     in_len2out = dict()
    #     for i, in_len in enumerate(in_lens):
    #         if in_len not in in_len2out:
    #             in_len2out[in_len] = [[pred_tgt_list[i]], [true_out_list[i]], 1]
    #         else:
    #             in_len2out[in_len][0].append(pred_tgt_list[i])
    #             in_len2out[in_len][1].append(true_out_list[i])
    #             in_len2out[in_len][2] += 1
    #
    #     logging.info('--- Count the output lengths')
    #
    #     out_lenss = [list(set([len(seq) for seq in test_out_seqs])) for test_out_seqs in test_tgts]
    #     max_out_len = max([max(out_lens) for out_lens in out_lenss])
    #     interval = math.ceil(1.0*max_out_len/tgt_sec_num)
    #     out_lenss = [['({},{}]'.format(((out_len-1) // interval) * interval, ((out_len-1) // interval + 1) * interval) for out_len in
    #                out_lens] for out_lens in out_lenss]
    #     # out_lenss = [[((out_len-1) // interval + 1) * interval for out_len in
    #     #            out_lens] for out_lens in out_lenss]
    #     out_len2out = dict()
    #     for i, out_lens in enumerate(out_lenss):
    #         for out_len in out_lens:
    #             if out_len not in out_len2out:
    #                 out_len2out[out_len] = [[pred_tgt_list[i]], [true_out_list[i]], 1]
    #             else:
    #                 out_len2out[out_len][0].append(pred_tgt_list[i])
    #                 out_len2out[out_len][1].append(true_out_list[i])
    #                 out_len2out[out_len][2] += 1
    #
    #     if not os.path.exists(eval_dir):
    #         os.makedirs(eval_dir)
    #
    #     # res_paths=[os.path.join(eval_dir,'bleu_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_ins.xlsx'),
    #     #            os.path.join(eval_dir,'bleu_by_outs.xlsx'),
    #     #            os.path.join(eval_dir,'meteor_by_outs.xlsx')]
    #     # metrics=[]
    #     eval_vs_in_len_path = os.path.join(eval_dir, 'eval_vs_in_len.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, eval_vs_in_len_path))
    #     if not os.path.exists(eval_vs_in_len_path):
    #         eval_vs_in_len_df = pd.DataFrame(columns=sorted(in_len2out.keys(),key=lambda x:int(x.split(',')[0][1:])))
    #     else:
    #         eval_vs_in_len_df = pd.read_excel(eval_vs_in_len_path, index_col=0, header=0)
    #
    #     for in_len in in_len2out.keys():
    #         eval_vs_in_len_df.loc['Proportion', in_len] = 1.0 * in_len2out[in_len][2] / data_num * 100
    #         # eval_vs_in_len_df.loc['av-BLEU', in_len] = bleu_score(in_len2out[in_len][0], in_len2out[in_len][1]) * 100
    #         for weight,metric_name in zip([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.25]*4],
    #                                         ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','av-BLEU']):
    #             eval_res=bleu_score(in_len2out[in_len][0],in_len2out[in_len][1],max_n=4, weights=weight)
    #             eval_vs_in_len_df.loc[metric_name, in_len] = eval_res * 100
    #         for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
    #             # print(len(in_len2out[in_len][0]))
    #             # print(len(in_len2out[in_len][1]))
    #             # print(metric_name)
    #             eval_res, _ = metric.compute_score(in_len2out[in_len][0], in_len2out[in_len][1])
    #             eval_vs_in_len_df.loc[metric_name, in_len] = eval_res * 100
    #     eval_vs_in_len_df.to_excel(eval_vs_in_len_path)
    #
    #     eval_vs_out_len_path = os.path.join(eval_dir, 'eval_vs_out_len.xlsx')
    #     logging.info('--- Evaluate the av-BLEU of {} and save the evaluation_result by input length into {}.'.
    #                  format(self.model_name, eval_vs_out_len_path))
    #     if not os.path.exists(eval_vs_out_len_path):
    #         eval_vs_out_len_df = pd.DataFrame(columns=sorted(out_len2out.keys(),key=lambda x:int(x.split(',')[0][1:])))
    #     else:
    #         eval_vs_out_len_df = pd.read_excel(eval_vs_out_len_path, index_col=0, header=0)
    #
    #     for out_len in out_len2out.keys():
    #         eval_vs_out_len_df.loc['Proportion', out_len] = 1.0 * out_len2out[out_len][2] / data_num * 100
    #         # eval_vs_out_len_df.loc['av-BLEU', out_len] = bleu_score(out_len2out[out_len][0],out_len2out[out_len][1]) * 100
    #         for weight,metric_name in zip([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.25]*4],
    #                                         ['BLEU-1','BLEU-2','BLEU-3','BLEU-4','av-BLEU']):
    #             eval_res=bleu_score(out_len2out[out_len][0],out_len2out[out_len][1],max_n=4, weights=weight)
    #             eval_vs_out_len_df.loc[metric_name, out_len] = eval_res * 100
    #         for metric, metric_name in zip([Meteor(), Rouge(), Cider()], ['METEOR', 'ROUGE', 'CIDER']):
    #             # print('metric_name')
    #             eval_res, _ = metric.compute_score(out_len2out[out_len][0], out_len2out[out_len][1])
    #             eval_vs_out_len_df.loc[metric_name, out_len] = eval_res * 100
    #     eval_vs_out_len_df.to_excel(eval_vs_out_len_path)
    #
    #     return eval_vs_in_len_df, eval_vs_out_len_df

    def predict(self,srcs,tgt_i2w):
        raise NotImplementedError

class RNNSeq2Seq(TransSeq2Seq):
    pass
    # def predict(self,srcs,beam_width=1):
    #     '''
    #     预测样本的类别标签的接口
    #     :param srcs: 样本特征集二维数组
    #     :return: 预测出的类别标签一维数组,或值
    #     '''
    #     logging.info('Predict outputs of %s' % self.model_name)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
    #     self.net = self.net.to(device)  # 数据转移到设备,不重新赋值不行
    #     self.net.eval()
    #     dataset = self.Dataset(srcs,
    #                            in_max_len=self.in_max_len,
    #                            out_max_len=self.out_max_len,
    #                            out_begin_idx=self.out_begin_idx)
    #     data_loader = DataLoader(dataset=dataset,
    #                              batch_size=self.batch_size,
    #                              shuffle=False,
    #                         num_workers=8)
    #     if beam_width==1:
    #         pred_tgt_prob_batches = []
    #         # batch_pred_tgts = []
    #         with torch.no_grad():
    #             for batch_features, batch_out_inputs in data_loader:
    #                 batch_features = batch_features.to(device)
    #                 batch_out_inputs = batch_out_inputs.to(device)
    #                 tmp_batch_outs = self.net(batch_features, batch_out_inputs, tf_rate=0)  # [B,D,L)
    #                 pred_tgt_prob_batches.append(tmp_batch_outs.to('cpu').data.numpy())  # [(B,D,L)]
    #                 # batch_pred_tgts.append(batch_out_inputs.to('cpu').data.numpy())
    #         # pred_tgt_prob_np = np.concatenate(pred_tgt_prob_batches, axis=0)  # (B+,D,L)
    #         # pred_tgt_np = np.argmax(pred_tgt_prob_np[:, 1:, :], axis=1) + 1  # (B+,L)
    #         pred_tgt_prob_np = np.concatenate(pred_tgt_prob_batches, axis=0)[:, :, :-1]  # (B+,D,L)
    #         pred_tgt_np = np.argmax(pred_tgt_prob_np[:, 1:, :], axis=1)[:, :-1] + 1  # (B+,L)
    #         for i,pred_tgts in enumerate(pred_tgt_np):
    #             for j,out_idx in enumerate(pred_tgts):    #prefix
    #                 if out_idx==self.tgt_end_idx:
    #                     break
    #             pred_tgt_np[i,j:]=0
    #     else:
    #         pass
    #     self.net.train()
    #     return pred_tgt_np,pred_tgt_prob_np

if __name__=='__main__':
    pass