#coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DualCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx',0)
        self.pad_idx=kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.copy_attention1=CrossAttention(query_dims=tgt_dims,
                                                   key_dims=src_dims,
                                                   drop_rate=drop_rate,
                                                   pad_idx=kwargs['pad_idx']
                                                   )
        self.copy_attention2 = CrossAttention(query_dims=tgt_dims,
                                                      key_dims=src_dims,
                                                      drop_rate=drop_rate,
                                                      pad_idx=kwargs['pad_idx']
                                                      )
        self.linear=nn.Linear(2*src_dims+tgt_dims,3)
        self.p_softmax=nn.Softmax(dim=-1)

    def forward(self,tgt_dec_out,
                src1_key,src1_map_idx,
                src2_key,src2_map_idx):
        '''

        :param tgt_dec_out:
        :param src_key:
        :param src_map_idx: (B,L_src)
        :return:
        '''
        #先求出本身的tgt_output并softmax求出概率分布，再填充0
        tgt_output = self.out_fc(tgt_dec_out)  # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        # tgt_output=self.tgt_softmax(tgt_output)    # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        src1_len,src2_len=src1_key.size(1),src2_key.size(1)
        tgt_output=F.pad(tgt_output,(0,src1_len+src2_len),value=self.pad_idx) #pad last dim (B,L_tgt,tgt_voc_size+2*L_src)

        #利用multiheadattention求出softmax后的attention并根据src_map映射到copy_output中
        tgt_mask = tgt_dec_out.abs().sum(-1).sign()  # (B,L_tgt)

        src1_mask = src1_key.abs().sum(-1).sign()  # (B,L_src1)
        att1,c1=self.copy_attention1(query=tgt_dec_out,
                                     key=src1_key,
                                     query_mask=tgt_mask,
                                     key_mask=src1_mask)    #(B,L_tgt,L_src1),(B,L_tgt,D_tgt)
        att1 = F.layer_norm(att1, (att1.size(-1),))
        copy_output1=torch.zeros_like(tgt_output) #(B,L_tgt,tgt_voc_size+2*L_tgt)
        src1_map=src1_map_idx.unsqueeze(dim=1).expand(-1,att1.size(1),-1)  #(B,L_tgt,L_src1)
        # indices2 = src_map.flatten()
        #利用meshgrid得到前两维度的所有映射索引
        indices0,indices1,_=torch.meshgrid(torch.arange(att1.size(0)),torch.arange(att1.size(1)),torch.arange(att1.size(2)))
        indices=(indices0.flatten(),indices1.flatten(),src1_map.flatten())   #copy_output的索引位置
        #将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        copy_output1.index_put_(indices=indices, values=att1.flatten(), accumulate=True)  #(B,L_tgt,tgt_voc_size+2*L_tgt)

        src2_mask = src2_key.abs().sum(-1).sign()  # (B,L_src2)
        att2, c2 = self.copy_attention2(query=tgt_dec_out,
                                        key=src2_key,
                                        query_mask=tgt_mask,
                                        key_mask=src2_mask)  # (B,L_tgt,L_src2),(B,L_tgt,D_tgt)
        att2 = F.layer_norm(att2, (att2.size(-1),))
        copy_output2 = torch.zeros_like(tgt_output)  # (B,L_tgt,tgt_voc_size+2*L_tgt)
        src2_map = src2_map_idx.unsqueeze(dim=1).expand(-1, att2.size(1), -1)  # (B,L_tgt,L_src1)
        # indices2 = src_map.flatten()
        # 利用meshgrid得到前两维度的所有映射索引
        indices0, indices1, _ = torch.meshgrid(torch.arange(att2.size(0)), torch.arange(att2.size(1)),
                                               torch.arange(att2.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src2_map.flatten())  # copy_output的索引位置
        # 将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        copy_output2.index_put_(indices=indices, values=att2.flatten(),
                                accumulate=True)  # (B,L_tgt,tgt_voc_size+2*L_tgt)

        #利用c和tgt_dec_out，生成p
        p=F.softmax(self.linear(torch.cat([tgt_dec_out,c1,c2],dim=-1)),dim=-1)    #(B,L_tgt,3)
        p=p.unsqueeze(2).expand(-1,-1,tgt_output.size(2),-1)   ##(B,L_tgt,tgt_voc_size+2*L_tgt,3)

        output=(tgt_output.mul(p[:,:,:,0])).add(copy_output1.mul(p[:,:,:,1])).add(copy_output2.mul(p[:,:,:,2]))   #(B,L_tgt,tgt_voc_size+2*L_tgt)
        # print(output.sum(dim=-1))
        # print(tgt_output.sum(dim=-1))
        # copy1=copy_output1.sum(dim=-1)
        # copy2=copy_output2.sum(dim=-1)
        return output

class CopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx',0)
        self.pad_idx=kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.copy_attention=CrossAttention(query_dims=tgt_dims,
                                           key_dims=src_dims,
                                           drop_rate=drop_rate,
                                           pad_idx=kwargs['pad_idx']
                                           )
        self.tgt_softmax = nn.Softmax(dim=-1)
        # self.linear=nn.Linear(2*tgt_dims,2)
        self.linear=nn.Linear(src_dims,1)
        self.p_softmax=nn.Softmax(dim=-1)
        # print(tgt_dims)

    def forward(self,tgt_dec_out,src_key,src_map_idx):
        '''

        :param tgt_dec_out:
        :param src_key:
        :param src_map_idx: (B,L_src)
        :return:
        '''
        #先求出本身的tgt_output并softmax求出概率分布，再填充0
        tgt_output = self.out_fc(tgt_dec_out)  # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output=F.layer_norm(tgt_output,(tgt_output.size(-1),))
        # tgt_output=self.tgt_softmax(tgt_output)    # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output=F.pad(tgt_output,(0,src_key.size(1)),value=self.pad_idx) #pad last dim (B,L_tgt,tgt_voc_size+L_tgt)

        #利用multiheadattention求出softmax后的attention并根据src_map映射到copy_output中
        tgt_mask = tgt_dec_out.abs().sum(-1).sign()  # (B,L_tgt)
        src_mask = src_key.abs().sum(-1).sign()  # (B,L_src)
        att,c=self.copy_attention(query=tgt_dec_out,
                                  key=src_key,
                                  query_mask=tgt_mask,
                                  key_mask=src_mask)    #(B,L_tgt,L_src),(B,L_tgt,D_src)
        # import time
        # print('*'*20,c.size())
        # time.sleep(10)
        # p = torch.sigmoid(self.linear(c))  # (B,L_tgt,1)
        att=F.layer_norm(att,(att.size(-1),))
        copy_output=torch.zeros_like(tgt_output) #(B,L_tgt,tgt_voc_size+L_tgt)
        # print(src_map_idx.size())
        src_map=src_map_idx.unsqueeze(dim=1).expand(-1,att.size(1),-1)  #(B,L_tgt,L_src)
        # indices2 = src_map.flatten()
        #利用meshgrid得到前两维度的所有映射索引
        indices0,indices1,_=torch.meshgrid(torch.arange(att.size(0)),torch.arange(att.size(1)),torch.arange(att.size(2)))
        indices=(indices0.flatten(),indices1.flatten(),src_map.flatten())   #copy_output的索引位置
        #将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        # print(copy_output.size(),att.size(),src_map.size())
        copy_output.index_put_(indices=indices, values=att.flatten(), accumulate=True)  #(B,L_tgt,tgt_voc_size+L_tgt)

        # #利用c和tgt_dec_out，生成p
        # # p=self.linear(torch.cat([tgt_dec_out,c],dim=-1))    #(B,L_tgt,2)
        # p=self.linear(torch.cat([tgt_dec_out,c],dim=-1))    #(B,L_tgt,2)
        # p=p.unsqueeze(2).expand(-1,-1,copy_output.size(2),-1)   ##(B,L_tgt,tgt_voc_size+L_tgt,2)
        #
        # output=(tgt_output.mul(p[:,:,:,0])).add(copy_output.mul(p[:,:,:,1]))   #(B,L_tgt,tgt_voc_size+L_tgt)

        p = torch.sigmoid(self.linear(c))  # (B,L_tgt,1)
        p = p.expand(-1, -1, copy_output.size(2))  ##(B,L_tgt,tgt_voc_size+L_tgt)
        output = (tgt_output.mul(p)).add(copy_output.mul(1-p))  # (B,L_tgt,tgt_voc_size+L_tgt)
        return output

class CrossAttention(nn.Module):
    def __init__(self,query_dims, key_dims,drop_rate=0.0,**kwargs):
        '''

        :param query_dim: 模型输出端（解码器端）的RNN隐藏层维度
        :param key_dim: 模型输入端（编码器端）的输出维度
        '''
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.attn = nn.Linear(key_dims + query_dims, query_dims)
        self.v = nn.Linear(query_dims, 1, bias=False)
        self.dropout = nn.Dropout(drop_rate)
        self.key_dims=key_dims
        # print(key_dims)

    def forward(self, query, key,query_mask,key_mask):
        '''

        :param query: 模型输出端（解码器端）的RNN隐藏层输出 (batch_size,len_key,dim_query)
        :param key: 模型输入端（编码器端）的输出 (batch_size,len_query,dim_key)
        :return:
        '''
        key_len = key.shape[1]
        query_len=query.shape[1]
        # repeat decoder hidden state key_len times
        query = query.unsqueeze(2).expand(-1, -1,key_len, -1)   #(batch_size,len_query,len_key,dim_query)
        key_=key.unsqueeze(1).expand(-1,query_len,-1,-1)    #(batch_size,len_query,len_key,dim_key)

        energy = torch.tanh(self.attn(torch.cat((query, key_), dim=-1))) #(batch_size,len_query,len_key,dim_query)
        del key_
        attention = self.v(energy).squeeze(-1)  #(batch_size,len_query,len_key)

        del energy
        out_att=attention.clone()
        if key_mask is not None:
            key_mask_ = key_mask.eq(self.pad_idx)  # (B,L_k)
            key_mask_ = key_mask_.unsqueeze(dim=1).expand(-1, query_len, -1)  # (B,L_q,L_k)
            attention = attention.masked_fill(key_mask_, -np.inf)  # (B,L_q,L_k)

            out_att = out_att.mul(key_mask[:, None, :].expand(-1, query.size(1), -1).float())   # (B,L_q,L_k)
        # print(attention.size())
        attention=F.softmax(attention,dim=-1) # (B,L_q,L_k)
        # print(attention.size())
        # print(key.size())
        weight = torch.bmm(attention, key)   #(batch_size,L_q, dim_key]
        # print(weight.size())
        weight=self.dropout(weight) #(batch_size,L_q, dim_key]
        del attention
        if query_mask is not None:
            # print(query_mask.size())
            query_mask_=query_mask[:,:,None].repeat(1,1,self.key_dims)  #(B,L_q,D_key)
            # print(query_mask_.size())
            # print(weight.size())
            weight=weight.mul(query_mask_.float()) #(B,L_q,D)
            out_att=out_att.mul(query_mask[:,:,None].expand(-1,-1,key.size(1)).float())
        # print(out_att.size(),weight.size())
        return out_att,weight   # (B,L_q,L_k),(batch_size,L_q, dim_key]

class DualMultiCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 att_heads=8,
                 att_head_dims=None,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx',0)
        self.pad_idx=kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.tgt_softmax = nn.Softmax(dim=-1)
        self.copy_attention1=MultiHeadCopyAttention(query_dims=tgt_dims,
                                                   key_dims=src_dims,
                                                   head_num=att_heads,
                                                   head_dims=att_head_dims,
                                                   drop_rate=drop_rate,
                                                   pad_idx=kwargs['pad_idx']
                                                   )
        self.copy_attention2 = MultiHeadCopyAttention(query_dims=tgt_dims,
                                                      key_dims=src_dims,
                                                      head_num=att_heads,
                                                      head_dims=att_head_dims,
                                                      drop_rate=drop_rate,
                                                      pad_idx=kwargs['pad_idx']
                                                      )
        self.linear=nn.Linear(3*tgt_dims,3)
        self.p_softmax=nn.Softmax(dim=-1)

    def forward(self,tgt_dec_out,
                src1_key,src1_map_idx,
                src2_key,src2_map_idx):
        '''

        :param tgt_dec_out:
        :param src_key:
        :param src_map_idx: (B,L_src)
        :return:
        '''
        #先求出本身的tgt_output并softmax求出概率分布，再填充0
        tgt_output = self.out_fc(tgt_dec_out)  # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        # tgt_output=self.tgt_softmax(tgt_output)    # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output = F.layer_norm(tgt_output, (tgt_output.size(-1),))
        src1_len,src2_len=src1_key.size(1),src2_key.size(1)
        tgt_output=F.pad(tgt_output,(0,src1_len+src2_len),value=self.pad_idx) #pad last dim (B,L_tgt,tgt_voc_size+2*L_src)

        #利用multiheadattention求出softmax后的attention并根据src_map映射到copy_output中
        tgt_mask = tgt_dec_out.abs().sum(-1).sign()  # (B,L_tgt)

        src1_mask = src1_key.abs().sum(-1).sign()  # (B,L_src1)
        att1,c1=self.copy_attention1(query=tgt_dec_out,
                                     key=src1_key,
                                     query_mask=tgt_mask,
                                     key_mask=src1_mask)    #(B,L_tgt,L_src1),(B,L_tgt,D_tgt)
        att1 = F.layer_norm(att1, (att1.size(-1),))
        copy_output1=torch.zeros_like(tgt_output) #(B,L_tgt,tgt_voc_size+2*L_tgt)
        src1_map=src1_map_idx.unsqueeze(dim=1).expand(-1,att1.size(1),-1)  #(B,L_tgt,L_src1)
        # indices2 = src_map.flatten()
        #利用meshgrid得到前两维度的所有映射索引
        indices0,indices1,_=torch.meshgrid(torch.arange(att1.size(0)),torch.arange(att1.size(1)),
                                           torch.arange(att1.size(2)))
        indices=(indices0.flatten(),indices1.flatten(),src1_map.flatten())   #copy_output的索引位置
        #将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        copy_output1.index_put_(indices=indices, values=att1.flatten(), accumulate=True)  #(B,L_tgt,tgt_voc_size+2*L_tgt)

        src2_mask = src2_key.abs().sum(-1).sign()  # (B,L_src2)
        att2, c2 = self.copy_attention2(query=tgt_dec_out,
                                        key=src2_key,
                                        query_mask=tgt_mask,
                                        key_mask=src2_mask)  # (B,L_tgt,L_src2),(B,L_tgt,D_tgt)
        att2 = F.layer_norm(att2, (att2.size(-1),))
        copy_output2 = torch.zeros_like(tgt_output)  # (B,L_tgt,tgt_voc_size+2*L_tgt)
        src2_map = src2_map_idx.unsqueeze(dim=1).expand(-1, att2.size(1), -1)  # (B,L_tgt,L_src1)
        # indices2 = src_map.flatten()
        # 利用meshgrid得到前两维度的所有映射索引
        indices0, indices1, _ = torch.meshgrid(torch.arange(att2.size(0)), torch.arange(att2.size(1)),
                                               torch.arange(att2.size(2)))
        indices = (indices0.flatten(), indices1.flatten(), src2_map.flatten())  # copy_output的索引位置
        # 将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        copy_output2.index_put_(indices=indices, values=att2.flatten(),
                                accumulate=True)  # (B,L_tgt,tgt_voc_size+2*L_tgt)

        #利用c和tgt_dec_out，生成p
        p=F.softmax(self.linear(torch.cat([tgt_dec_out,c1,c2],dim=-1)),dim=-1)    #(B,L_tgt,3)
        p=p.unsqueeze(2).expand(-1,-1,tgt_output.size(2),-1)   ##(B,L_tgt,tgt_voc_size+2*L_tgt,3)

        output=(tgt_output.mul(p[:,:,:,0])).add(copy_output1.mul(p[:,:,:,1])).add(copy_output2.mul(p[:,:,:,2]))   #(B,L_tgt,tgt_voc_size+2*L_tgt)
        # print(output.sum(dim=-1))
        # print(tgt_output.sum(dim=-1))
        # copy1=copy_output1.sum(dim=-1)
        # copy2=copy_output2.sum(dim=-1)
        return output

class MultiCopyGenerator(nn.Module):
    def __init__(self,
                 tgt_dims,
                 tgt_voc_size,
                 src_dims,
                 att_heads=8,
                 att_head_dims=None,
                 drop_rate=0.,
                 **kwargs):
        super().__init__()
        kwargs.setdefault('pad_idx',0)
        self.pad_idx=kwargs['pad_idx']
        self.out_fc = nn.Linear(tgt_dims, tgt_voc_size)
        self.copy_attention=MultiHeadCopyAttention(query_dims=tgt_dims,
                                                   key_dims=src_dims,
                                                   head_num=att_heads,
                                                   head_dims=att_head_dims,
                                                   drop_rate=drop_rate,
                                                   pad_idx=kwargs['pad_idx']
                                                   )
        self.tgt_softmax = nn.Softmax(dim=-1)
        # self.linear=nn.Linear(2*tgt_dims,2)
        self.linear=nn.Linear(tgt_dims,1)
        self.p_softmax=nn.Softmax(dim=-1)

    def forward(self,tgt_dec_out,src_key,src_map_idx):
        '''

        :param tgt_dec_out:
        :param src_key:
        :param src_map_idx: (B,L_src)
        :return:
        '''
        #先求出本身的tgt_output并softmax求出概率分布，再填充0
        tgt_output = self.out_fc(tgt_dec_out)  # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output=F.layer_norm(tgt_output,(tgt_output.size(-1),))
        # tgt_output=self.tgt_softmax(tgt_output)    # (B,L_tgt,tgt_voc_size)包含begin_idx和end_idx
        tgt_output=F.pad(tgt_output,(0,src_key.size(1)),value=self.pad_idx) #pad last dim (B,L_tgt,tgt_voc_size+L_src)

        #利用multiheadattention求出softmax后的attention并根据src_map映射到copy_output中
        tgt_mask = tgt_dec_out.abs().sum(-1).sign()  # (B,L_tgt)
        src_mask = src_key.abs().sum(-1).sign()  # (B,L_src)
        att,c=self.copy_attention(query=tgt_dec_out,
                                  key=src_key,
                                  query_mask=tgt_mask,
                                  key_mask=src_mask)    #(B,L_tgt,L_src),(B,L_tgt,D_tgt)
        att=F.layer_norm(att,(att.size(-1),))
        copy_output=torch.zeros_like(tgt_output) #(B,L_tgt,tgt_voc_size+L_tgt)
        src_map=src_map_idx.unsqueeze(dim=1).expand(-1,att.size(1),-1)  #(B,L_tgt,L_src)
        # indices2 = src_map.flatten()
        #利用meshgrid得到前两维度的所有映射索引
        indices0,indices1,_=torch.meshgrid(torch.arange(att.size(0)),torch.arange(att.size(1)),torch.arange(att.size(2)))
        indices=(indices0.flatten(),indices1.flatten(),src_map.flatten())   #copy_output的索引位置
        #将指定的索引位置以累计方式替换为对应的att值，即为copy_output
        copy_output.index_put_(indices=indices, values=att.flatten(), accumulate=True)  #(B,L_tgt,tgt_voc_size+L_tgt)

        # #利用c和tgt_dec_out，生成p
        # # p=self.linear(torch.cat([tgt_dec_out,c],dim=-1))    #(B,L_tgt,2)
        # p=self.linear(torch.cat([tgt_dec_out,c],dim=-1))    #(B,L_tgt,2)
        # p=p.unsqueeze(2).expand(-1,-1,copy_output.size(2),-1)   ##(B,L_tgt,tgt_voc_size+L_tgt,2)
        #
        # output=(tgt_output.mul(p[:,:,:,0])).add(copy_output.mul(p[:,:,:,1]))   #(B,L_tgt,tgt_voc_size+L_tgt)
        p = torch.sigmoid(self.linear(c))  # (B,L_tgt,1)
        p = p.expand(-1, -1, copy_output.size(2))  ##(B,L_tgt,tgt_voc_size+L_tgt)
        output = (tgt_output.mul(p)).add(copy_output.mul(1-p))  # (B,L_tgt,tgt_voc_size+L_tgt)
        return output

class MultiHeadCopyAttention(nn.Module):
    def __init__(self,
                 query_dims=512,
                 key_dims=None,
                 head_num=8,
                 head_dims=None,
                 drop_rate=0.,
                 **kwargs
                 ):
        '''
        init    和multiheadattention一样，只是多返回一个attention值
        :param unit_num: A scalar. Attention size.
        :param head_num: An int. Number of heads.
        :param drop_rate: A floating point number.
        :param head_mode: An int. head模式，0表示标准transformer头，1表示共享Linear层transformer头，2表示共享卷积层transformer头
        :param residual: Boolean. 是否加入残差
        :param norm: Boolean. 是否归一化
        :param causality: Boolean. If true, units that reference the future are masked.
        '''
        super().__init__()
        kwargs.setdefault('pad_idx', 0)
        self.pad_idx = kwargs['pad_idx']
        self.query_dims = query_dims
        self.key_dims = query_dims if key_dims is None else key_dims
        self.head_num=head_num
        self.head_dims = query_dims // head_num if head_dims is None else head_dims
        self.hid_dims=self.head_num*self.head_dims
        self.conv1d_ins=nn.ModuleList([nn.Conv1d(io_dims, self.hid_dims, kernel_size=1,padding=0)
                                    for io_dims in [self.query_dims,self.key_dims,self.key_dims]])

        self.conv1d_out=nn.Conv1d(self.hid_dims,self.query_dims, kernel_size=1,padding=0)

        self.softmax=nn.Softmax(dim=-1)
        self.dropout=nn.Dropout(drop_rate)

    def forward(self, query,key,query_mask,key_mask,value=None):
        '''
        Applies multihead attention
        :param query: A 3d tensor with shape of [B, T_q, D_q]
        :param key: A 3d tensor with shape of [B, T_k, D_k]
        :param query_mask: A 3d tensor with shape of [B, T_q]
        :param key_mask: A 3d tensor with shape of [B, T_k]
        :return: A 3d tensor with shape of [B, T_q, D]
        '''
        if value is None:
            value=key.clone()   #深度拷贝
        batch_size=query.size(0)    #B

        #Linear
        query_,key_,value_=[conv1d_in(x.transpose(1,2)) for conv1d_in,x in
                            zip(self.conv1d_ins,(query,key,value))]  #(B,D,L_q),(B,D,L_k),(B,D,L_v)
        query_, key_, value_ = [x.view(batch_size, self.head_num, self.head_dims, -1).transpose(2, 3)
                                    for x in (query_, key_, value_)]  # (B,h,L_q,D/h)(B,h,L_k,D/h)(B,h,L_v,D/h)

        #Multiplication
        # attention=torch.bmm(query_,key_.transpose(1,2).contiguous())   #(B*h,L_q,L_k)
        query_=query_.mul(float(self.head_dims)**-0.5)        #Scale
        attention=torch.einsum('abcd,abed->abce',query_,key_)  #(B,h,L_q,L_k)

        out_att=attention.clone()
        if key_mask is not None:
            # value_mask = key_mask[:, None, :, None].expand(-1, self.head_num, -1, self.head_dims)  # (B,h,L_v,D/h)
            key_mask_=key_mask.eq(self.pad_idx) #(B,L_k)
            key_mask_=key_mask_.unsqueeze(dim=1).repeat(1,self.head_num,1)   #(B,h,L_k)
            key_mask_=key_mask_.unsqueeze(dim=2).expand(-1,-1,query.size(1),-1)    #(B,h,L_q,L_k)
            attention=attention.masked_fill(key_mask_,-np.inf)   #(B,h,L_q,L_k)

            out_att=out_att.mul(key_mask[:,None,None,:].expand(-1,self.head_num,query.size(1),-1).float())

        attention = self.softmax(attention)  # (B,h,L_q,L_k)

        #Dropouts
        attention=self.dropout(attention)   #(B,h,L_q,L_k)

        # Weighted sum
        output=torch.matmul(attention,value_)  #(B,h,L_q,D/h)
        output=output.transpose(1,2).contiguous().view(batch_size,-1,self.hid_dims)    #(B,L_q,D)
        output=self.conv1d_out(output.transpose(1,2)).transpose(1,2)  #(B,L_q,D)

        if query_mask is not None:
            query_mask_=query_mask[:,:,None].repeat(1,1,self.query_dims)  #(B,L_q,D)
            output=output.mul(query_mask_.float()) #(B,L_q,D)
            out_att=out_att.mul(query_mask[:,None,:,None].expand(-1,self.head_num,-1,key.size(1)).float())

        #attention的形状是(B,h,L_q,L_k),使用对第一维度平均去掉h
        return out_att.mean(dim=1),output #(B,L_q,L_k) #(B,L_q,D)