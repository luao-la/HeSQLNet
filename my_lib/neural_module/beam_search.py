#coding=utf-8
import numpy as np
import torch
import torch.nn.functional as F

def trans_beam_search(net,beam_width,length_penalty=1,dec_input_arg_name='dec_input',begin_idx=None,end_idx=None,pad_idx=0,**net_args,):
    assert dec_input_arg_name in net_args.keys()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
    batch_size=net_args[dec_input_arg_name].size(0)
    max_len=net_args[dec_input_arg_name].size(1)

    advance_out = torch.ones_like(net_args[dec_input_arg_name])*pad_idx  # (Batch,max_Len)
    advance_mean_prob = torch.zeros_like(advance_out[:,0]).float()  #(Batch,)

    # acc_prob = torch.zeros(batch_size, ).to(device)  # (Batch,)
    # pred_output = torch.zeros(batch_size, max_len).to(device)  # (Batch,max_Len)
    pred_out = net(**net_args)[:, :, 0]  # (Batch,Dim_out)
    pred_out[:,begin_idx]=-np.inf
    pred_out[:,end_idx] = -np.inf
    pred_out[:,pad_idx]=-np.inf
    if begin_idx is None and end_idx is None:
        begin_idx=pred_out.size(1)-1
        end_idx=pred_out.size(1)-2
    pred_out = F.softmax(pred_out, dim=1)  # (Batch,Dim_out)
    # print(pred_out[:2, :10])
    pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)  # (Batch,Beam)
    # print(pred_out[:2,:],pred_out_ids[:2,:])
    acc_prob = pred_out.clone() # (Batch,Beam)
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].unsqueeze(1).expand(-1,beam_width,-1).clone()    # (Batch,Beam,max_Len)
    net_args[dec_input_arg_name][:, :, 1] = pred_out_ids    # (Batch,Beam,max_Len)
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)  # (Batch*Beam,max_Len)

    for key in net_args.keys():
        if key!=dec_input_arg_name:
            size=list(net_args[key].size())
            size.insert(1,beam_width)
            net_args[key]=net_args[key].unsqueeze(1).expand(size) #(Batch,Beam,*)
            # print(size,net_args[key].size())
            size.pop(0)
            size[0]=batch_size*beam_width
            # print(size,net_args[key].size())
            net_args[key]=net_args[key].reshape(size)    #(Batch*Beam,*)
            # print(size, net_args[key].size())
    # acc_prob=acc_prob.unsqueeze(1).expand(-1,beam_width).view(-1)   # (Batch*Beam)
    # pred_output=pred_output.unsqueeze(1).expand(-1,beam_width,-1).view(-1,max_len)  # (Batch*Beam,max_Len)
    # print(max_len,net_args[dec_input_arg_name].size())

    for i in range(1,max_len):
        pred_out = net(**net_args)[:, :, i]  # (Batch*Beam,Dim_out)
        pred_out[:, begin_idx] = -np.inf
        pred_out[:, pad_idx] = -np.inf
        # if i==1:
        #     pred_out[:, end_idx] = -np.inf
        pred_out = F.softmax(pred_out, dim=1)  # (Batch*Beam,Dim_out)
        pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)  # (Batch*Beam,Beam)
        pred_out = pred_out.view(-1, beam_width * beam_width)  # (Batch,Beam*Beam)
        pred_out_ids = pred_out_ids.view(-1, beam_width * beam_width)  # (Batch,Beam*Beam)

        net_args[dec_input_arg_name] = net_args[dec_input_arg_name].unsqueeze(1).expand(-1, beam_width, -1). \
            reshape(-1, beam_width * beam_width, max_len)  # (Batch,Beam*Beam,max_Len)
        # pred_output=pred_output.unsqueeze(1).expand(-1,beam_width,-1).view(-1,beam_width*beam_width,-1) #(Batch,Beam*Beam,max_Len)
        acc_prob=acc_prob.unsqueeze(2).expand(-1,-1,beam_width).reshape(-1,beam_width*beam_width)  #(Batch,Beam*Beam)
        acc_prob=acc_prob.add(pred_out)   #(Batch,Beam*Beam)
        mean_prob=acc_prob/(i+1)**length_penalty    #(Batch,Beam*Beam)

        if i<max_len-1:
            mean_prob, topk_ids = mean_prob.topk(beam_width, largest=True, dim=1)  # (Batch,Beam)
            # print(mean_prob[0,:5])
            pred_out_id_list,dec_input_list,acc_prob_list=[],[],[]
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j,:].index_select(dim=0,index=topk_ids[j,:])    #(Beam,)
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :]) #(Beam,max_Len)
                j_acc_prob=acc_prob[j,:].index_select(dim=0,index=topk_ids[j,:])     #(Beam,)
                for k in range(beam_width):
                    if j_pred_out_ids[k].item()==end_idx:
                        j_acc_prob[k]=-np.inf
                        if mean_prob[j,k]>advance_mean_prob[j]:
                            advance_mean_prob[j]=mean_prob[j,k]
                            advance_out[j,:-1]=j_dec_input[k,1:]
                            advance_out[j,i]=j_pred_out_ids[k]
                pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                dec_input_list.append(j_dec_input.unsqueeze(0))
                acc_prob_list.append(j_acc_prob.unsqueeze(0))

            pred_out_ids=torch.cat(pred_out_id_list,dim=0)  #(Batch,Beam)
            net_args[dec_input_arg_name]=torch.cat(dec_input_list,dim=0)    #(Batch,Beam,max_Len)
            acc_prob=torch.cat(acc_prob_list,dim=0)    #(Batch,Beam)
            net_args[dec_input_arg_name][:,:,i + 1] = pred_out_ids
            net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)  #(Batch*Beam,max_Len)
            # print(acc_prob[0,:5])
            # print(net_args[dec_input_arg_name][:5,:])
        if i==max_len-1:
            mean_prob, topk_ids = mean_prob.topk(1, largest=True, dim=1)  # (Batch,1)
            # pred_out_id_list, dec_input_list = [], []
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j, :].index_select(dim=0, index=topk_ids[j, :]) #(1,)
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :]) #(1,max_Len)
                if mean_prob[j,0]>advance_mean_prob[j] or advance_mean_prob[j].item()==0.:
                    advance_out[j, :-1] = j_dec_input[0, 1:]
                    advance_out[j, -1] = j_pred_out_ids[0]
                # if advance_out[j,:].sum(dim=0)==0:
                #     print(j,advance_out)
                # pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                # dec_input_list.append(j_dec_input)
            # pred_out_ids = torch.cat(pred_out_id_list, dim=0)  # (Batch,1)
            # net_args[dec_input_arg_name] = torch.cat(dec_input_list, dim=0)  # (Batch,max_Len)
            # net_args[dec_input_arg_name]=torch.cat([net_args[dec_input_arg_name][:,1:],pred_out_ids],dim=-1)    #(Batch,max_Len)
            # print(net_args[dec_input_arg_name].size())
            # print(advance_out)
            return advance_out #(Batch,max_Len)


def rnn_beam_search(net,beam_width,length_penalty=1,
                    dec_input_arg_name='dec_input',dec_hid_arg_name='dec_hid',
                    begin_idx=None,end_idx=None,pad_idx=0,**net_args,):
    assert dec_input_arg_name in net_args.keys()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 选择GPU优先
    batch_size=net_args[dec_input_arg_name].size(0)
    max_len=net_args[dec_input_arg_name].size(1)

    advance_out = torch.ones_like(net_args[dec_input_arg_name])*pad_idx  # (Batch,max_Len)
    advance_mean_prob = torch.zeros_like(advance_out[:,0]).float()  #(Batch,)

    # acc_prob = torch.zeros(batch_size, ).to(device)  # (Batch,)
    # pred_output = torch.zeros(batch_size, max_len).to(device)  # (Batch,max_Len)
    pred_out,net_args[dec_hid_arg_name] = net(**net_args)  # (Batch,Out_dims),(Dec_RNN_layers,Batch=1,Dec_hid_dims)
    pred_out[:,begin_idx]=-np.inf
    pred_out[:,end_idx] = -np.inf
    pred_out[:,pad_idx]=-np.inf
    if begin_idx is None and end_idx is None:
        begin_idx=pred_out.size(1)-1
        end_idx=pred_out.size(1)-2
    pred_out = F.softmax(pred_out, dim=1)  # (Batch,Dim_out)
    # print(pred_out[:2, :10])
    pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)  # (Batch,Beam)
    # print(pred_out[:2,:],pred_out_ids[:2,:])
    acc_prob = pred_out.clone() # (Batch,Beam)
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].unsqueeze(1).expand(-1,beam_width,-1).clone()    # (Batch,Beam,max_Len)
    net_args[dec_input_arg_name][:, :, 1] = pred_out_ids    # (Batch,Beam,max_Len)
    net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)  # (Batch*Beam,max_Len)

    for key in net_args.keys():
        if key!=dec_input_arg_name:
            size=list(net_args[key].size())
            size.insert(1,beam_width)
            net_args[key]=net_args[key].unsqueeze(1).expand(size) #(Batch,Beam,*)
            # print(size,net_args[key].size())
            size.pop(0)
            size[0]=batch_size*beam_width
            # print(size,net_args[key].size())
            net_args[key]=net_args[key].reshape(size)    #(Batch*Beam,*)
            # print(size, net_args[key].size())
    # acc_prob=acc_prob.unsqueeze(1).expand(-1,beam_width).view(-1)   # (Batch*Beam)
    # pred_output=pred_output.unsqueeze(1).expand(-1,beam_width,-1).view(-1,max_len)  # (Batch*Beam,max_Len)
    # print(max_len,net_args[dec_input_arg_name].size())

    for i in range(1,max_len):
        pred_out,net_args[dec_hid_arg_name] = net(**net_args)  # (Batch*Beam,Dim_out)
        pred_out[:, begin_idx] = -np.inf
        pred_out[:, pad_idx] = -np.inf
        # if i==1:
        #     pred_out[:, end_idx] = -np.inf
        pred_out = F.softmax(pred_out, dim=1)  # (Batch*Beam,Dim_out)
        pred_out, pred_out_ids = pred_out.topk(beam_width, dim=1, largest=True)  # (Batch*Beam,Beam)
        pred_out = pred_out.view(-1, beam_width * beam_width)  # (Batch,Beam*Beam)
        pred_out_ids = pred_out_ids.view(-1, beam_width * beam_width)  # (Batch,Beam*Beam)

        net_args[dec_input_arg_name] = net_args[dec_input_arg_name].unsqueeze(1).expand(-1, beam_width, -1). \
            reshape(-1, beam_width * beam_width, max_len)  # (Batch,Beam*Beam,max_Len)
        # pred_output=pred_output.unsqueeze(1).expand(-1,beam_width,-1).view(-1,beam_width*beam_width,-1) #(Batch,Beam*Beam,max_Len)
        acc_prob=acc_prob.unsqueeze(2).expand(-1,-1,beam_width).reshape(-1,beam_width*beam_width)  #(Batch,Beam*Beam)
        acc_prob=acc_prob.add(pred_out)   #(Batch,Beam*Beam)
        mean_prob=acc_prob/(i+1)**length_penalty    #(Batch,Beam*Beam)

        if i<max_len-1:
            mean_prob, topk_ids = mean_prob.topk(beam_width, largest=True, dim=1)  # (Batch,Beam)
            # print(mean_prob[0,:5])
            pred_out_id_list,dec_input_list,acc_prob_list=[],[],[]
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j,:].index_select(dim=0,index=topk_ids[j,:])    #(Beam,)
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :]) #(Beam,max_Len)
                j_acc_prob=acc_prob[j,:].index_select(dim=0,index=topk_ids[j,:])     #(Beam,)
                for k in range(beam_width):
                    if j_pred_out_ids[k].item()==end_idx:
                        j_acc_prob[k]=-np.inf
                        if mean_prob[j,k]>advance_mean_prob[j]:
                            advance_mean_prob[j]=mean_prob[j,k]
                            advance_out[j,:-1]=j_dec_input[k,1:]
                            advance_out[j,i]=j_pred_out_ids[k]
                pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                dec_input_list.append(j_dec_input.unsqueeze(0))
                acc_prob_list.append(j_acc_prob.unsqueeze(0))

            pred_out_ids=torch.cat(pred_out_id_list,dim=0)  #(Batch,Beam)
            net_args[dec_input_arg_name]=torch.cat(dec_input_list,dim=0)    #(Batch,Beam,max_Len)
            acc_prob=torch.cat(acc_prob_list,dim=0)    #(Batch,Beam)
            net_args[dec_input_arg_name][:,:,i + 1] = pred_out_ids
            net_args[dec_input_arg_name]=net_args[dec_input_arg_name].view(-1,max_len)  #(Batch*Beam,max_Len)
            # print(acc_prob[0,:5])
            # print(net_args[dec_input_arg_name][:5,:])
        if i==max_len-1:
            mean_prob, topk_ids = mean_prob.topk(1, largest=True, dim=1)  # (Batch,1)
            # pred_out_id_list, dec_input_list = [], []
            for j in range(batch_size):
                j_pred_out_ids=pred_out_ids[j, :].index_select(dim=0, index=topk_ids[j, :]) #(1,)
                j_dec_input=net_args[dec_input_arg_name][j, :, :].index_select(dim=0, index=topk_ids[j, :]) #(1,max_Len)
                if mean_prob[j,0]>advance_mean_prob[j] or advance_mean_prob[j].item()==0.:
                    advance_out[j, :-1] = j_dec_input[0, 1:]
                    advance_out[j, -1] = j_pred_out_ids[0]
                # if advance_out[j,:].sum(dim=0)==0:
                #     print(j,advance_out)
                # pred_out_id_list.append(j_pred_out_ids.unsqueeze(0))
                # dec_input_list.append(j_dec_input)
            # pred_out_ids = torch.cat(pred_out_id_list, dim=0)  # (Batch,1)
            # net_args[dec_input_arg_name] = torch.cat(dec_input_list, dim=0)  # (Batch,max_Len)
            # net_args[dec_input_arg_name]=torch.cat([net_args[dec_input_arg_name][:,1:],pred_out_ids],dim=-1)    #(Batch,max_Len)
            # print(net_args[dec_input_arg_name].size())
            # print(advance_out)
            return advance_out #(Batch,max_Len)



