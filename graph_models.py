import torch
import torch.nn as nn
from configs import EncoderConfig, Type3Config, Type4Config, Type12Config
from layers import GraphConvolution
from torch.nn import functional as F
from typing import List
import sys



sys.path.insert(0, "../")
import scgpt as scg
from scgpt.model.model_default import TransformerModel

dataset_name="ms"


class scGPTForAnnotation(nn.Module):
    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.c=config  
        model=torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{self.c.dataset_name}_median/model.pt")
        model.load_state_dict(torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save_scgcn/scgpt_{self.c.dataset_name}_median/model_ckpt.pt"))
        
        self.transformer= model
        

    def forward(self,src, values, src_key_padding_mask, batch_labels=None, cls=True):
         with torch.cuda.amp.autocast(enabled=True): # In case of GPU Usage
             output_dict = self.transformer(src, values, src_key_padding_mask, batch_labels, cls)
         return output_dict

class Type12(nn.Module):
    def __init__(self, config: Type12Config, adj_list):
        super().__init__()
        self.c = config
        hidden_sizes = [self.c.fan_mid // (2 ** i) for i in range(len(adj_list))]
        
        self.first_graph_conv_layer = GraphConvolution(self.c.fan_in, hidden_sizes[0])
        self.graph_conv_layers = nn.ModuleList([GraphConvolution(hidden_size, hidden_size // 2) for hidden_size in hidden_sizes])
        
        self.first_layer_norm = nn.LayerNorm(hidden_sizes[0])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for hidden_size in hidden_sizes[1:]])
        
        self.linear = nn.Linear(hidden_sizes[-1], self.c.fan_out)
    
    def forward(self, x, adj_matrices):
        x = self.first_graph_conv_layer(x, adj_matrices[0])
        x = self.first_layer_norm(x)
        for i, (gcn, ln) in enumerate(zip(self.graph_conv_layers, self.layer_norms)):
            x = gcn(x, adj_matrices[i+1])
            x = ln(x)
            x = F.leaky_relu(x)
            x = F.dropout(x, self.c.dropout, training=self.training)
        
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


class Type3(nn.Module):
    def __init__(self, config: Type3Config, adj_list):
        super().__init__()
        self.c = config
        self.gcn = Type12(self.c.type12_config, adj_list)

    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor]):
        gcn_pred = self.gcn(x, A_s)  #! already log_softmax from model
        cls_pred = F.log_softmax(self.c.cls_logit, dim=1)

        pred = (gcn_pred) * self.c.lmbd + cls_pred * (1 - self.c.lmbd)
   
        return pred


########################################################################################################

class Type4(nn.Module):

    def __init__(self, config: Type4Config, adj_list):
        super().__init__()
        self.c = config
        self.encoder =  scGPTForAnnotation(self.c.encoder_config)
        self.gcn = Type12(self.c.type12_config, adj_list) 
      
    def forward(self, x: torch.Tensor, A_s: List[torch.Tensor], src, values, src_key_padding_mask, idx):
        encoder_preds = self.encoder(src, values, src_key_padding_mask, batch_labels=None, cls=self.c.encoder_config.CLS)["cls_output"] # not log_softmax originally
        gcn_pred = self.gcn(x, A_s)[idx]  #! already log_softmax from model
      
        pred = (gcn_pred) * self.c.lmbd + F.log_softmax(encoder_preds,dim=1) * (1 - self.c.lmbd)
        return pred




if __name__=="__main__":   

    model=torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save/dev_ms-Apr27-16-24/ms_model.pt",map_location="cpu")
    model.load_state_dict(torch.load(f"/auto/k2/aykut3/scgpt/scGPT/scgpt_gcn/save/dev_ms-Apr27-16-24/ms_model_ckpt.pt",map_location="cpu"))
    print(model)

    # Define the dimensions
    rows = 32
    cols = 700

    random_integers = torch.randint(low=1, high=5001, size=(rows, cols), dtype=torch.int64)
    src=random_integers.to("cuda")
    values= torch.rand(32,700, dtype=torch.float32).to("cuda")
    src_key_padding_mask= torch.zeros(32,700).bool().to("cuda")
    print(src_key_padding_mask)
    model.to("cuda")
    
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output= model(src,values, src_key_padding_mask)
            print(output["cell_emb"].size())
  
    