import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda:0')

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # bias & scale of cosine classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)
       

        # backbone
        self.backbone = backbone

    def cos_classifier(self, support, query):
        """
        w.shape = B, nC, d
        f.shape = B, M, d
        """
        query = F.normalize(query, p=2, dim=query.dim()-1, eps=1e-12)
        support = F.normalize(support, p=2, dim=support.dim()-1, eps=1e-12)
        all_distances1 = torch.zeros(query.size(0),support.size(0))
        all_distances1 = all_distances1.to(device)
        
        for i  in range(0,query.size(0)):
            for j in range (0,support.size(0)):
                a = query[i]  
                b = support[j]   #Take one element from query and one from support, noting that query comes before support
               
                c = a @ b.transpose(0, 1)
                d = c.max(dim=1) # dim=1 is maximized by column (it is commonly understood to maximize each row)
                a, idx1 = torch.sort(d[0], descending=True)# descending is False, ascending, and True, descending
                max6 = a[0:6].sum()
                all_distances1[i][j] = max6
       
        all_distances1 = all_distances1.unsqueeze(0)
        cls_scores = all_distances1
        cls_scores = self.scale_cls * (cls_scores + self.bias)
    
        return cls_scores

    def forward(self, supp_x, supp_y, x):
        """
        supp_x.shape = [B, nSupp, C, H, W]
        supp_y.shape = [B, nSupp]
        x.shape = [B, nQry, C, H, W]
        """
        num_classes = supp_y.max() + 1 # NOTE: assume B==1
       
        B, nSupp, C, H, W = supp_x.shape
        supp_f = supp_x.view(-1, C, H, W)
        supp_f = self.backbone.forward(supp_x.view(-1, C, H, W))
        patch_num = supp_f.size(1)
        supp_f = supp_f.view(B, nSupp, -1)
        supp_y_1hot = F.one_hot(supp_y, num_classes).transpose(1, 2) # B, nC, nSupp

        # B, nC, nSupp x B, nSupp, d = B, nC, d
        prototypes = torch.bmm(supp_y_1hot.float(), supp_f)
        prototypes = prototypes / supp_y_1hot.sum(dim=2, keepdim=True) # NOTE: may div 0 if some classes got 0 images
   
        #patch base
        prototypes = prototypes.view(5, patch_num,-1)

        feat = self.backbone.forward(x.view(-1, C, H, W))
        logits = self.cos_classifier(prototypes, feat) # B, nQry, nC

        return logits
