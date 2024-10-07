import torch
import ggml_mm
import torch.nn.functional as F
from torch import nn
from my_gguf import  GGUFLoader

class GGMLLinear(nn.Module):
    init = dict()
    def __init__(self,infeatures,outfeatures,weight,weight_type,device = "cuda:0",new_ggml = False):
        super(GGMLLinear, self).__init__()
        init = GGMLLinear.init
        key_v = str(infeatures)+','+str(outfeatures)+','+str(weight_type)
        if key_v not in init.keys():
            init[key_v] = ggml_mm.init(outfeatures,infeatures,1,weight_type)
            self.ggml_idx=init[key_v]
        elif new_ggml:
            self.ggml_idx=ggml_mm.init(outfeatures,infeatures,1,weight_type)
        else:
            self.ggml_idx=init[key_v]

            
        self.dev=device
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.register_buffer("weight",weight)
        self.weight_type = weight_type
   
        # self.workspace = MarlinLinear.workspace
    def forward(self,A,C=None,offset = None):
        ori_shape = A.shape
        out_shape = ori_shape[:-1] + (self.outfeatures,)
        ori_type = A.dtype
        if ori_type != torch.float:
            A = A.to(torch.float)
        if len(ori_shape)>2:
            A = A.view(-1,ori_shape[-1])
        if C is None:
            C = torch.empty((A.shape[0], self.outfeatures), dtype=torch.float, device=A.device)
        
        self.ggml_idx = ggml_mm.matmul(self.weight,A,C,self.weight_type , self.ggml_idx,self.infeatures,self.outfeatures,ori_shape[0])

        if len(ori_shape)>2:
            C = C.view(out_shape)
        if ori_type != torch.float:
            C = C.to(ori_type)

        return C



a1 = torch.randn(1,4096,dtype=torch.float)
a1_cuda = a1.cuda()
gguf_path = "/share/models/Mixtral-8x7b_q4km_new/"
gguf_loader=GGUFLoader(gguf_path)
i=0
j=0
str_name =  f"blk.{i}.ffn_up.{j}.weight"
print(gguf_loader.tensor_info[str_name])
tensor = gguf_loader.get_mmap_tensor(str_name)
tensor_fp32 = gguf_loader.load_gguf_tensor(str_name).to(torch.float)
np_dims = gguf_loader.tensor_info[str_name]['np_dims']
b1 = torch.tensor(tensor,dtype= torch.uint8).view(np_dims)

b_type = gguf_loader.tensor_info[str_name]['ggml_type']
# print(ggml_mm.matmul(b1.cuda(),a1.cuda(),n,m,8,1,1))
linear1 = GGMLLinear(4096,14336,b1.cuda(),b_type)


tensor_fp32 = tensor_fp32[:,:2048]
a1_half = torch.randn(20,2048,dtype=torch.float).cuda()
out0 = F.linear(a1_half.cpu(),tensor_fp32)
print(out0)
tmp = torch.empty(14336,2304//2,dtype = torch.uint8).cuda()
tmp.copy_(linear1.weight[:,:2304//2])
linear1.weight.data = tmp
linear1.infeatures //=2

out1 = linear1(a1_half)
print(out1)
print((out1-out0.cuda()).abs().max())

def decorate_trace_handler(rank):
    def trace_handler(prof):
        if rank in [0]:
            prof.export_chrome_trace("test"+str(rank)+".json")
    return trace_handler

prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=False,
        schedule=torch.profiler.schedule(
            wait=5,
            warmup=5,
            active=2),
        on_trace_ready=decorate_trace_handler(0)
    )
with prof:
    for i in range(0):

        linear1(a1_cuda)
        # g.replay()
        torch.cuda.synchronize()
        prof.step()