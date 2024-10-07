#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include "ggml-cuda.h"
// #ifdef GGML_USE_CUDA
// #include "ggml-cuda.h"
// #endif
#include <iostream>
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <torch/extension.h>
#include <torch/torch.h>
#include <torch/all.h>
#include <torch/python.h>
static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;
    struct ggml_tensor * c;
    int n;
    int k;
    int m;
    // the backend to perform the computation (CPU, CUDA, METAL)
    ggml_backend_t backend = NULL;

    // the backend buffer to storage the tensors data of a and b
    ggml_backend_buffer_t buffer;

    // the context to define the tensor information (dimensions, size, memory address)
    struct ggml_context * ctx;
};
std::vector<simple_model> models;
std::vector<ggml_gallocr_t> allocrs;
std::vector<struct ggml_cgraph*> gfs;
std::vector<struct ggml_tensor*> results;

enum ggml_type conver_int2ggmltype(int a_type)
{
    if (a_type == 12){
        return GGML_TYPE_Q4_K;
    }
    if (a_type == 2){
        return GGML_TYPE_Q4_0;
    }  
    if (a_type == 3){
        return GGML_TYPE_Q4_1;
    }
    if (a_type == 8){
        return GGML_TYPE_Q8_0;
    }
    if (a_type == 13)
    {
        return GGML_TYPE_Q5_K;
    }
    if (a_type == 14)
    {
        return GGML_TYPE_Q6_K;
    }
    if (a_type == 30){
        return GGML_TYPE_BF16;
    }
    if (a_type == 1){
        return GGML_TYPE_F16;
    }
    return GGML_TYPE_F32;
}
// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, int rows_A, int cols_A, int rows_B, int cols_B,int a_type,int b_type) {
    // initialize the backend
// #ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
// #endif

// #ifdef GGML_USE_METAL
//     fprintf(stderr, "%s: using Metal backend\n", __func__);
//     ggml_backend_metal_log_set_callback(ggml_log_callback_default, nullptr);
//     model.backend = ggml_backend_metal_init();
//     if (!model.backend) {
//         fprintf(stderr, "%s: ggml_backend_metal_init() failed\n", __func__);
//     }
// #endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 3;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    model.a = ggml_new_tensor_2d(model.ctx, conver_int2ggmltype(a_type), cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, conver_int2ggmltype(b_type), cols_B, rows_B);
    model.c = ggml_new_tensor_2d(model.ctx, conver_int2ggmltype(b_type), rows_A, rows_B);
    // create tensors
   

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    // ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
    // ggml_backend_tensor_set(model.b, b, 0, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result = a*b^T
    struct ggml_tensor * result = ggml_mul_mat(ctx0, model.a, model.b);
    
    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);



    ggml_backend_graph_compute(model.backend, gf);
    
    // in this case, the output tensor is the last one in the graph
    return gf->nodes[gf->n_nodes - 1];
}


int init(int out_feature, int in_feature, int seq_len,int a_type){
    
    int cur_size = models.size() ;
    if (cur_size == 0) ggml_time_init();
    simple_model model;
    // ggml_gallocr_t allocr;
    // struct ggml_cgraph* gf;

    model.n = seq_len;
    model.m = out_feature;
    model.k = in_feature;

    load_model(model, model.m, model.k,  model.n,model.k,a_type,0);
    // allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    // gf = build_graph(model);
    // ggml_gallocr_reserve(allocr, gf);
   
    models.push_back(model);
    // allocrs.push_back(allocr);

    // struct ggml_cgraph * gf1 = build_graph(model);
    // ggml_gallocr_alloc_graph(allocr, gf);
    // struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    // fprintf(stderr, "%s", result);
    // results.push_back(result);
    // allocate tensors
    
    // gfs.push_back(gf);
    return cur_size;
    // return  torch::from_blob(model.a->data, {m, k},torch::kCUDA);
}

int tmp_int = 0;
bool flag_tmp = false;
int matmul(torch::Tensor& a,torch::Tensor& b,torch::Tensor& c,int a_type,int idx,int in_feature,int out_feature,int seq_len){
    simple_model model = models[idx];
    int out_idx = idx;
    // auto a_sizes = a.sizes();
    // auto b_sizes = b.sizes();
    // auto c_sizes = c.sizes();
    // model.a->ne[0] = in_feature;
    // model.a->ne[1] = out_feature;
    // model.b->ne[0] = in_feature;
    // model.c->ne[0] = out_feature;
    model.b->ne[1] = seq_len;
    model.c->ne[1] = seq_len;
    if ( (model.a->ne[0] != in_feature) || (model.a->ne[1] != out_feature) )
    {   
        tmp_int = 0;
        flag_tmp = false;
        for (simple_model item : models)
        {
            if ( (item.a->ne[0] == in_feature) && (item.a->ne[1] == out_feature) )
            {
                model = item;
                out_idx = tmp_int;
                flag_tmp = true;
                break;
            }
            tmp_int++;
        }
        if (!flag_tmp){
            out_idx = init(out_feature,in_feature,seq_len,a_type);
            model = models[out_idx];
        }
        idx = out_idx;
    }
    // std::cout<<model.b->ne[1]<<std::endl;
    if (a_type==30)
    {
        model.a->data = a.data_ptr<at::BFloat16>();
    }
    else if (a_type==1)
    {
        model.a->data = a.data_ptr<at::Half>();
    }
    else if ((a_type==2)||(a_type == 12)||(a_type == 3)||(a_type == 8)||(a_type == 13)||(a_type == 14))
    {
        model.a->data = a.data_ptr<uint8_t>();
    }
    else{
        model.a->data = a.data_ptr<float>();
    }
    model.b->data = b.data_ptr<float>();
    model.c->data = c.data_ptr<float>();
    // ggml_gallocr_t allocr = allocrs[idx];
    // struct ggml_cgraph* gf = gfs[idx];
    // struct ggml_tensor* result = model.c;
    
    // struct ggml_cgraph * gf1 = build_graph(model);
    // ggml_gallocr_alloc_graph(allocr, gf1);
    // struct ggml_tensor * result = gf->nodes[gf->n_nodes - 1];
    // result->data = c.data_ptr<float>();
    // ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)model.backend->context;
    // result = compute(model, allocr);
    
    ggml_cuda_mul_mat_1(model.backend,model.a,model.b,model.c);
    // 
    // return torch::from_blob(model.c->data, {model.n, model.m},torch::kCUDA);
    return out_idx;
}
void free_ggml(){
    for (auto allocr : allocrs)
    {
        ggml_gallocr_free(allocr);
    }
    for (auto model : models)
    {
        // free memory
        ggml_free(model.ctx);

        // release backend memory and free backend
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init", &init, "init");
    m.def("matmul", &matmul, "mm");
    m.def("free_ggml", &free_ggml, "free_ggml");
    
}
