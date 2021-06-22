import json
import sys
import os
import numpy as np
import pickle

index_calculate = '''
\t\t int tid = blockDim.x * blockIdx.x + threadIdx.x;
\t\t int idx = tid;

\t\t if(blockDim.x != 256) {
\t\t\t printf("not supported block dimension , it must be 256\\n");
\t\t\t return;
\t\t }
\t\t int real_tid =  threadIdx.x;
\t\t int warp_id = real_tid /32;
\t\t int warp_lane = real_tid %32;
'''
def get_mem_def(num_layer):
    shared_mem_def = '\t\t __shared__ half neuron_out[8][512];\n'
    for i in range(num_layer):
        shared_mem_def += '\t\t __shared__ half weight_%d_shared[128];\n'%(i+1)
        shared_mem_def += '\t\t __shared__ half bias_%d_shared[256];\n'%(i+1)

    shared_mem_def += '''\t\t wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag_col;
    \t\t wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    \t\t wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;\n\n'''
    shared_mem_def += '\t\t //define or set constant WMMA_M = 32, WMMA_N = 8, WMMA_K = 16 to use this mma_sync variant\n'
    return shared_mem_def
def get_mem_load(num_layer, inputs, means_in, scales_in):
    mem_load_str = '''\t\t weight_1_shared[real_tid] = 0.0;
    \t\t weight_2_shared[real_tid] = 0.0;
    \t\t weight_3_shared[real_tid] = 0.0;
    \t\t // 8 is the number of warps, simply reset buffer mem for all warps
    \t\t for (int i = 0; i<8 ; i++){
    \t\t      neuron_out[i][real_tid] = 0.0;
    \t\t      neuron_out[i][real_tid+256] = 0.0;
    \t\t    }
    \t\t __syncthreads();\n'''

    mem_load_str += '\t\t// if condition to avoid illegal memaccess.\n'
    mem_load_str += '\t\t // can further reduce number of mem_load by checking this condition separately for each layer e.g. in case layer 1 only needs to load 16 or 32 weights.\n'
    mem_load_str += '\t\t  if (real_tid < %d)\n'%(len(inputs)*8)
    mem_load_str += '\t\t\t weight_1_shared[real_tid] = weight_1_half_d[real_tid];\n'
    mem_load_str += '\t\t if (real_tid <64){\n'
    for i in range(1,num_layer):
        mem_load_str += '\t\t\t weight_%d_shared[real_tid] = weight_%d_half_d[real_tid];\n'%(i+1,i+1)
    mem_load_str += '\t\t }\n'
    for i in range(num_layer):
        mem_load_str += '\t\t bias_%d_shared[real_tid] = bias_%d_half_d[warp_id];\n'%(i+1, i+1)
    input_scaling = False
    if (len(means_in) == len(inputs)):
        input_scaling = True
    if (not input_scaling and len(means_in)!= 0):
        print (" len mismatch between scaling parameter %d and inputs %d "%(len(means_in), len(inputs)))
    for i in range(len(inputs)):
        if( input_scaling):
            mem_load_str += '\t\t neuron_out[warp_id][warp_lane+ %d*32] = ( %s - %E )* %E ;\n'%(i, inputs[i], means_in[i], scales_in[i])
        else:
            mem_load_str += '\t\t neuron_out[warp_id][warp_lane+ %d*32] = %s;\n'%(i, inputs[i])
    mem_load_str += '\t\t __syncthreads();\n'
    return mem_load_str

def get_core_mlp(num_layer):
    core_mlp_str = ''
    for i in range(num_layer):
        core_mlp_str+= '\t\t wmma::load_matrix_sync(a_frag_col, (const __half*)neuron_out[warp_id], 32);\n'
        core_mlp_str+= '\t\t wmma::load_matrix_sync(b_frag, (const __half*)&weight_%d_shared, 8);\n'%(i+1)
        core_mlp_str+= '\t\t wmma::load_matrix_sync(c_frag, (const half*)&bias_%d_shared, 32, wmma::mem_col_major);\n'%(i+1)
        core_mlp_str+= '\t\t wmma::mma_sync(c_frag, a_frag_col, b_frag, c_frag);\n'
        if (i != num_layer-1): # dont do RELU on last layer
            core_mlp_str+= '\t\t for (int i = 0; i< c_frag.num_elements; i ++)\n'
            core_mlp_str+= '\t\t\t c_frag.x[i] = relu(c_frag.x[i]);\n'
        else:
            core_mlp_str+= '\t\t //performance trick: if the last layer only has 2-3 outputs, it maybe faster to compute dot product explicitly (e.g. 2 outputs cost 16 multiplying-accumulate ops) without using mma_sync\n'
        core_mlp_str+= '\t\t wmma::store_matrix_sync((half*)neuron_out[warp_id], c_frag, 32,wmma::mem_col_major);\n'
        core_mlp_str+= '\t\t // do not need __syncthreads() here because all instructions above are sync. End of layer %d\n'%(i+1)
    return core_mlp_str
def get_output_save(outputs, scales_out):
    output_scaling = False
    output_save_str = ''
    if (len(scales_out) == len(outputs)):
        output_scaling = True
    if (not output_scaling and len(scales_out)!= 0):
        print (" len mismatch between scaling parameter %d and outputs %d "%(len(scales_out), len(outputs)))
    for i in range(len(outputs)):
        if (output_scaling):
            output_save_str+='\t\t %s = __half2float(neuron_out[warp_id][warp_lane+%d*32])*%E;\n'%(outputs[i],i,scales_out[i])
        else:
            output_save_str+='\t\t %s = __half2float(neuron_out[warp_id][warp_lane+%d*32]);\n'%(outputs[i],i)

    output_save_str+= '\t\t// end of Tensor approx region\n'
    return output_save_str
def get_define_str(num_layer,weights,bias):
    res_str = (
        "//begin tensorox_define \n"
        "#include <mma.h>\n"
        "#include <cuda_fp16.h>\n"
        "using namespace nvcuda;\n"
        "const int WMMA_M = 32;\n"
        "const int WMMA_N = 8;\n"
        "const int WMMA_K = 16;\n"
        "__device__ __inline__ half relu( half x){\n"
        "\t return (x>__float2half_rn(0.0))? x:__float2half_rn(0.0);\n"
        "}\n"
        )
    #first layer need to consider low dimension inputs to save space
    res_str += '__constant__ half weight_1_half_d[%d];\n'%(len(weights[0]))
    res_str += '__constant__ half bias_1_half_d[8];\n'
    for i in range (1,num_layer):
        res_str += '__constant__ half weight_%d_half_d[64];\n'%(i+1)
        res_str += '__constant__ half bias_%d_half_d[8];\n'%(i+1)

    res_str += 'float weight_1[%d]= { %s };\n'%(len(weights[0]), ','.join([str(a) for a in weights[0]] ))
    res_str += 'float bias_1[8]= { %s };\n'%(','.join([str(a) for a in bias[0]]))
    for i in range (1,num_layer):
        res_str += 'float weight_%d[%d]= { %s };\n'%(i+1,len(weights[i]), ','.join([str(a) for a in weights[i]]))
        res_str += 'float bias_%d[8]= { %s };\n'%(i+1,','.join([str(a) for a in bias[i]]))

    res_str += 'half weight_1_half[%d], bias_1_half[8]'%(len(weights[0]))
    for i in range (1,num_layer):
        res_str += ',weight_%d_half[64], bias_%d_half[8]'%(i+1,i+1)
        if (i == num_layer-1):
            res_str+=';\n'
    res_str += ("void prepare_half_prec_weights(){\n"
                "\t for (int i =0 ; i< 8 ; i ++){\n"
                )
    for i in range(num_layer):
            	res_str += '\t\t bias_%d_half[i] = __float2half_rn(bias_%d[i]);\n'%(i+1,i+1)

    res_str += '\t }\n'

    res_str += 	'\t for (int i =0 ; i<%d; i ++)\n'%(len(weights[0]))
    res_str += 	'\t\t weight_1_half[i] = __float2half_rn(weight_1[i]);\n'
    for i in range (1,num_layer):
        if (i!= num_layer-1):
            res_str += '\t for (int i = 0 ; i<64; i++)\n'
            res_str += '\t\t weight_%d_half[i] = __float2half_rn(weight_%d[i]);\n'%(i+1,i+1)
        else:
            #special treatment for last layer
            col_idx = len(weights[i])/8

            res_str += '\t for (int i =0; i <8; i ++){\n'
            res_str += '\t\t for (int j =0 ; j <%d ; j ++)\n'%(col_idx)
            res_str += '\t\t\t weight_%d_half[i*8+j] = __float2half_rn(weight_%d[i*%d + j]);\n'%(i+1,i+1,col_idx)
            res_str += '\t\t for (int j = %d; j <8 ;j++)\n'%(col_idx)
            res_str += '\t\t\t weight_%d_half[i*8+j] = __float2half_rn(0.0);\n'%(i+1)
            res_str += '\t}\n'
    res_str += '}\n'
    res_str += '//end of tensorox_define, call prepare_half_prec_weights in the main function to initialize these arrays, then use cudaMemcpyToSymbol to copy the weight_i_half to weight_i_half_d \n'
    return res_str

def rewrite(config):
    data = {}
    try:
        f = open(config,'r')
        data = json.load(f)
    except:
        print ('cannot read configuration file. please refer to the sample_config.json')
        exit(0)
    print ('Config: ', data)

    weights = pickle.load( open( data["weight_file"], "rb" ))
    bias = pickle.load( open( data["bias_file"], "rb" ))
    for i in range(len(weights)):
        weights[i] = weights[i].flatten()
        weights[i][np.abs(weights[i])<1e-10] = 0.0



    source_file = open(data['source_file'], 'r')
    source_lines = source_file.readlines()

    line_num = 0
    res_lines = []

    for i in range(len(source_lines)):
        if 'tensorox_input' not in source_lines[i]:
            res_lines.append(source_lines[i])#res_file.write(source_lines[i])
        else:
            line_num = i
            break
    print ('inputs desc detected : ', source_lines[line_num])
    inputs = source_lines[line_num].replace('\n','').split(':')[1:]
    print (inputs)
    for i in range(line_num, len(source_lines)):
        if 'tensorox_output' in source_lines[i]:
            line_num = i
            break

    print ('outputs desc detected : ', source_lines[line_num])
    outputs = source_lines[line_num].replace('\n','').split(':')[1:]
    print (outputs)

    means_in = data['means_in']
    scales_in = data['scales_in']
    scales_out = data['scales_out']

    print (len(weights[0]))
    print (bias[0])
    num_layer = len(weights)

    res_lines.append(index_calculate)
    res_lines.append(get_mem_def(num_layer))
    res_lines.append(get_mem_load(num_layer, inputs, means_in, scales_in))
    res_lines.append(get_core_mlp(num_layer))


    res_lines.append(get_output_save(outputs, scales_out))


    #write the rest of the source file
    for i in range(line_num+1, len(source_lines)):
        res_lines.append(source_lines[i])

    #second pass to write constant weights declaration and conversion
    define_line_num = 0
    for i in range (len(res_lines)):
        if ("tensorox_define" in res_lines[i]):
            define_line_num = i
            break
    if (define_line_num == len(res_lines) -1):
        print ('cant find location of //tensorox_define in the code, writing define to the beginning of the file')
        define_line_num = 0

    res_lines.insert(define_line_num, get_define_str(num_layer, weights, bias))

    res_file = open(data["destination_file"], "w")
    for line in res_lines:
        res_file.write(line)
    res_file.close()
if __name__ == '__main__':
    arguments = sys.argv[1:]
    if len(arguments)!=1:
        print("Usage: python rewrite.py config.json")
        exit(0)
    rewrite(arguments[0])
