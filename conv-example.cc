#include "mkldnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CODE_MESSAGE(ret) \
    do { \
        fprintf(stdout, "%s:error at: %d,errcode:%d\n", __FILE__, __LINE__,ret); \
    }while(false)

#define MKLDNN_CHECK(flag) do { \
    mkldnn_status_t status = flag; \
    if (status != mkldnn_success) { \
        CODE_MESSAGE(status); \
        exit(1); \
    } \
} while(false)


int main(int argc, char **argv)
{
    const int c = 1;
    const int h = 8;
    const int w = 8;
    const int inChannel = 1;
    const int outChannel = 1;

    size_t size = h*w*sizeof(float);
    float* pSrc = (float*)malloc(size*c);
    float* pDst = (float*)malloc(size*outChannel);
    memset(pDst,0,size*outChannel);

    float* pWeight = (float*)malloc(outChannel*inChannel*3*3*sizeof(float));
    float* pBias = (float*)malloc(outChannel*sizeof(float));

    for(int i = 0 ;i<h*w*c;i++){
        pSrc[i] = i;
    }

    for(int i=0;i<outChannel;i++){
        for(int j=0;j<inChannel;j++){
            for(int ki =0;ki<3;ki++){
                for(int kj =0;kj<3;kj++){
                    int index = i*inChannel*3*3 + j*3*3+ ki*3+kj;
                    pWeight[index] = ki;
                }
            }
        }
    }


    mkldnn_engine_t engine;
    auto ret = mkldnn_engine_create(&engine, mkldnn_cpu, 0);
    MKLDNN_CHECK(ret);

    mkldnn_stream_t stream;
    ret = mkldnn_stream_create(&stream, engine, mkldnn_stream_default_flags);
    MKLDNN_CHECK(ret);

    mkldnn_dim_t srcDims[4] = {1,inChannel,h,w};
    mkldnn_dim_t weightDims[4] = {outChannel,inChannel,3,3};
    mkldnn_dim_t biasDims[1] = {outChannel};
    mkldnn_dim_t dstDims[4] = {1,outChannel,h,w};

    mkldnn_memory_desc_t srcDesc,weightDesc,biasDesc,dstDesc;
    ret = mkldnn_memory_desc_init_by_tag(&srcDesc,4,srcDims,mkldnn_f32,mkldnn_nchw);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_desc_init_by_tag(&weightDesc,4,weightDims,mkldnn_f32,mkldnn_oihw);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_desc_init_by_tag(&biasDesc,1,biasDims,mkldnn_f32,mkldnn_x);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_desc_init_by_tag(&dstDesc,4,dstDims,mkldnn_f32,mkldnn_nchw);
    MKLDNN_CHECK(ret);

    mkldnn_memory_t srcMemory,weightMemroy,biasMemory,dstMemory;
    ret = mkldnn_memory_create(&srcMemory,&srcDesc,engine,pSrc);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_create(&weightMemroy,&weightDesc,engine,pWeight);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_create(&biasMemory,&biasDesc,engine,pBias);
    MKLDNN_CHECK(ret);
    ret = mkldnn_memory_create(&dstMemory,&dstDesc,engine,pDst);
    MKLDNN_CHECK(ret);

    /*conv*/
    mkldnn_dim_t strides[2] = {1,1};
    mkldnn_dim_t padding[2] = {1,1};
    mkldnn_convolution_desc_t convDesc;
    ret = mkldnn_convolution_forward_desc_init(&convDesc, mkldnn_forward,
            mkldnn_convolution_direct, &srcDesc, &weightDesc,
            &biasDesc, &dstDesc, strides, padding,
            padding, mkldnn_padding_zero);
    MKLDNN_CHECK(ret);

    mkldnn_primitive_desc_t opDesc;
    ret = mkldnn_primitive_desc_create(&opDesc,&convDesc,NULL,engine,NULL);
    MKLDNN_CHECK(ret);

    mkldnn_primitive_t convOp;
    ret = mkldnn_primitive_create(&convOp,opDesc);
    MKLDNN_CHECK(ret);

    
    mkldnn_exec_arg_t convArgs[4] = {
        {MKLDNN_ARG_SRC, srcMemory},
        {MKLDNN_ARG_WEIGHTS, weightMemroy},
        {MKLDNN_ARG_BIAS, biasMemory},
        {MKLDNN_ARG_DST, dstMemory},
    };

    ret = mkldnn_primitive_execute(convOp,stream,4,convArgs);
    MKLDNN_CHECK(ret);

    for(int i=0;i<h*w*outChannel;i++){
        printf("%d:%f\n",i,pDst[i]);
    }


    mkldnn_memory_destroy(srcMemory);
    mkldnn_memory_destroy(weightMemroy);
    mkldnn_memory_destroy(biasMemory);
    mkldnn_memory_destroy(dstMemory);
    mkldnn_stream_destroy(stream);
    mkldnn_engine_destroy(engine);
    mkldnn_primitive_desc_destroy(opDesc);
    mkldnn_primitive_destroy(convOp);

    free(pSrc);
    free(pWeight);
    free(pBias);
    free(pDst);

    printf("over!~\n");

    return 1;

}