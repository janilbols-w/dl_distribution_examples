/****************************************************************************
 * Author: wanghuize@virtaitech.com
 * Create Date: 16 Nov, 2020
 * Update Date: 10 Dec, 2020
 * Purpose: Tensorflow distribute method is using multi-threads nccl,
 *          which will cause Orion Server Hanging.
 *          This toy example is used to reproduce the issue, at a minimal 
 *          size.
****************************************************************************/
#include <nccl.h>
#include <thread>
#include "cuda_runtime.h"
#include "cuda.h"
#include <unistd.h>


#define FLAG_PARALLEL_INIT
#define FLAG_COMM_GROUP
#define FLAG_REDUCE_GROUP

#define DEFAULT_NCCL_OP_TYPE_ENUM 0
enum ncclOpType{
    op_ncclAllReduce = 0,
    op_ncclBroadcast = 1,
    op_ncclBcast = 2,
    op_ncclReduce = 3,
    op_ncclAllGather = 4,
    op_ncclReduceScatter = 5,
    op_ncclSendAndRecv = 6,
};

#define FLAG_DEBUG_PRINT

struct globalInfo{
    ncclUniqueId* ncclId;
    int nDev;
    int rank;
    int size = 12; //32*1024*1024;
    ncclComm_t* comms;
    cudaStream_t * streams;
    int func_type = DEFAULT_NCCL_OP_TYPE_ENUM;
};

#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",             \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)
  
  
#define NCCLCHECK(cmd) do {                         \
    ncclResult_t r = cmd;                             \
    if (r!= ncclSuccess) {                            \
        printf("Failed, NCCL error %s:%d '%s'\n",       \
            __FILE__,__LINE__,ncclGetErrorString(r));   \
        exit(EXIT_FAILURE);                             \
    }                                                 \
} while(0)

void printFloatList(int rank, float *buff, char const * buffName, int buffLen){
    printf("rank(%d): %s - ", rank, buffName);
    for(int i = 0; i < buffLen; i++){
        printf("%.2f, ", buff[i]);
    }
    printf("\n");
}


void run(globalInfo gInfo){   
    printf("WUICHAK: Starting thread %d of %d\n", gInfo.rank, gInfo.nDev);
    printf("WUICHAK(thread %d of %d): cudaSetDevice\n", gInfo.rank, gInfo.nDev);
    CUDACHECK(cudaSetDevice(gInfo.rank));
#ifdef FLAG_PARALLEL_INIT
    ncclComm_t comm;
    // for(int i =10; i>0; i--){
    //     // usleep(gInfo.rank * 10000);
    //     printf("WUICHAK(thread %d of %d): waiting count down %d\n", gInfo.rank, gInfo.nDev, i);
    // }
#ifdef FLAG_COMM_GROUP
    printf("WUICHAK(thread %d of %d): PARALLEL ncclGroupStart\n", gInfo.rank, gInfo.nDev);
    NCCLCHECK(ncclGroupStart());
#endif
    printf("WUICHAK(thread %d of %d): PARALLEL ncclCommInitRank\n", gInfo.rank, gInfo.nDev);
    NCCLCHECK(ncclCommInitRank(&comm, gInfo.nDev, *(gInfo.ncclId), gInfo.rank));
    printf("WUICHAK(thread %d of %d): Done PARALLEL ncclCommInitRank\n", gInfo.rank, gInfo.nDev);
#ifdef FLAG_COMM_GROUP
    printf("WUICHAK(thread %d of %d): PARALLEL ncclGroupEnd\n", gInfo.rank, gInfo.nDev);
    NCCLCHECK(ncclGroupEnd());
    printf("WUICHAK(thread %d of %d): PARALLEL done ncclGroupEnd\n", gInfo.rank, gInfo.nDev);
#endif
    gInfo.comms[gInfo.rank] = comm;
    // create stream, allocat mem
    cudaStream_t s;
    printf("WUICHAK(thread %d of %d): PARALLEL cudaStreamCreate\n", gInfo.rank, gInfo.nDev);
    CUDACHECK(cudaStreamCreate(&s));
    gInfo.streams[gInfo.rank] = s;
#endif

    float *sendbuff, *recvbuff, *hostbuff;
    float *databuff = new float[gInfo.size] ();
    hostbuff = (float*)malloc( gInfo.size * sizeof(float));
    for(int i = 0; i < gInfo.size; i++){
        databuff[i] = (1+i)*(gInfo.rank+1) ;
    }
#ifdef FLAG_DEBUG_PRINT
    printFloatList(gInfo.rank, databuff, "databuff", gInfo.size);
#endif
    printf("WUICHAK(thread %d of %d): cudaMalloc\n", gInfo.rank, gInfo.nDev);
    CUDACHECK(cudaMalloc(&sendbuff, gInfo.size * sizeof(float)));
    CUDACHECK(cudaMalloc(&recvbuff, gInfo.size * sizeof(float)));
    
    printf("WUICHAK(thread %d of %d): cudaMemset\n", gInfo.rank, gInfo.nDev);
    CUDACHECK(cudaMemcpy(sendbuff, databuff, gInfo.size * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recvbuff, 0 , gInfo.size * sizeof(float)));
    // sleep(gInfo.rank*1);
#ifdef FLAG_REDUCE_GROUP
    printf("WUICHAK(thread %d of %d): REDUCE_GROUP ncclGroupStart\n", gInfo.rank, gInfo.nDev);
    NCCLCHECK(ncclGroupStart());
#endif
    printf("WUICHAK(thread %d of %d): start op\n", gInfo.rank, gInfo.nDev);
	printf("sendbuf:%p, recvbuff:%p, size:%d, datatype:%d, op:%d, comm:%p, stream:%p\n",
            (const void*)sendbuff, (void*)recvbuff, gInfo.size, ncclFloat, ncclSum, 
            gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]);

    char const *op_name = "nccl_unknown";
    switch(gInfo.func_type){
        case op_ncclAllReduce:
            op_name = "op_ncclAllReduce";
            NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, gInfo.size, ncclFloat, ncclSum, 
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            break;
        case op_ncclBroadcast:
            op_name = "op_ncclBroadcast";
            NCCLCHECK(ncclBroadcast((const void*)sendbuff, (void*)recvbuff, gInfo.size, ncclFloat, 0, // broadcast value from rank_0
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            break;
        case op_ncclBcast:
            op_name = "op_ncclBcast";
            NCCLCHECK(ncclBcast((void*)sendbuff, gInfo.size, ncclFloat, 0, // broadcast value from rank_0, via in place mode
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            break;
        case op_ncclReduce:
            op_name = "op_ncclReduce";
            // NCCLCHECK(ncclReduce((const void*)sendbuff, (void*)recvbuff, gInfo.size, ncclFloat, ncclSum, 0, // reduce to rank_0
            //                         gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            NCCLCHECK(ncclReduce((const void*)sendbuff, 
                                    gInfo.rank == 0 ? (void*)recvbuff : nullptr, 
                                    gInfo.size, ncclFloat, ncclSum, 0, // reduce to rank_0
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));                
            break;
        case op_ncclAllGather:
            op_name = "op_ncclAllGather";
            NCCLCHECK(ncclAllGather((const void*)sendbuff, (void*)recvbuff, gInfo.size, ncclFloat,
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            break;
        case op_ncclReduceScatter:
	    op_name = "op_ncclReduceScatter";
            NCCLCHECK(ncclReduceScatter((const void*)sendbuff, (void*)recvbuff, gInfo.size/gInfo.nDev, ncclFloat, ncclSum, 
                                    gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            break;
#ifdef __NCCL_2_7_H
        case op_ncclSendAndRecv:
            op_name = "op_ncclSendAndRecv";
            NCCLCHECK(ncclGroupStart());
            int send_peer = (gInfo.rank + 1) % gInfo.nDev;
            int recv_peer = (gInfo.rank - 1 + gInfo.nDev) % gInfo.nDev;
            NCCLCHECK(ncclSend((const void*)sendbuff, gInfo.size, ncclFloat, send_peer, 
                                gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            NCCLCHECK(ncclRecv((void*)recvbuff, gInfo.size, ncclFloat, recv_peer, 
                                gInfo.comms[gInfo.rank], gInfo.streams[gInfo.rank]));
            NCCLCHECK(ncclGroupEnd());
            break;
#endif
    }        
    printf("WUICHAK(thread %d of %d): end %s\n", gInfo.rank, gInfo.nDev, op_name);

#ifdef FLAG_REDUCE_GROUP
    printf("WUICHAK(thread %d of %d): REDUCE_GROUP ncclGroupEnd\n", gInfo.rank, gInfo.nDev);
    NCCLCHECK(ncclGroupEnd());
    printf("WUICHAK(thread %d of %d): REDUCE_GROUP ncclGroupEnd done\n", gInfo.rank, gInfo.nDev);
#endif
    printf("WUICHAK(thread %d of %d): start cudaStreamSynchronize\n", gInfo.rank, gInfo.nDev);
    CUDACHECK(cudaStreamSynchronize(gInfo.streams[gInfo.rank]));
    printf("WUICHAK(thread %d of %d): end cudaStreamSynchronize\n", gInfo.rank, gInfo.nDev);
#ifdef FLAG_DEBUG_PRINT
    CUDACHECK(cudaMemcpy(hostbuff, sendbuff, gInfo.size * sizeof(float), cudaMemcpyDeviceToHost));
    printFloatList(gInfo.rank, hostbuff, "sendbuff", gInfo.size);
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff, gInfo.size * sizeof(float), cudaMemcpyDeviceToHost));
    printFloatList(gInfo.rank, hostbuff, "recvbuff", gInfo.size);
#endif
    printf("WUICHAK(thread %d of %d): destroy and free mem\n", gInfo.rank, gInfo.nDev);
    //free device buffers
    CUDACHECK(cudaFree(sendbuff));
    CUDACHECK(cudaFree(recvbuff));

#ifdef FLAG_PARALLEL_INIT
    NCCLCHECK(ncclCommDestroy(gInfo.comms[gInfo.rank]));
#endif
};

int main(int argc, char* argv[]) {
    
#ifdef FLAG_PARALLEL_INIT
    printf("WUICHAK: FLAG_PARALLEL_INIT - TRUE\n");
#else
    printf("WUICHAK: FLAG_PARALLEL_INIT - FALSE\n");
#endif

#ifdef FLAG_COMM_GROUP
    printf("WUICHAK: FLAG_COMM_GROUP - TRUE\n");
#else
    printf("WUICHAK: FLAG_COMM_GROUP - FALSE\n");
#endif

#ifdef FLAG_REDUCE_GROUP
    printf("WUICHAK: FLAG_REDUCE_GROUP - TRUE\n");
#else
    printf("WUICHAK: FLAG_REDUCE_GROUP - FALSE\n");
#endif

    int nThreads = 2;
    int nccl_func_type = DEFAULT_NCCL_OP_TYPE_ENUM;
    if (argc > 1){
        printf("%s", argv[1]);
        nThreads = (int)strtol(argv[1], NULL, 10);
        if (argc > 2){
            nccl_func_type = (int)strtol(argv[2], NULL, 10);
        }
    }else{
        printf("INFO: using default value, nThread=2, you could change the nThread value\n");
        printf("      as an args for the program\n");
        printf("         e.g.  nccl_thread_test 4\n");
        printf("         to run 4 threads with this test.\n");
    }
    printf("WUICHAK: running with %d threads\n", nThreads);

    ncclUniqueId ncclId;
    ncclComm_t comms[nThreads];
    cudaStream_t streams[nThreads];

#ifndef FLAG_PARALLEL_INIT
    ncclGetUniqueId(&ncclId);
    printf("WUICHAK: ncclCommInit/cudaStreamCreate in Main Thread\n");
#ifdef FLAG_COMM_GROUP
    printf("WUICHAK: MainThread ncclGroupStart\n"); 
    NCCLCHECK(ncclGroupStart());
#endif
    for (int i=0; i<nThreads; i++) {
        printf("WUICHAK: MainThread ncclCommInitRank %d\n", i);
        CUDACHECK(cudaSetDevice(i));
        NCCLCHECK(ncclCommInitRank(comms+i, nThreads, ncclId, i));
    }
#ifdef FLAG_COMM_GROUP
    printf("WUICHAK: MainThread ncclGroupEnd\n"); 
    NCCLCHECK(ncclGroupEnd());
#endif
    for (int i=0; i<nThreads; i++) {
        printf("WUICHAK: MainThread cudaSetDevice %d\n", i);
        CUDACHECK(cudaSetDevice(i));
        printf("WUICHAK: MainThread cudaStreamCreate %d\n", i);
        CUDACHECK(cudaStreamCreate(streams+i));
    }
#endif
    int repeatTimes = 100;
    std::thread ths[nThreads];
    for (int rpt = 0; rpt < repeatTimes; rpt++){
	printf("========== repeat %d ==========\n", rpt);
#ifdef FLAG_PARALLEL_INIT
        ncclGetUniqueId(&ncclId);
#endif
        for (int t=nThreads-1; t>=0; t--) {
            globalInfo gInfo;
            gInfo.nDev = nThreads;
            gInfo.ncclId = &ncclId;
            gInfo.rank = t;
            gInfo.comms = comms;
            gInfo.streams = streams;
            gInfo.func_type = nccl_func_type;
            ths[t] = std::thread(run, gInfo);
        }
        for (int t=nThreads-1; t>=0; t--) {
            ths[t].join();
            // ths[t].detach();
        }
        // usleep(100000);
        printf("WUICHAK: DONE repeat%d\n", rpt);
    }
    
    //finalizing NCCL
#ifndef FLAG_PARALLEL_INIT
    for (int i=0; i<nThreads; i++) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }
#endif
    return 0;
}

