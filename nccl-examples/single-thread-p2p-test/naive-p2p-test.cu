#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"

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
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void printFloatList(float *buff, char* buffName, int buffLen){
    printf("WUICHAK: %s - ", buffName);
    for(int i = 0; i < buffLen; i++){
        printf("%.2f, ", buff[i]);
    }
    printf("\n");
}

int main(int argc, char* argv[])
{
  int nccl_version;
  NCCLCHECK(ncclGetVersion(&nccl_version));
  printf("NCCL VERSION - %d\n", nccl_version);
  //managing 4 devices
  int nDev = 4;
  int size = 10;
//   int devs[2] = { 0, 1};
  int devs[4] = { 0, 1, 2, 3};
  ncclComm_t comms[nDev];


  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * size * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * size * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);


  float *databuff = new float[size] ();    
  float *hostbuff = (float*)malloc( size * sizeof(float));
  for (int i = 0; i < nDev; ++i) {
    for(int idatum = 0; idatum < size; idatum++){
        databuff[idatum] = (1+idatum)*(i+1) ;
    }
    printFloatList(databuff, "databuff", size);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemcpy(sendbuff[i], databuff, size * sizeof(float), cudaMemcpyHostToDevice));
    // CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }


  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i){
    int send_peer = (i + 1) % nDev;
    int recv_peer = (i + nDev - 1) % nDev;
    // NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
    //     comms[i], s[i]));
    NCCLCHECK(ncclSend((const void*)sendbuff[i], size, ncclFloat, send_peer, comms[i], s[i]));
    NCCLCHECK(ncclRecv((void*)recvbuff[i], size, ncclFloat, recv_peer, comms[i], s[i]));
    
    printf("WUICHAK: p2p nDev-%d\n", i);
  }
  NCCLCHECK(ncclGroupEnd());


  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }


  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    printf("Device %d:\n", i);
    CUDACHECK(cudaMemcpy(hostbuff, sendbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
    printFloatList(hostbuff, "sendbuff", size);
    CUDACHECK(cudaMemcpy(hostbuff, recvbuff[i], size * sizeof(float), cudaMemcpyDeviceToHost));
    printFloatList(hostbuff, "recvbuff", size);
    
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
