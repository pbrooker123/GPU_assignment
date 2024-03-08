
//# Starting point was new cudaRuntime project in VS which was an array addition project
//# Intelisense is up and running in win10 on my alienware minitower
//# Graphics card is NVIDIA GeForce GTX 970: 4GB, 1664 cores

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib> // For rand() and srand()...from chatGPT
#include <ctime>   // For time()....from chatGPT

using namespace std;


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// same kernel for whether it is on GPU#1 or GPU#2
// the kernel call will pass in different array pointers
// also, the cudaMemcpy() will use different source arrays
__global__ void SVKernel(int* d_forDotProduct,  int* d_SV01, int* d_SV02, int* d_SV03, int* d_SV04, int* d_SV05,
                                                int* d_SV06, int* d_SV07, int* d_SV08, int* d_SV09, int* d_SV10, 
                                                int* d_dot01, int* d_dot02, int* d_dot03, int* d_dot04, int* d_dot05, 
                                                int* d_dot06, int* d_dot07, int* d_dot08, int* d_dot09, int* d_dot10 )
{
    int i = threadIdx.x;
    d_dot01[i] = d_forDotProduct[i] * d_SV01[i];
    d_dot02[i] = d_forDotProduct[i] * d_SV02[i];
    d_dot03[i] = d_forDotProduct[i] * d_SV03[i];
    d_dot04[i] = d_forDotProduct[i] * d_SV04[i];
    d_dot05[i] = d_forDotProduct[i] * d_SV05[i];
    d_dot06[i] = d_forDotProduct[i] * d_SV06[i];
    d_dot07[i] = d_forDotProduct[i] * d_SV07[i];
    d_dot08[i] = d_forDotProduct[i] * d_SV08[i];
    d_dot09[i] = d_forDotProduct[i] * d_SV09[i];
    d_dot10[i] = d_forDotProduct[i] * d_SV10[i];
}

void DrawBoard(string boardStr[], string label)
{
    cout << endl;
    cout << " " << boardStr[0] << " | " << boardStr[1] << " | " << boardStr[2] << " | " << boardStr[3] << " " << endl;
    cout << "---|---|---|---\n";
    cout << " " << boardStr[4] << " | " << boardStr[5] << " | " << boardStr[6] << " | " << boardStr[7] << endl;
    cout << "---|---|---|---  " << label << endl;
    cout << " " << boardStr[8] << " | " << boardStr[9] << " | " << boardStr[10] << " | " << boardStr[11] << " " << endl;
    cout << "---|---|---|---\n";
    cout << " " << boardStr[12] << " | " << boardStr[13] << " | " << boardStr[14] << " | " << boardStr[15] << " " << endl;
    cout << endl;
}

string* CreateBoardString(int* h_boardOfNums)
{
    string* boardStr = new string[16];
    for (int i = 0; i < 16; ++i)
    {
        if (h_boardOfNums[i] == -1)
        {
            boardStr[i] = "X";
        } else if (h_boardOfNums[i] == 0)
        {
            boardStr[i] = "-";
        } else 
        {
            boardStr[i] = "O";
        }
    }
    return boardStr;
}


int main()
{

    cout << "**************************************\n";
    cout << "* 2 GPUs compete in 4x4 Tic-Tack-Toe *\n";
    cout << "**************************************\n\n";
    cout << endl;
    
    int nDevices;

    cudaGetDeviceCount(&nDevices);

    if ( nDevices >= 1)
    {
        cout << "Number of CUDA capable devices = " << nDevices << endl;
    }
    else
    {
        cout << "No CUDA capable devices found." << endl;
        return 1;
    }


    // need to set up the 4x4=64 board
    // to pass to the device, it will just be a int* array of 0, 1 & 2
    // 0 = empty = "-"
    // 1 = X
    // 2 = O
    // initially, everything will be 0 which will be represented 

    string boardStr[16];
    for (int i = 0; i < 16; ++i)
    {
        boardStr[i] = "-";
        //cout << boardStr[i] <<" ";
    }
    
    // DrawBoard(boardStr, "Starting Board");

    // set up solution vectors for 10 possible solution. hz row, vt col, diagonal.
    int h_SV01[16] = {  1,1,1,1, 0,0,0,0, 0,0,0,0, 0,0,0,0 }; // top row
    int h_SV02[16] = {  0,0,0,0, 1,1,1,1, 0,0,0,0, 0,0,0,0 }; // 2nd row
    int h_SV03[16] = {  0,0,0,0, 0,0,0,0, 1,1,1,1, 0,0,0,0 }; // 3rd row
    int h_SV04[16] = {  0,0,0,0, 0,0,0,0, 0,0,0,0, 1,1,1,1 }; // bot row
    int h_SV05[16] = {  1,0,0,0, 1,0,0,0, 1,0,0,0, 1,0,0,0 }; // 1st col
    int h_SV06[16] = {  0,1,0,0, 0,1,0,0, 0,1,0,0, 0,1,0,0 }; // 2nd col
    int h_SV07[16] = {  0,0,1,0, 0,0,1,0, 0,0,1,0, 0,0,1,0 }; // 3rd col
    int h_SV08[16] = {  0,0,0,1, 0,0,0,1, 0,0,0,1, 0,0,0,1 }; // 4th col
    int h_SV09[16] = {  1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1 }; // NW-SE diag
    int h_SV10[16] = {  0,0,0,1, 0,0,1,0, 0,1,0,0, 1,0,0,0 }; // SW-NE diag

    // GPU #1, deviceNum = 0

    //Allocate memory on GPU #1 for solution vectors

    // Loop over GPU1 & GPU2
    // If 2 GPUs, both 0 & 1 will be loaded with SVs and of course, there will be d0_ and d1_ pointers defined
    // If only 1 GPU, will still explicitly be d0_ & d1_ arrays but both will be on 0 gpu
    // If more than 2 GPUs, only 0 & 1 will be loaded with SVs

 


    // h_forDotProduct
    int* d0_forDotProduct = 0; // will differ for GPU1 vs GPU2
    //                            -1000 for others piece + 10 for your pieces, +1 for open
    cudaMalloc((void**)&d0_forDotProduct, 16 * sizeof(int));
    // pointer in device for d_forDotProduct now defined on device

    int* d0_SV01 = 0;
    int* d0_SV02 = 0;
    int* d0_SV03 = 0;
    int* d0_SV04 = 0;
    int* d0_SV05 = 0;
    int* d0_SV06 = 0;
    int* d0_SV07 = 0;
    int* d0_SV08 = 0;
    int* d0_SV09 = 0;
    int* d0_SV10 = 0;
    cudaMalloc((void**)&d0_SV01, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV02, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV03, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV04, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV05, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV06, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV07, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV08, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV09, 16 * sizeof(int));
    cudaMalloc((void**)&d0_SV10, 16 * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(d0_SV01, h_SV01, 16 * sizeof(int), cudaMemcpyHostToDevice);

    int hd0_SV01[16];
    cudaMemcpy(hd0_SV01, d0_SV01, 16 * sizeof(int), cudaMemcpyDeviceToHost);
    // cout << "hd0_SV01: ";
    for (int i = 0; i< 16; ++i )
    {
        // cout << hd0_SV01[i] << " ";
    }
    cout << endl;

    cudaMemcpy(d0_SV02, h_SV02, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV03, h_SV03, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV04, h_SV04, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV05, h_SV05, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV06, h_SV06, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV07, h_SV07, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV08, h_SV08, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV09, h_SV09, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d0_SV10, h_SV10, 16 * sizeof(int), cudaMemcpyHostToDevice);

    // need to set up memory for array multiplies the SVs with the baord array
    // cannot sum array in device. will return 10 array multiplies back and dot sum will be done on host
    int* d0_dot01 = 0;
    int* d0_dot02 = 0;
    int* d0_dot03 = 0;
    int* d0_dot04 = 0;
    int* d0_dot05 = 0;
    int* d0_dot06 = 0;
    int* d0_dot07 = 0;
    int* d0_dot08 = 0;
    int* d0_dot09 = 0;
    int* d0_dot10 = 0;
    cudaMalloc((void**)&d0_dot01, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot02, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot03, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot04, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot05, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot06, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot07, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot08, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot09, 16 * sizeof(int));
    cudaMalloc((void**)&d0_dot10, 16 * sizeof(int));

    // we will be explicitly defining all of the array for the 2nd GPU
    // in the case of only 1 GPU, all of the arrays will be copied onto the first GPU
    // need to change device if there are 2 GPUs and pointing to 2nd GPU
    // NOte: "1" refers to the 2nd GPU. 1st GPU is "0"
    if (nDevices > 1)
    {
        cudaSetDevice(1);
    }


    // h_forDotProduct
    int* d1_forDotProduct = 0; // will differ for GPU1 vs GPU2
    //                            -1000 for others piece + 10 for your pieces, +1 for open
    cudaMalloc((void**)&d1_forDotProduct, 16 * sizeof(int));
    // pointer in device for d_forDotProduct now defined on device

    int* d1_SV01 = 0;
    int* d1_SV02 = 0;
    int* d1_SV03 = 0;
    int* d1_SV04 = 0;
    int* d1_SV05 = 0;
    int* d1_SV06 = 0;
    int* d1_SV07 = 0;
    int* d1_SV08 = 0;
    int* d1_SV09 = 0;
    int* d1_SV10 = 0;
    cudaMalloc((void**)&d1_SV01, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV02, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV03, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV04, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV05, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV06, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV07, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV08, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV09, 16 * sizeof(int));
    cudaMalloc((void**)&d1_SV10, 16 * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(d1_SV01, h_SV01, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV02, h_SV02, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV03, h_SV03, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV04, h_SV04, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV05, h_SV05, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV06, h_SV06, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV07, h_SV07, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV08, h_SV08, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV09, h_SV09, 16 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d1_SV10, h_SV10, 16 * sizeof(int), cudaMemcpyHostToDevice);

    // need to set up memory for array multiplies the SVs with the baord array
    // cannot sum array in device. will return 10 array multiplies back and dot sum will be done on host
    int* d1_dot01 = 0;
    int* d1_dot02 = 0;
    int* d1_dot03 = 0;
    int* d1_dot04 = 0;
    int* d1_dot05 = 0;
    int* d1_dot06 = 0;
    int* d1_dot07 = 0;
    int* d1_dot08 = 0;
    int* d1_dot09 = 0;
    int* d1_dot10 = 0;
    cudaMalloc((void**)&d1_dot01, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot02, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot03, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot04, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot05, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot06, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot07, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot08, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot09, 16 * sizeof(int));
    cudaMalloc((void**)&d1_dot10, 16 * sizeof(int));

    // the arrays for the 2 GPUs are all set
    // d0_ is 1st GPU and d1_ is the 2nd GPU


    // decide who goes first
    srand(static_cast<unsigned int>(time(0))); // Seed the random number generator

    // Generate a random number 0 or 1
    int randomNum = rand() % 2;
    // cout << "randomNum = " << randomNum << endl;

    int GPU_num;
    if (randomNum == 0)
    {
        GPU_num = -1;   // GPU1 first
        cout << "1st GPU starts";
    }
    else
    {
        GPU_num = 1; // GPU2 first
        cout << "2nd GPU starts";
    }

    int h_boardOfNums[16] = { 0 }; // this is -1 for GPU1, 0 for open, +1 for GPU2, initially, all set to 0

    // Here is where we set the initial turn
    // different start depending on if GPU1 is first or GPU2 is first
    // but no cuda calls to the kernel yet
   
    if (GPU_num == -1)
    {
        // need to udate the board with first guess at top right for GPU#1
        // int* h_boardOfNums[16] = 0; // this is -1 for GPU2, 0 for open +1 for GPU1
        h_boardOfNums[0] = -1;

    }
    else
    {
        // GPU2 only picks random space to start
        int randomNum = rand() % 15;
        h_boardOfNums[randomNum] = 1;
    }
    string* iniBoardStr = CreateBoardString(h_boardOfNums);

    cout << endl;
    DrawBoard(iniBoardStr, "End of First Turn");

    // first turn is done
    // time to switch GPU
    // since GPU_num = -1 or +1, all we need to do is to multiply by -1 to select other GPUs
    GPU_num = GPU_num * (-1);

    //cout << "New GPU_num for 2nd turn = " << GPU_num << endl;
    //cout << "h_boardOfNums: ";
    for (int i=0; i < 16; ++i)
    {
        //cout << h_boardOfNums[i] << " ";
    }
    //cout << endl;

    // here is the loop for the turns
    // will break if 4 in a row
    for (int i = 0; i < 16; ++i)
    {
        
        if (nDevices > 1 && GPU_num == 1)
        {
            cudaSetDevice(1);
        }
        else
        {
            cudaSetDevice(0);
        }
        // correct cuda device now set
        // will also have to use GPU_num as switch for kernel call as well as copyBack to host


        // need to create int array good for multiplying the 10 solution vectors
        // different for each GPU
        if (GPU_num == -1) 
        {
            cout << "\n\nNow 1st GPU\n";
            cout << "h_boardOfNums:           ";
            for (int i = 0; i < 16; ++i)
            {
                cout << setw(6) << h_boardOfNums[i];
            }
            cout << endl;

            cout << "Vector for SV multiply : ";
            int h_forDotProduct[16];
            for (int j =0; j < 16; ++j)
            {
                //cout << "h_boardOfNums[j] = " << h_boardOfNums[j] << endl;
                if ( h_boardOfNums[j] == -1 )
                {
                    //cout << "if true";
                    h_forDotProduct[j] = 10;
                }
                else if (h_boardOfNums[j] == 1)
                {
                    //cout << "else if true";
                    h_forDotProduct[j] = -100;
                }
                else if (h_boardOfNums[j] == 0)
                {
                    h_forDotProduct[j] = 1;
                    //cout << "else true: j= " << j << "  h_boardOfNums[j] = " << h_boardOfNums[j] << endl;
                    //cout << "forDotProduct[j] = " << forDotProduct[j] << endl;
                }
                else
                {
                    cout << "Trouble: h_BoardOfNums[j] noy -1,0, or 1" << endl;
                }
                cout << setw(6) << h_forDotProduct[j] ;
            }

            cout << endl;

            // time to copy h_forDotProduct to d_forDotProduc
            // cudaMalloc((void**)&d0_forDotProduct, 16 * sizeof(int));
            cudaMemcpy(d0_forDotProduct,h_forDotProduct, 16 * sizeof(int), cudaMemcpyHostToDevice);

            SVKernel<<<1,16>>>(d0_forDotProduct,
                d0_SV01, d0_SV02, d0_SV03, d0_SV04, d0_SV05,
                d0_SV06, d0_SV07, d0_SV08, d0_SV09, d0_SV10,
                d0_dot01, d0_dot02, d0_dot03, d0_dot04, d0_dot05,
                d0_dot06, d0_dot07, d0_dot08, d0_dot09, d0_dot10);

            cudaDeviceSynchronize();

            int hd_dot01[16], hd_dot02[16], hd_dot03[16], hd_dot04[16], hd_dot05[16], hd_dot06[16], hd_dot07[16], hd_dot08[16], hd_dot09[16], hd_dot10[16];

            cudaMemcpy(hd_dot01, d0_dot01, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot02, d0_dot02, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot03, d0_dot03, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot04, d0_dot04, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot05, d0_dot05, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot06, d0_dot06, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot07, d0_dot07, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot08, d0_dot08, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot09, d0_dot09, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot10, d0_dot10, 16 * sizeof(int), cudaMemcpyDeviceToHost);

            int SP01 = 0, SP02 = 0, SP03 = 0, SP04 = 0, SP05 = 0, SP06 = 0, SP07 = 0, SP08 = 0, SP09 = 0, SP10 = 0;
            int SP01_g=0, SP02_g=0, SP03_g=0, SP04_g=0, SP05_g=0, SP06_g=0, SP07_g=0, SP08_g=0, SP09_g=0, SP10_g=0; // will be index of last open in SV
            //                                                                                                      // will change if "1" is encouterred in the SV vector
            //                                                                                                      // will always return the index of the last 1 in the SV vector
            //cout << "dot01: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot01[j] << " ";
                SP01 += hd_dot01[j];
                if (hd_dot01[j] == 1)
                {
                    SP01_g = j;
                }
            }
            //cout << endl;
            //cout << "dot02: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot02[j] << " ";
                SP02 += hd_dot02[j];
                if (hd_dot02[j] == 1)
                {
                    SP02_g = j;
                }
            }
            //cout << endl;
            //cout << "dot03: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot03[j] << " ";
                SP03 += hd_dot03[j];
                if (hd_dot03[j] == 1)
                {
                    SP03_g = j;
                }
            }
            //cout << endl;
            //cout << "dot04: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot04[j] << " ";
                SP04 += hd_dot04[j];
                if (hd_dot04[j] == 1)
                {
                    SP04_g = j;
                }
            }
            //cout << endl;
            //cout << "dot05: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot05[j] << " ";
                SP05 += hd_dot05[j];
                if (hd_dot05[j] == 1)
                {
                    SP05_g = j;
                }
            }
            //cout << endl;
            //cout << "dot06: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot06[j] << " ";
                SP06 += hd_dot06[j];
                if (hd_dot06[j] == 1)
                {
                    SP06_g = j;
                }
            }
            //cout << endl;
            //cout << "dot07: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot07[j] << " ";
                SP07 += hd_dot07[j];
                if (hd_dot07[j] == 1)
                {
                    SP07_g = j;
                }
            }
            //cout << endl;
            //cout << "dot08: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot08[j] << " ";
                SP08 += hd_dot08[j];
                if (hd_dot08[j] == 1)
                {
                    SP08_g = j;
                }
            }
            //cout << endl;
            //cout << "dot09: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot09[j] << " ";
                SP09 += hd_dot09[j];
                if (hd_dot09[j] == 1)
                {
                    SP09_g = j;
                }
            }
            //cout << endl;
            //cout << "dot10: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot10[j] << " ";
                SP10 += hd_dot10[j];
                if (hd_dot10[j] == 1)
                {
                    SP10_g = j;
                }
            }
            //cout << endl;
            //cout << "SP Best guess index: " << SP01_g << " " << SP02_g << " " << SP03_g << " " << SP04_g
            //     << " " << SP05_g << " " << SP06_g << " " << SP07_g << " " << SP08_g << " " << SP09_g << " " << SP10_g << endl;

            //cout << "Scalar Product Array:" << SP01 << " " << SP02 << " " << SP03 << " " << SP04 << " " << SP05 << " " 
            //                                << SP06 << " " << SP07 << " " << SP08 << " " << SP09 << " " << SP10 << endl;
            
            int SP_values[10] = { SP01,SP02,SP03,SP04,SP05,SP06,SP07,SP08,SP09,SP10 };
            // looking for the max SP. If three are 3 X, it will be 10+10+10+1 = 31
            //  If 31 and you guess the last open place, game over!
            // Loop below is just idenifying the max Scalar Product 
            int max_SP = -1000;
            int max_SP_index = 0;
            for (int j = 0; j < 10; ++j)
            {
                if ( SP_values[j]  > max_SP )
                {
                    max_SP = SP_values[j];
                    max_SP_index = j;
                }
                // cout << "loop index: " << j << "  max_SP: " << max_SP << "  max_SP_index: " << max_SP_index << endl;
            }

            if (max_SP == 40 )
            {
                cout << "\n4 in a row for GPU1\n";
                cout << "end of game";
                return 0;
            }

            // we are on the GPU_num = -1 side. New guess needs to be set to -1. Subscript is given by SPXX_g 
            if (max_SP_index == 0)
            {
                h_boardOfNums[SP01_g] = -1;
            }
            else if (max_SP_index == 1)
            {
                h_boardOfNums[SP02_g] = -1;
            }
            else if (max_SP_index == 2)
            {
                h_boardOfNums[SP03_g] = -1;
            }
            else if (max_SP_index == 3)
            {
                h_boardOfNums[SP04_g] = -1;
            }
            else if (max_SP_index == 4)
            {
                h_boardOfNums[SP05_g] = -1;
            }
            else if (max_SP_index == 5)
            {
                h_boardOfNums[SP06_g] = -1;
            }
            else if (max_SP_index == 6)
            {
                h_boardOfNums[SP07_g] = -1;
            }
            else if (max_SP_index == 7)
            {
                h_boardOfNums[SP08_g] = -1;
            }
            else if (max_SP_index == 8)
            {
                h_boardOfNums[SP09_g] = -1;
            }
            else if (max_SP_index == 9)
            {
                h_boardOfNums[SP10_g] = -1;
            }
            else 
            {
                cout << "trouble deciding next guess\n";
            }

            // If max_SP = 31, next guess should be 4 in a row and game should end
            if (max_SP == 31)
            {

                string* iniBoardStr = CreateBoardString(h_boardOfNums);
                cout << endl;
                DrawBoard(iniBoardStr, "Four in a row for 1st GPU ");
                cout << "\n";
                cout << "\n4 in a row for GPU1\n";
                cout << "**********end of game************";
                return 0;
            }

        }
        else if (GPU_num == 1)  
        {
            cout << "\n\nNow 2nd GPU.\n";

            cout << "h_boardOfNums:           ";
            for (int i = 0; i < 16; ++i)
            {
                cout << setw(6) << h_boardOfNums[i];
            }
            cout << endl;

            cout << "Vector for SV multiply : ";
            int h_forDotProduct[16];
            for (int j = 0; j < 16; ++j)
            {
                if (h_boardOfNums[j] == -1)   //this is GPU0...1st GPU
                {
                    h_forDotProduct[j] = -100;
                }
                else if (h_boardOfNums[j] == 1)  // this is GPU1...2nd GPU
                {
                    h_forDotProduct[j] = 10;
                }
                else if (h_boardOfNums[j] == 0)
                {
                    h_forDotProduct[j] = 1;
                }
                else
                {
                    cout << "Trouble: h_BoardOfNums[j] noy -1,0, or 1" << endl;
                }
                cout << setw(6) << h_forDotProduct[j];

            }
            
            cout << endl;

            // time to copy h_forDotProduct to d_forDotProduc
            // cudaMalloc((void**)&d0_forDotProduct, 16 * sizeof(int));
            cudaMemcpy(d1_forDotProduct, h_forDotProduct, 16 * sizeof(int), cudaMemcpyHostToDevice);

            SVKernel << <1, 16 >> > (d1_forDotProduct,
                d1_SV01,  d1_SV02,  d1_SV03,  d1_SV04,  d1_SV05,
                d1_SV06,  d1_SV07,  d1_SV08,  d1_SV09,  d1_SV10,
                d1_dot01, d1_dot02, d1_dot03, d1_dot04, d1_dot05,
                d1_dot06, d1_dot07, d1_dot08, d1_dot09, d1_dot10);


            cudaDeviceSynchronize();

            int hd_dot01[16], hd_dot02[16], hd_dot03[16], hd_dot04[16], hd_dot05[16], hd_dot06[16], hd_dot07[16], hd_dot08[16], hd_dot09[16], hd_dot10[16];

            cudaMemcpy(hd_dot01, d1_dot01, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot02, d1_dot02, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot03, d1_dot03, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot04, d1_dot04, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot05, d1_dot05, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot06, d1_dot06, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot07, d1_dot07, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot08, d1_dot08, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot09, d1_dot09, 16 * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(hd_dot10, d1_dot10, 16 * sizeof(int), cudaMemcpyDeviceToHost);

            int SP01 = 0, SP02 = 0, SP03 = 0, SP04 = 0, SP05 = 0, SP06 = 0, SP07 = 0, SP08 = 0, SP09 = 0, SP10 = 0;
            int SP01_g = 0, SP02_g = 0, SP03_g = 0, SP04_g = 0, SP05_g = 0, SP06_g = 0, SP07_g = 0, SP08_g = 0, SP09_g = 0, SP10_g = 0; // will be index of last open in SV
            //                                                                                                      // will change if "1" is encouterred in the SV vector
            //                                                                                                      // will always return the index of the last 1 in the SV vector
            //cout << "dot01: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot01[j] << " ";
                SP01 += hd_dot01[j];
                if (hd_dot01[j] == 1)
                {
                    SP01_g = j;
                }
            }
            //cout << endl;
            //cout << "dot02: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot02[j] << " ";
                SP02 += hd_dot02[j];
                if (hd_dot02[j] == 1)
                {
                    SP02_g = j;
                }
            }
            //cout << endl;
            //cout << "dot03: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot03[j] << " ";
                SP03 += hd_dot03[j];
                if (hd_dot03[j] == 1)
                {
                    SP03_g = j;
                }
            }
            //cout << endl;
            //cout << "dot04: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot04[j] << " ";
                SP04 += hd_dot04[j];
                if (hd_dot04[j] == 1)
                {
                    SP04_g = j;
                }
            }
            //cout << endl;
            //cout << "dot05: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot05[j] << " ";
                SP05 += hd_dot05[j];
                if (hd_dot05[j] == 1)
                {
                    SP05_g = j;
                }
            }
            //cout << endl;
            //cout << "dot06: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot06[j] << " ";
                SP06 += hd_dot06[j];
                if (hd_dot06[j] == 1)
                {
                    SP06_g = j;
                }
            }
            //cout << endl;
            //cout << "dot07: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot07[j] << " ";
                SP07 += hd_dot07[j];
                if (hd_dot07[j] == 1)
                {
                    SP07_g = j;
                }
            }
            //cout << endl;
            //cout << "dot08: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot08[j] << " ";
                SP08 += hd_dot08[j];
                if (hd_dot08[j] == 1)
                {
                    SP08_g = j;
                }
            }
            //cout << endl;
            //cout << "dot09: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot09[j] << " ";
                SP09 += hd_dot09[j];
                if (hd_dot09[j] == 1)
                {
                    SP09_g = j;
                }
            }
            //cout << endl;
            //cout << "dot10: ";
            for (int j = 0; j < 16; ++j)
            {
                //cout << setw(5) << hd_dot10[j] << " ";
                SP10 += hd_dot10[j];
                if (hd_dot10[j] == 1)
                {
                    SP10_g = j;
                }
            }
            //cout << endl;
            //cout << "SP Best guess index: " << SP01_g << " " << SP02_g << " " << SP03_g << " " << SP04_g
            //     << " " << SP05_g << " " << SP06_g << " " << SP07_g << " " << SP08_g << " " << SP09_g << " " << SP10_g << endl;

            //cout << "Scalar Product Array:" << SP01 << " " << SP02 << " " << SP03 << " " << SP04 << " " << SP05 << " " 
            //                                << SP06 << " " << SP07 << " " << SP08 << " " << SP09 << " " << SP10 << endl;


            //cout << "Scalar Product Array:" << SP01 << " " << SP02 << " " << SP03 << " " << SP04 << " " << SP05 << " " 
            //                                << SP06 << " " << SP07 << " " << SP08 << " " << SP09 << " " << SP10 << endl;

            int SP_values[10] = { SP01,SP02,SP03,SP04,SP05,SP06,SP07,SP08,SP09,SP10 };
            // looking for the max SP. If three are 3 X, it will be 10+10+10+1 = 31
            //  If 31 and you guess the last open place, game over!
            // Loop below is just idenifying the max Scalar Product 
            int max_SP = -1000;
            int max_SP_index = 0;
            for (int j = 0; j < 10; ++j)
            {
                if (SP_values[j] > max_SP)
                {
                    max_SP = SP_values[j];
                    max_SP_index = j;
                }
                // cout << "loop index: " << j << "  max_SP: " << max_SP << "  max_SP_index: " << max_SP_index << endl;
            }

            if (max_SP == 40)
            {
                cout << "\n4 in a row for 2nd GPU\n";
                cout << "end of game";
                return 0;
            }

            // we are on the GPU_num = +1 side. New guess needs to be set to +1. Subscript is given by SPXX_g 
            if (max_SP_index == 0)
            {
                h_boardOfNums[SP01_g] = 1;
            }
            else if (max_SP_index == 1)
            {
                h_boardOfNums[SP02_g] = 1;
            }
            else if (max_SP_index == 2)
            {
                h_boardOfNums[SP03_g] = 1;
            }
            else if (max_SP_index == 3)
            {
                h_boardOfNums[SP04_g] = 1;
            }
            else if (max_SP_index == 4)
            {
                h_boardOfNums[SP05_g] = 1;
            }
            else if (max_SP_index == 5)
            {
                h_boardOfNums[SP06_g] = 1;
            }
            else if (max_SP_index == 6)
            {
                h_boardOfNums[SP07_g] = 1;
            }
            else if (max_SP_index == 7)
            {
                h_boardOfNums[SP08_g] = 1;
            }
            else if (max_SP_index == 8)
            {
                h_boardOfNums[SP09_g] = 1;
            }
            else if (max_SP_index == 9)
            {
                h_boardOfNums[SP10_g] = 1;
            }
            else
            {
                cout << "trouble deciding next guess\n";
            }

            // If max_SP = 31, next guess should be 4 in a row and game should end
            if (max_SP == 31)
            {

                string* iniBoardStr = CreateBoardString(h_boardOfNums);
                cout << endl;
                DrawBoard(iniBoardStr, "Four in a row for 2nd GPU ");
                cout << "\n";
                cout << "\n4 in a row for 2nd GPU\n";
                cout << "**********end of game************";
                return 0;
            }

            
        }
        else
        {
            cout << "GPU_num must be -1(GPU1) or +1(GPU2)\n";
        }

        // correct  GPU set above

        // h_boardOfNums changed
        string* iniBoardStr = CreateBoardString(h_boardOfNums);

        cout << endl;
        DrawBoard(iniBoardStr, "End of Turn");

        //switch to other GPU
        GPU_num = GPU_num * (-1);
    }




    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    // Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    // }

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaDeviceReset();


    return 0;  // 0 indicates successful execution like error = 0
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
