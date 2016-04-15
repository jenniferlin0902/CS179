CS179 
Jennifer, Chih Wen, Lin 

Part I 
Question 1.1 
An arithmetic instrction on a GPU take around 10 ns and the clock is around 1GHz. In GTK110, each SM contains 4 warp scheduler with dual 
instruction dispatch unix. Thus, to hide the latency, we need to initiate
4 * 2 * 10 = 80 instructions. 

Question 1.2 
a) 
This code does not diverage. 
Note that any thread with the same threadIdx.y will be in the same 
warp. idx % 32 = threadIdx.y. So, for all thread in the same warp
it will all have the same result on the if statement. Thus, the code 
does not diverage. 

b) 
This code diverages. For each thread with different threadIdx.x 
that is in the same warp, the number of loop need to go through 
in the for loop is different. 

Question 1.3 
a)
This write is coalesced since subsequent 32 data is accessed by 
different thread in the same warp at the same time. This will hit 
1 cache line. 

b) 
Note that in a warp, all threadIdx.y is the same and threadIdx.x ranges from 0 to 31. This write has stride of 32 float, which is 128 byte. Thus, 
each thread will require a cache load. Total cache line hit per warp
is 32. 

c) 
This write is not coalesced, each warp will try to access data 
from data[32 * n + 1] to data [32 *n + 33]. Note that data[32*n] to 
data[32*n + 31] will be loaded into the same cache, while accessing 
data[32*n + 33] will require an extra cache load. Thus, the 
cache hit is 2. 


Question 1.4 

a) No. On lhs, thre is no bank conflict since for each thread, it will access a column, whre each column is located in a different bank. For rhs,
there is also no bank conflict. Since within a warp, j are identical, 
so all threads always access the same element. 


b) 
line1 : 
l1 = lhs[i + 32 * k]
r1 = rhs[k + 128 * j]
out = output[i + 32 * j]
sum = l1 *r1 + out; 
output[i + 32 * j] = sum; 

line2:
l2 = lhs[i + 32 * (k+1)]
r2 = rhs[k + 128 * j]
out2 = output[i + 32 * j]
sum = l2 * r2 + out2; 
output[i + 32 * j] = sum; 

c) 
For each line,
Instruction 4 is dependent on instruciton 1-3 and instruction 5 is dependent on the result of instruction 4. 

In addition, instruction 3 in line 2 is dependent on instruction 5 on line1.

d) 

output[i + 32*j] += lhs[i + 32 * k] * rhs[k + 128 * j] + hs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j]; 


e) 
Aside from the minimizing the instruction dependency, we can unwind some loop in each thread. 
Currently, each thread goes through 64 loop, where each loop it performs two product sums. We can unwind some loop to speed up the process. 
For example, 

for (int k = 0; k < 128; k += 4) {
    output[i + 32 * j] += lhs[i + 32 * k] * rhs[k + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 1)] * rhs[(k + 1) + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 2)] * rhs[(k + 2) + 128 * j];
    output[i + 32 * j] += lhs[i + 32 * (k + 3)] * rhs[(k + 3) + 128 * j];

}

And in each loop we should repeat step d, to minimize the instruction dependencies. 



Part II 

Index of the GPU with the lowest temperature: 1 (55 C)
Time limit for this program set to 1000 seconds
Size 512 naive CPU: 0.801184 ms
Size 512 GPU memcpy: 0.034592 ms
Size 512 naive GPU: 0.100384 ms
Size 512 shmem GPU: 0.033792 ms
Size 512 optimal GPU: 0.030560 ms

Size 1024 naive CPU: 3.858112 ms
Size 1024 GPU memcpy: 0.086944 ms
Size 1024 naive GPU: 0.295520 ms
Size 1024 shmem GPU: 0.095808 ms
Size 1024 optimal GPU: 0.088512 ms

Size 2048 naive CPU: 37.389599 ms
Size 2048 GPU memcpy: 0.266656 ms
Size 2048 naive GPU: 1.151872 ms
Size 2048 shmem GPU: 0.320704 ms
Size 2048 optimal GPU: 0.306272 ms

Size 4096 naive CPU: 155.413116 ms
Size 4096 GPU memcpy: 1.004832 ms
Size 4096 naive GPU: 4.116608 ms
Size 4096 shmem GPU: 1.257120 ms
Size 4096 optimal GPU: 1.187328 ms


BONUS 
The 2 calls is worse the vec_add. 
1. 
For the 2-call, we need 4 memory loading and 2 memory writing per loop in the calculation. 
For the three eleement vec_add, we need 3 memory loading and 1 memory writing per loop. Thus, the later is more efficient. 

2. 
In addition, the vec_add(a, z, a, n) is not parallelizable. since output z denpends on input a. 



