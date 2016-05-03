Chih Wen Lin
CS179 Lab4 

Question 1.1 

No. Note that to parallelize each loop, each thread that loop on 
frontier will need to access X, C, F to the next layer. Since
the topology of the graph is unknown, there is no effective way to
avoid bank conflict. Also, if the graph is big, the global array F,
X, C won't fit into shared memory. In that case, we might need to 
load the global memory more than once in the kernal. 

Question 1.2 
If F is not all false, then sum over the array will be 0. Thus, a parallelizable way to perform this task is to store F is shared memory and use reduction to sum over the shared memory. Last, use atomic operation to determine the total sum. If the total sum is 0, then we know F is all false.

Question 1.3 
We can merge the code that check "while F is not all false" into the GPU kernal. 
We may keep a global flag to indicate if the while loop in CPU code 
should continue or not. 

int finish_flag = 0;

while(!finish_flag){
	call GPU_kernal(F, X, C, Va, Ea, finish_flag)
}

In the kernal 

finish_flag = true;
if F[threadId] is true:
 	F[threadId] <- false
 	X[threadId] <- true
 	
 	for Ea[Va[threadId]] ≤ j < Ea[Va[threadId + 1]]:
 		if X[j] is false:
 			C[j] <- C[threadId] + 1
 			F[j] <- true	
 			finish_flag = false;
}

Note that the flag setting operation takes a global write or 
shared memory write if the user chose to place the finish flag 
in shared memory. However, even with shared memory, will still run 
into bank conflict. 
The advantage of this method is that we don't need to 
launch a seperate kernal to do reduction everytime. 
The disadvantage of this method is that the flag setting operation
may slow down the GPU, especially if the graph is dense. 
(if we need to go through the for loop multiple time in each 
thread)

If the graph is sparse, then this method is better. If the graph 
is dense then method 1 which use a seperate kernal is bettter. 


Question 2 
Note that since each measurement will correspond to multiple crystal and multiple measurement may correspond to the same crystal. Thus, each thread may be acccessing the same pixel data at the same time; adding measurement to each pixel need to be atomic. NOte that now the access to sinogram reads are not coalesced nor textured. Since the emitions is a stochastic process, the location of d_i on a sinogram is random. 

With these two reasoning, we should expect the PET reconstruction to be slower than that of X-ray CT. 



