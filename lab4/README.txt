Question 1.2 
If F is not all false, then sum over the array will be 0. Thus, a parallelizable way to perform this task is to store F is shared memory and use reduction to sum over the shared memory. Last, use atomic operation to determine the total sum. If the total sum is 0, then we know F is all false.

Question 1.3 
