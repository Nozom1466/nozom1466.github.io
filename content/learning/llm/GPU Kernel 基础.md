---
title: 'GPU Kernel 基础'
date: 2026-01-19T22:11:19-05:00
slug:
summary: 简单的 GPU Kernel 编程基础
description:
cover:
    image:
    alt:
    caption:
    relative: false
showtoc: false
draft: false
tags: ['cuda', 'kernel', 'llm', 'system']
categories:
---


这个 GPU 编程搞的非常头疼，主要原因是涉及到 thread 并行以及不了解 cuda 运行的主要流程。

代码参考:  [llmsys assignment 1 official codebase](https://github.com/llmsystem/llmsys_hw1)



### GPU 的结构管理

**GPU 物理结构：** GPU 里边有多个 Streaming Multiprocessor, 是真正执行 kernel 的硬件计算单元。一个 SM 里边有 CUDA cores, Tensor cores, warp scheduler，寄存器文件，shared memory 等。在这个上边运行的最小执行单位是一个 warp。这个 warp 里边有 32 个 threads，是 nvidia 硬件规定。我们在编程的时候不需要写 warp 调用，但是需要意识到这个存在，比如说一个 warp 内有 if-else 判断走不同分支的时候可能会影响效率之类。

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20260119120234646.png" width="400" caption="Streaming Multiprocessors" align="center">}}



**GPU 概念结构：** 在一块 GPU 里主要是 Grid -> Block -> (Warp) -> Thread 的概念结构。在Grid 中有组织好的 Block 结构，这个结构是一个三维结构，具有 x, y, z 坐标：`blockIdx.x, blockIdx.y, blockIdx.z`，并且有每个坐标轴的单位长度：`blockDim.x, blockDim.y, blockDim.z`，nvidia 的 GPU 需要满足：`blockDim.x * blockDim.y * blockDim.z ≤ 1024` 。在此基础上能够通过这个坐标系定位到 block 的坐标。我们可以在初始化的时候指定 block 的各个维度，设置一个 block 中有多少 thread。但是考虑到一个 warp 是 32 个 threads，我们最好能设置成 32 的倍数，防止最后 warp 真正执行的时候影响效率。并且在每个方向上能够初始化的最大 threads 数量也是有限制的。在 [官方文档](https://docs.nvidia.com/cuda/archive/9.2/pdf/CUDA_Runtime_API.pdf) 中，这个最大的 threads 数量需要你通过访问你的 GPU 得到。在一个 block 内部有组织好的 Thread 结构，同样具有三维坐标: `threadIdx.x, threadIdx.y, threadIdx.z`。这个坐标是在某一个 block *内部* 的坐标。一个初始化的示例：

```C++
// 1 dim
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
kernel<<<blocksPerGrid, threadsPerBlock>>>(...);

//2 dim
dim3 block(16, 16);   // 256 threads
dim3 grid(
    (width  + block.x - 1) / block.x,
    (height + block.y - 1) / block.y
);

kernel<<<grid, block>>>(...);

```

如果想要访问一个 thread 在 grid 中的具体坐标：

```c++
// 1 dim
int x = blockIdx.x * blockDim.x + threadIdx.x;

// 2 dim
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```

**Kernel 编程与内存访问：** 在编程的时候，GPU 会为 block 当中的每个 thread 启动一份 kernel，但是一个 kernel里边是可以调用多个 threads 来操作的。这里要注意在 block 中的每一个 thread 都要执行这个 kernel，所以一般要进行边界检查。在同一个 block 内，你可以通过 `__syncthreads();` 来同步一个 block 中所有 threads 的行为，但是跨 block 不行。在一个 block 内的 threads 可以访问一块 shared_memory，在一个 grid 内的所有 block 中的 threads 可以访问 global memory。不同 Block 之间 **不能** 共享 shared_memory ，但是能访问同一块 global_memory。这两个 memory 的访问速度就 *类似于* 多级缓存的访问速度。 global 快一点，shared 慢一点。 

**一些关系说明：** grid, block, threads 属于概念层面描述，SM，warp是在物理执行层面描述，在真正执行的时候，一个 block 内的所有 threads 将会被分配到一个 SM 上执行，warp 是执行的最小单元。这也解释了为什么 shared_memory 不能跨 Block 访问，因为可能不再一个 SM 上，即使在一个 SM 上，SM 分配给每个 Block 的 shared memory 也是互相独立的。

**跨 GPU 通信：** 包括GPU 和 CPU/其他 GPU 之间的通信，前者通过内存拷贝比如说 `cudaMemcpy` ，后者可以用  `NNCL`（通信库） 通过 `NVLink` 通信。其中 `NVLink` 专用于 GPU 间通信，传输速度快；`PCIe` 是通用数据总线，传数据都可以用，但是比较慢。



### Kernel 编写的概念与例子

就是注意普通写法就是一个 thread 处理一个 out 元素位置，并行就是多个 thread 处理一个 out 元素位置。并且在一个 kernel 中，thread 既可以扮演搬运数据角色，也可以扮演计算的角色。

**映射关系：*一维存储数组* 与 *高维概念数组*  的映射关系：** 简单来说，在 kernel 看来你所谓的二维、三维等等数组其实全都以 **1维** 的形式来存储。也就是说在 kernel 中，你需要手动进行 *一维存储数组* 与 *高维概念数组* 之间的映射。比如说，一个二维数组的映射关系：

```
2 dim tensor: [2, 3]
[[1, 2, 3], 
[4, 5, 6]]

stored as 1 dim tensor: [6]
- stride = [3, 2]: [1, 2, 3, 4, 5, 6] <- kernel 实际看到的
- stride = [2, 1]: [1, 4, 2, 5, 3, 6]
```

其中多种存储方式主要是取决于 **stride** 的设置。你可以通过使用 stride 快速完成从*一维存储数组* 到 *高维概念数组* 的映射。stride 的一个语义含义就是，你想要在 *一维存储数组* 当中访问 *高维概念数组*  的在这个维度的下一个元素，需要下标 + 多少 ，比如说在 `[2, 1]` 中，2 这个元素要在 1 这个维度上访问 5，需要 +1，而这个 1 就是 `stride[1]`。具体方式：

```c++
__device__ void to_index(int ordinal, const int *shape, int *out_index, int num_dims)
{
  /**
   * Convert an ordinal to an index in the shape. Should ensure that enumerating position 0 ... size of
   * a tensor produces every index exactly once. It may not be the inverse of index_to_position.
   * Args:
   *    ordinal: ordinal position to convert
   *    shape: tensor shape
   *    out_index: return index corresponding to position
   *    num_dims: number of dimensions in the tensor
   *
   * Returns:
   *    None (Fills in out_index)
   */
  int cur_ord = ordinal;
  for (int i = num_dims - 1; i >= 0; --i)
  {
    int sh = shape[i];
    out_index[i] = cur_ord % sh;
    cur_ord /= sh;
  }
}
```

以上是从 *一维存储数组* 的下标 映射到 *高维概念数组* 的坐标，反过来也是一样通过 stride 计算下标即可：

```c++
__device__ int index_to_position(const int *index, const int *strides, int num_dims)
{
  /**
   * Converts a multidimensional tensor index into a single-dimensional position in storage
   * based on strides.
   * Args:
   *    index: index tuple of ints
   *    strides: tensor strides
   *    num_dims: number of dimensions in the tensor, e.g. shape/strides of [2, 3, 4] has 3 dimensions
   *
   * Returns:
   *    int - position in storage
   */
  int position = 0;
  for (int i = 0; i < num_dims; ++i)
  {
    position += index[i] * strides[i];
  }
  return position;
}
```


**Broadcasting ：** 思路就是：我输出算作比较大的数组，为了计算某一个输出的元素，我需要到输入数组中找到对应位置来进行计算。但是由于输入数组的形状比较小，而你要找的那个元素在输入数组中的位置不存在，这时候你就需要去找这个维度上，需要最小的那个元素作为你的输入，这个流程就是广播。例子：

```
A = [[1, 2, 3],        # shape: (2, 3)
     [4, 5, 6]]

B = [10, 20, 30]       # shape: (3,)  → 广播成 (2, 3)

A + B = [[11, 22, 33],
         [14, 25, 36]]
```

比如说 `A+B` 的 `[1, 2]` 这个位置，对应到 `A` 和 `B` 中应该是 `[1, 2]` 这个位置，`A` 中有这个位置，`B` 中没有，所以 `B` 应该去找 `B[0, 2] = 20`, 从结果上看就是 `B` 沿着第 0 个维度复制了一下，实际上是通过访存下标变化来实现。具体代码：

```c++
__device__ void broadcast_index(const int *big_index, const int *big_shape, const int *shape, int *out_index, int num_dims_big, int num_dims)
{
  /**
   * Convert a big_index into big_shape to a smaller out_index into shape following broadcasting rules.
   * In this case it may be larger or with more dimensions than the shape given.
   * Additional dimensions may need to be mapped to 0 or removed.
   *
   * Args:
   *    big_index: multidimensional index of bigger tensor
   *    big_shape: tensor shape of bigger tensor
   *    shape: tensor shape of smaller tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    out_index: multidimensional index of smaller tensor
   *    nums_big_dims: number of dimensions in bigger tensor
   *    num_dims: number of dimensions in smaller tensor
   *
   * Returns:
   *    None (Fills in out_index)
   */
  for (int i = 0; i < num_dims; ++i)
  {
    if (shape[i] > 1)
    {
      out_index[i] = big_index[i + (num_dims_big - num_dims)];
    }
    else
    {
      out_index[i] = 0;
    }
  }
}
```


**例子 1：mapKernel**  

```c++
__global__ void mapKernel( float *out, int *out_shape, int *out_strides, int out_size, float *in_storage, int *in_shape, int *in_strides, int shape_size, int fn_id)
{
  /**
   * Map function. Apply a unary function to each element of the input array and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  in_storage: compact 1D array of size in_size
   *  in_shape: shape of the input array
   *  in_strides: strides of the input array
   *  shape_size: number of dimensions in the input and output arrays, assume dimensions are the same
   *  fn_id: id of the function to apply to each element of the input array
   *
   * Returns:
   *  None (Fills in out array)
   */

  int out_index[MAX_DIMS];
  int in_index[MAX_DIMS];
  
  int global_id = blockIdx.x * blockDim.x + threadIdx.x; //  计算这个 thread 的 global id，对应的就是 out 中的元素位置。注意 out 是实际存储的数组，是一维的

  // 排除多余的 threads
  if (global_id >= out_size) return;

  to_index(global_id, out_shape, out_index, shape_size);  // 将 1 dim 映射到 2dim
  broadcast_index(out_index, out_shape, in_shape, in_index, shape_size, shape_size); // 广播一下， in_index 就是我们要从输入数组中找的位置，是一个 二维的位置

  int in_pos = index_to_position(in_index, in_strides, shape_size); // 实际输入的 1 dim 中的位置
  int out_pos = index_to_position(out_index, out_strides, shape_size); // 实际输出的 1 dim 中的位置

  out[out_pos] = fn(fn_id, in_storage[in_pos]);
}

```


**例子 2：zipKernel**

```c++
__global__ void zipKernel( float *out, int *out_shape, int *out_strides, int out_size, int out_shape_size, float *a_storage, int *a_shape, int *a_strides, int a_shape_size, float *b_storage, int *b_shape, int *b_strides, int b_shape_size, int fn_id)
{
  /**
   * Zip function. Apply a binary function to elements of the input array a & b and store the result in the output array.
   * Optimization: Parallelize over the elements of the output array.
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   * - broadcast_index: converts an index in a smaller array to an index in a larger array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  out_shape_size: number of dimensions in the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  a_shape_size: number of dimensions in the input array
   *  b_storage: compact 1D array of size in_size
   *  b_shape: shape of the input array
   *  b_strides: strides of the input array
   *  b_shape_size: number of dimensions in the input array
   *  fn_id: id of the function to apply to each element of the a & b array
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

  int out_index[MAX_DIMS];
  int a_index[MAX_DIMS];
  int b_index[MAX_DIMS];

  int global_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_id >= out_size) return;

  to_index(global_id, out_shape, out_index, out_shape_size);
  broadcast_index(out_index, out_shape, a_shape, a_index, out_shape_size, a_shape_size); // 计算输入位置
  broadcast_index(out_index, out_shape, b_shape, b_index, out_shape_size, b_shape_size); // 计算输入位置

  int a_pos = index_to_position(a_index, a_strides, a_shape_size);  // 实际输入的 1 dim 中的位置
  int b_pos = index_to_position(b_index, b_strides, b_shape_size);  // 实际输入的 1 dim 中的位置
  int out_pos = index_to_position(out_index, out_strides, out_shape_size);  // 实际输出的 1 dim 中的位置

  out[out_pos] = fn(fn_id, a_storage[a_pos], b_storage[b_pos]);
}
```


**例子 3: reduceKernel**

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20260119160414429.png" width="400" caption="Reduce operation" align="center">}}


第一种方式是针对一个输出位置，循环原数组中的对应位置，得到最终结果。也就是说一个 thread 对应一个输出。比如说图中 15 这个位置，对应 1 5 9 三个位置。在kernel 中需要进行循环，不断累积结果，最终输出到 15 这个位置。

```c++
__global__ void reduceKernel( float *out, int *out_shape, int *out_strides, int out_size, float *a_storage, int *a_shape, int *a_strides, int reduce_dim, float reduce_value, int shape_size, int fn_id)
{
  /**
   * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
   * Optimization:
   * Parallelize over the reduction operation. Each kernel performs one reduction.
   * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  reduce_dim: dimension to reduce on
   *  reduce_value: initial value for the reduction
   *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
   *  fn_id: id of the reduce function, currently only support add, multiply, and max
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

  __shared__ double cache[BLOCK_DIM]; // Uncomment this line if you want to use shared memory to store partial results
  int out_index[MAX_DIMS];

  int global_id = blockDim.x * blockIdx.x + threadIdx.x;

  if (global_id >= out_size) {
    return;
  }

  to_index(global_id, out_shape, out_index, shape_size);
  int out_pos = index_to_position(out_index, out_strides, shape_size);

  int reduce_size = a_shape[reduce_dim];
  for (int i = 0; i < reduce_size; ++i) {
    out_index[reduce_dim] = i;
    int a_pos = index_to_position(out_index, a_strides, shape_size);
    reduce_value = fn(fn_id, reduce_value, a_storage[a_pos]);
  }
  out[out_pos] = reduce_value;
}
```


另一种算法是针对这个累求结果的过程（上述实现中的循环过程）进行优化，如下图：

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20260119160603306.png" width="400" caption="Parallel reduction" align="center">}}

每一段进行并行计算，最后再merge起来。这时候我们就不能一个 thread 处理一个输出位置了。这里的想法是：对于每一行的并行过程，我们使用的是一个 block 当中的不同thread 来并行完成。每个 step 选择一部分的 thread 完成操作，经过同步之后进行到下一个 step。最后第一个位置就是整个数组的累求和。数组存储就用 shared_memory 进行存储。这个 thread id 正好对应的是数组的下标，也算是一种设计。

> 累求和这个表述意义就是 a1 + a2 + ... +a_n 或者 a1 * a2 * ... * a_n 或者 ....，就是对所有元素的一种描述。想不出其他词了....

下边的实现其实是一个折中版本：考虑到 `blockDim` 的限制，如果要 reduce 的那一个维度巨长无比，甚至超出了  1024 （block 里边线程数量最大值），那我就先按照第一版的写法，手动计算，直到整个数组缩小到了 `blockDim` 以内。比如说：`blockDim = 8`，但是我有 16 个要 reduce 的元素，那我就先 `a[0] += a[16]; a[1] += a[17]`，最后只要计算 `a[0: 16]` 就可以。具体实现：

```c++
__global__ void reduceKernel( float *out, int *out_shape, int *out_strides, int out_size, float *a_storage, int *a_shape, int *a_strides, int reduce_dim, float reduce_value, int shape_size, int fn_id)
{
  /**
   * Reduce function. Apply a reduce function to elements of the input array a and store the result in the output array.
   * Optimization:
   * Parallelize over the reduction operation. Each kernel performs one reduction.
   * e.g. a = [[1, 2, 3], [4, 5, 6]], kernel0 computes reduce([1, 2, 3]), kernel1 computes reduce([4, 5, 6]).
   *
   * You may find the following functions useful:
   * - index_to_position: converts an index to a position in a compact array
   * - to_index: converts a position to an index in a multidimensional array
   *
   * Args:
   *  out: compact 1D array of size out_size to write the output to
   *  out_shape: shape of the output array
   *  out_strides: strides of the output array
   *  out_size: size of the output array
   *  a_storage: compact 1D array of size in_size
   *  a_shape: shape of the input array
   *  a_strides: strides of the input array
   *  reduce_dim: dimension to reduce on
   *  reduce_value: initial value for the reduction
   *  shape_size: number of dimensions in the input & output array, assert dimensions are the same
   *  fn_id: id of the reduce function, currently only support add, multiply, and max
   *
   *
   * Returns:
   *  None (Fills in out array)
   */

  __shared__ double cache[BLOCK_DIM]; 
  int out_index[MAX_DIMS];

  int tid = threadIdx.x;
  int out_id = blockIdx.x;
  if (out_id >= out_size) return;

  //out pos
  to_index(out_id, out_shape, out_index, shape_size);
  int out_pos = index_to_position(out_index, out_strides, shape_size);
  int reduce_size = a_shape[reduce_dim];

  // the number of threads in a block is restricted to 32. Therefore,
  // we have to add some elements before thread-level parallelization.
  // We are doing this because the tid will be exploited if a_shape[reduce_dim]
  // is greater than blockDim.
  float local_acc = reduce_value;
  for (int i = tid; i < reduce_size; i += blockDim.x) {
    out_index[reduce_dim] = i;
    int a_pos = index_to_position(out_index, a_strides, shape_size);
    local_acc = fn(fn_id, local_acc, a_storage[a_pos]);
  }

  cache[tid] = local_acc;
  __syncthreads();  // 等待所有元素在 shared memory 中加载好

  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      cache[tid] = fn(fn_id, cache[tid], cache[tid + s]);
    }
    __syncthreads();  // 等待这一层 (step) 的所有 thread 操作完毕
  }
  
  if (tid == 0) {
    out[out_pos] = cache[tid];  // 第一个元素就是最后的答案
  }
}

```


**例子 4: MatrixMultiplyKernel：**

这里我们跳过 *一个 thread 负责一个 out 位置* 的写法，直接实现 **Tiling** 的矩阵乘法。传统写法里边，输出矩阵 C 的一个 out 的位置对应的是输入矩阵 A 的一行，以及输入矩阵 B 的一列。那么传统的写法里边就是：一个 thread 负责一个 C 矩阵的位置，找到输入矩阵 A 和 B 对应的行和列，串行乘积累积和。但是每次 kernel 想要加载行列到 SM 的时候，其实是在访问 global_memory，并且由于每一行和列都要计算多次，实际上需要多次从 global_memory 中加载对应的行列，造成时间浪费。

那么应该怎么进行优化？Tiling 的想法是：首先我们让一个 block 负责一个 tile 的输出，让 threads 负责 tile 中的一个元素的计算。假设这个输出的 tile 的大小是 3 * 3，那么一共有 9 个 threads 参与了这个 tile 的运算。跟这个 tile 有关的所有 A 和 B 中的 tile 应该是对应行列的 tiles (见下图中的绿色与蓝色矩形)。而最终输出矩阵中的 tile 结果（紫色）就是如图下方所示的加和。我们沿着 K 维度遍历两个输入矩阵中的 tiles，让对应位置的 tile 做乘积和，结果加到最终的输出矩阵 C 对应的 tile 中即可。这个过程宏观上可以表述为（以下边的图片为例）：

1. 初始化 kernel 中的临时 tile_1, tile_2 (均在 shared_memory)
2. 将 A 中浅绿色 tile 加载到 tile_1 中
3. 将 B 中浅蓝色 tile 加载到 tile_2 中
4. 做乘积和，将对应位置结果加在 C 的紫色 tile 中
5. 将 A 中深绿色 tile 加载到 tile_1 中
6. 将 B 中深蓝色 tile 加载到 tile_2 中
7. 做乘积和，将对应位置结果加在 C 的紫色 tile 中 -> 得到最终结果

这样做首先不需要多次访问 global_memory，只需要一次加载好需要的两个 tile 到 shared_memory 中，然后访问的是 shared_memory 进行计算，节省时间。

{{< figure src="https://myblog-1316371247.cos.ap-shanghai.myqcloud.com/myblog/20260119210246116.png" width="400" caption="Tiling Matrix Multiplication" align="center">}}

*(图片来自 [Tutorial: OpenCL SGEMM tuning for Kepler](https://cnugteren.github.io/tutorial/pages/page1.html))*


在实现上，threads 在 kernel 里的行为分成两部分：第一部分是将数据从 global_memory 搬到 shared_memory；第二部分是针对 tile 中的某个位置进行计算。具体来说针对一个 threads，有以下步骤：

for `num_tiles`: 
1. 初始化 kernel 中的临时 tile_1, tile_2 (均在 shared_memory)
2. 本 threads 将 A 对应位置的 tile 中的某一个位置的数据从 global_memory 搬运到 tile_1
3. 本 threads 将 B 对应位置的 tile 中的某一个位置的数据从 global_memory 搬运到 tile_2
4. 等待所有 9 个 threads 把两个 tile 的数据全部搬运完 (因为计算的时候是要用到整行整列的，所以要同步)
5. for tile_length:
	1. 计算乘积和，将结果加到 C 矩阵中，本 threads 对应的输出位置上
6. 等待所有 9 个 threads 的计算完成 （避免不同轮之间的 tile shared_memory 读写混乱）

实际上，我们使用 `TILE_SIZE = 32`, 这样一个 tile 需要的 threads 数量就是一个 block 中含有的 threads 的最大值，最大化利用率。

具体实现：

```c++
__global__ void MatrixMultiplyKernel( float *out, const int *out_shape, const int *out_strides, float *a_storage, const int *a_shape, const int *a_strides, float *b_storage, const int *b_shape, const int *b_strides)
{
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix. Matrix a and b are both in a batch
   * format, with shape [batch_size, m, n], [batch_size, n, p].
   * Requirements:
   * - All data must be first moved to shared memory.
   * - Only read each cell in a and b once.
   * - Only write to global memory once per kernel.
   * There is guarantee that a_shape[0] == b_shape[0], a_shape[2] == b_shape[1],
   * and out_shape[0] == a_shape[0], out_shape[1] == b_shape[1]
   *
   * Args:
   *   out: compact 1D array of size batch_size x m x p to write the output to
   *   out_shape: shape of the output array
   *   out_strides: strides of the output array
   *   a_storage: compact 1D array of size batch_size x m x n
   *   a_shape: shape of the a array
   *   a_strides: strides of the a array
   *   b_storage: compact 1D array of size batch_size x n x p
   *   b_shape: shape of the b array
   *   b_strides: strides of the b array
   *
   * Returns:
   *   None (Fills in out array)
   */

  __shared__ float a_shared[TILE][TILE];  // TILE == 32
  __shared__ float b_shared[TILE][TILE];

  int batch = blockIdx.z;
  int a_batch_stride = a_shape[0] > 1 ? a_strides[0] : 0;
  int b_batch_stride = b_shape[0] > 1 ? b_strides[0] : 0;

  int m = a_shape[1];
  int n = a_shape[2];
  int p = b_shape[2];

  int row = blockIdx.x * TILE + threadIdx.x;
  int col = blockIdx.y * TILE + threadIdx.y;

  int a_index[3] = {batch, row, 0};
  int b_index[3] = {batch, 0, col};
  int a_pos;
  int b_pos;
  float acc = 0.0f;

  int a_strides_local[3] = {a_batch_stride, a_strides[1], a_strides[2]};
  int b_strides_local[3] = {b_batch_stride, b_strides[1], b_strides[2]};

  int num_tiles = (n + TILE - 1) / TILE;

  for (int i = 0; i < num_tiles; ++i) {
    // load elements for A and B
    int a_col = i * TILE + threadIdx.y;
    a_index[2] = a_col;
    if (row < m && a_col < n) {
      a_pos = index_to_position(a_index, a_strides_local, 3);
      a_shared[threadIdx.x][threadIdx.y] = a_storage[a_pos];
    } else {
      a_shared[threadIdx.x][threadIdx.y] = 0.0f;
    }

    int b_row = i * TILE + threadIdx.x;
    b_index[1] = b_row;
    if (b_row < n && col < p) {
      b_pos = index_to_position(b_index, b_strides_local, 3);
      b_shared[threadIdx.x][threadIdx.y] = b_storage[b_pos];
    } else {
      b_shared[threadIdx.x][threadIdx.y] = 0.0f;
    }

    __syncthreads();

    // Calculation for C
    for (int k = 0; k < TILE; ++k) {
      acc += a_shared[threadIdx.x][k] * b_shared[k][threadIdx.y];
    }
    __syncthreads();
  }

  if (row < m && col < p) {
    int out_index[3] = {batch, row, col};
    int out_pos = index_to_position(out_index, out_strides, 3);
    out[out_pos] = acc;
  }
}
```

这里有几个小点需要注意一下：

1. `num_tiles` 的计算：我们其实要计算的应该是 $\lceil K / TILE \rceil$ 由于 C++ 是下取整，我们可以进行等价变形： $\lceil K / TILE \rceil = \lfloor (K + TILE - 1) / TILE \rfloor$
2. 边界检查：在边界的 tile 往往会访问到数组之外，我们需要做边界检查，然后把边界之外的数字赋值成 `0.0f`。不用 if/else，而是把越界元素写成 0，是为了保持 warp 内所有 threads 行为一致，避免 divergence，同时保证数学正确性。



### Python 使用 Kernel

**nvcc compilation**

```bash
nvcc -o minitorch/cuda_kernels/combine.so --shared src/combine.cu -Xcompiler -fPIC
```

**参数说明**

| 参数 | 含义 |
|------|------|
| `nvcc` | NVIDIA CUDA 编译器 |
| `-o <path>` | 指定输出文件路径 |
| `--shared` | 编译成共享库（.so），而非可执行文件 |
| `src/combine.cu` | CUDA 源代码文件 |
| `-Xcompiler` | 将后续参数传递给 C++ 编译器 |
| `-fPIC` | Position Independent Code，生成位置无关代码（动态库必需） |

**编译流程**

```
.cu 源码 → nvcc 编译 → .so 共享库 → Python ctypes 加载 → 调用 CUDA 函数
```

**注意事项**

- `.so` = Shared Object，Linux 下的动态链接库
- 每次修改 `.cu` 文件后都需要重新编译
- `-fPIC` 是生成动态库的必要参数，否则无法被正确加载


**Python 调用 kernel 完成运算：** 先用 `nvcc` 编译好动态链接库 （`.so`），python 加载这个库，绑定签名，执行调用：

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("path/to/your.so")
datatype = np.float32

# 1) 绑定签名
lib.yourKernel.argtypes = [
    np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags="C_CONTIGUOUS"),  # out_storage
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # out_shape
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # out_strides
    ctypes.c_int,                                                         # out_size
    np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags="C_CONTIGUOUS"),  # in_storage
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # in_shape
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),  # in_strides
    ctypes.c_int,                                                         # in_size
    # ... other args
]
lib.yourKernel.restype = None

# 2) 调用（把底层 buffer/shape/strides 塞进去）
lib.yourKernel(
    out_storage,
    out_shape.astype(np.int32),
    out_strides.astype(np.int32),
    out_size,
    in_storage,
    in_shape.astype(np.int32),
    in_strides.astype(np.int32),
    in_size,
    # ... other args
)
```





### 一些拓展资料

- [cuda 13.0 以前的 guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#) 
- [cuda 13.0 以后的 guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [Tutorial: OpenCL SGEMM tuning for Kepler](https://cnugteren.github.io/tutorial/pages/page1.html)

> cuda 现在转向 cuTile 编程，好像类似于 `triton` 的想法，把 thread 底层的调用封装起来了
