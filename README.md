# CUDA Point Cloud Histogram Equalization

Parallel local histogram equalization on 3D point clouds (CUDA + OpenMP), comparing three neighborhood strategies for building the local histogram each point is equalized against.

Given N 3D points each with an intensity in `{0, ..., 255}`, every point's intensity is remapped based on the CDF of intensities in its local neighborhood. The neighborhood can be defined three ways:

| Method | Neighborhood definition | Complexity (seq → CUDA) |
|---|---|---|
| **Exact KNN** | k nearest neighbors by Euclidean distance | O(n² log k) → O(n log k)/thread |
| **Approximate KNN** | k-NN via 3D spatial grid hashing | O(n² log k) → O(C log k)/thread, C ≪ n |
| **K-Means** | cluster membership (k clusters) | O(T·n·k) → O(k)/thread |

## Key implementation details

- **Thread-local max-heap** (size k) for top-k maintenance — O(n log k) instead of O(nk) per thread.
- **Shared-memory tiling** in the exact-KNN kernel: blocks cooperatively stage 512 points into shared memory per iteration, cutting global memory reads by 512×.
- **Spatial grid hashing** for approximate KNN: points are bucketed into a uniform grid (~8 points/cell) and each thread searches only the 5×5×5 neighborhood of cells around its query — ~100× fewer distance computations than exact KNN, with `__ldg()` texture-cache reads for the sorted coordinate arrays.
- **Constant memory** for K-Means centroids, broadcast to all 32 threads in a warp for free.
- Integer (not `long long`) squared distances, early-exit CDF accumulation, and buffered `fwrite` output — each cuts measurable overhead at scale.

## Build & run

```bash
make                              # nvcc -arch=sm_35 -Xcompiler -fopenmp -O3
./a2 input.txt knn         # -> knn.txt
./a2 input.txt approx_knn  # -> approx_knn.txt
./a2 input.txt kmeans      # -> kmeans.txt
```

## Results

Measured on a Tesla K40m (2880 cores) against a sequential `g++ -O2` baseline:

| N | Exact KNN | Approx KNN | K-Means |
|---|---|---|---|
| 1,000 | 45× | 46× | 8× |
| 10,000 | 70× | 141× | 23× |
| 100,000 | 265× | **1528×** | 424× |

All three methods are numerically correct (MAE = 0 vs. sequential reference for exact KNN and K-Means; MAE = 0.0014 for approximate KNN, well under the assignment's error threshold of 3). Approximate KNN scales best because its per-thread cost stays roughly constant (~1000 candidates) while the sequential baseline it's compared against grows as O(n²).

See `report.pdf` for the full writeup, including CUDA kernel listings and per-optimization analysis.
