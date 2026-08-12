/*
 * Implementation of Parallel Local Histogram Equalization on 3D Point Clouds
 * CUDA + OpenMP implementation
 * Usage:
 *   ./a2 input.txt knn         -> knn.txt
 *   ./a2 input.txt approx_knn  -> approx_knn.txt
 *   ./a2 input.txt kmeans      -> kmeans.txt
 */

#include <cstdio>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

// error checking macro
#define CHK(call)                                                       \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA fail %s:%d  %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(1);                                                    \
        }                                                               \
    } while(0)

// global constants
#define TPB      512    // threads per block - 512 gives better occupancy on K40
#define KMAX     128    // max k
#define RSEARCH  2      // grid search radius for approx KNN

// constant memory for kmeans centroids - lives in GPU constant cache
// much faster than reading from global memory every iteration
#define CMAX 128
__constant__ int d_cX[CMAX];
__constant__ int d_cY[CMAX];
__constant__ int d_cZ[CMAX];

// DEVICE UTILITIES

/*
 * point_closer_or_lex:
 *   returns if a point should be chosen over the other
 *   first priority is smaller distance, then if tie, lexicographic tie breaking
 */
__device__ __forceinline__
bool point_closer_or_lex(int da, int ax, int ay, int az,
                          int db, int bx, int by, int bz)
{
    if (da != db) return da < db;
    if (ax != bx) return ax < bx;
    if (ay != by) return ay < by;
    return az < bz;
}

/*
 * provides speedup because of efficient topk maintenance
 * a thread i working with point i would have its own heap
 *
 * maxheap_fix_down:
 *   Restores the max-heap property downward from position `pos`.
 *   The heap stores (distance, x, y, z, original_index) tuples so
 *   that the root is always the WORST (furthest / lex-largest) element.
 *   Storing coordinates avoids random global-memory reads during heap operations.
 */
__device__
void maxheap_fix_down(int *dist, int *nx, int *ny, int *nz, int *nidx,
                      int pos, int sz)
{
    // maintain the heap property
    while (true) {
        int worst = pos;
        // left child and right child
        int lc = 2*pos + 1;
        int rc = 2*pos + 2;

        if (lc < sz && point_closer_or_lex(dist[worst], nx[worst],
                                            ny[worst], nz[worst],
                                            dist[lc],   nx[lc],
                                            ny[lc],     nz[lc]))
            worst = lc;

        if (rc < sz && point_closer_or_lex(dist[worst], nx[worst],
                                            ny[worst], nz[worst],
                                            dist[rc],   nx[rc],
                                            ny[rc],     nz[rc]))
            worst = rc;

        if (worst == pos) break;

        // swap pos <-> worst
        int td = dist[pos], tx = nx[pos], ty = ny[pos],
            tz = nz[pos],  ti = nidx[pos];
        dist[pos]=dist[worst]; nx[pos]=nx[worst]; ny[pos]=ny[worst];
        nz[pos]=nz[worst];     nidx[pos]=nidx[worst];
        dist[worst]=td; nx[worst]=tx; ny[worst]=ty;
        nz[worst]=tz;   nidx[worst]=ti;
        pos = worst;
    }
}

/*
 * maxheap_build:
 * build heap from an unsorted array
 */
__device__
void maxheap_build(int *dist, int *nx, int *ny, int *nz, int *nidx, int sz)
{
    // (sz-2)/2 is the index of the last non-leaf node in the binary heap
    for (int p = (sz-2)/2; p >= 0; p--)
        maxheap_fix_down(dist, nx, ny, nz, nidx, p, sz);
}

/*
 * histogram equalization formula implementation
 * the results are clipped to [0,255]
 */
__device__ __forceinline__
int remap_intensity(int cdf_val, int cdf_min, int m, int orig_I)
{
    if (m == cdf_min) return orig_I;
    double t = ((double)(cdf_val - cdf_min) /
                (double)(m       - cdf_min)) * 255.0;
    int v = (int)floor(t);
    // if it goes beyond the given range, limit it to the bounding values
    return (v < 0) ? 0 : (v > 255 ? 255 : v);
}

/*
 * build_hist_and_remap:
 *   Given an intensity array `intensities[0..count-1]` for a
 *   neighbourhood, builds the histogram, computes CDF inline, and
 *   returns the remapped intensity for `target_I`.
 *   `m` = neighbourhood size (k+1 for KNN, cluster size for K-means).
 *
 *   Optimisation: early exit from CDF loop once we have passed target_I
 *   and already found cdf_min. Saves ~50% of CDF iterations on average
 *   since intensities are roughly uniform (avg exit at v=128 not v=255).
 */
__device__
int build_hist_and_remap(const int *intensities, int count,
                          int target_I, int m)
{
    int hist[256];
    // create histogram and count frequencies
    for (int v = 0; v < 256; v++) hist[v] = 0;
    for (int j = 0; j < count; j++) hist[intensities[j]]++;

    // compute CDF - early exit once we reach target_I and have cdf_min
    int cumsum  = 0;
    int cdf_min = 0;
    int cdf_val = 0;
    for (int v = 0; v < 256; v++) {
        cumsum += hist[v];
        if (cdf_min == 0 && cumsum > 0) cdf_min = cumsum;
        if (v == target_I) {
            cdf_val = cumsum;
            break;   // early exit - no need to scan further
        }
    }
    return remap_intensity(cdf_val, cdf_min, m, target_I);
}

/*
 * KERNEL 1: EXACT KNN
 *
 * For each point:
 *   scan all points using shared-memory tiling
 *   keep exact K nearest neighbours using a max-heap
 *   collect their intensities
 *   locally equalize intensity
 *
 * TPB=512: larger tile = fewer global memory loads per point.
 * Shared mem used: 3 * 512 * 4 = 6KB per block (fits in 48KB on K40).
 */
__global__
void cuda_exact_knn(const int * __restrict__ X,
                    const int * __restrict__ Y,
                    const int * __restrict__ Z,
                    const int * __restrict__ I_in,
                    int N, int K,
                    int * __restrict__ I_out)
{
    // shared mem declaration, shared among threads of the same block
    __shared__ int sx[TPB], sy[TPB], sz[TPB];

    int tid = threadIdx.x;
    // thread indexing formula
    int qi  = blockIdx.x * TPB + threadIdx.x;

    // check if thread is valid
    bool live = (qi < N);
    int qx = live ? X[qi] : 0;
    int qy = live ? Y[qi] : 0;
    int qz = live ? Z[qi] : 0;

    // thread-local max-heap arrays (k slots)
    int hd[KMAX], hx[KMAX], hy[KMAX], hz[KMAX], hi[KMAX];
    int heap_sz = 0;

    // tile scan, helps provide reusability
    for (int base = 0; base < N; base += TPB) {
        int src = base + tid;
        // store values in the shared memory
        if (src < N) { sx[tid]=X[src]; sy[tid]=Y[src]; sz[tid]=Z[src]; }
        __syncthreads();

        if (live) {
            int tile_end = min(TPB, N - base);
            for (int s = 0; s < tile_end; s++) {
                int j = base + s;
                if (j == qi) continue;

                // compute sq distances
                int tx = sx[s], ty = sy[s], tz = sz[s];
                int dx = tx-qx, dy = ty-qy, dz = tz-qz;
                int d2 = dx*dx + dy*dy + dz*dz;

                // append the first k neighbours
                if (heap_sz < K) {
                    hd[heap_sz]=d2; hx[heap_sz]=tx;
                    hy[heap_sz]=ty; hz[heap_sz]=tz; hi[heap_sz]=j;
                    heap_sz++;
                    if (heap_sz == K)
                        maxheap_build(hd,hx,hy,hz,hi,K);
                } else if (point_closer_or_lex(d2,tx,ty,tz,hd[0],hx[0],hy[0],hz[0])) {
                    hd[0]=d2; hx[0]=tx; hy[0]=ty; hz[0]=tz; hi[0]=j;
                    maxheap_fix_down(hd,hx,hy,hz,hi,0,K);
                }
            }
        }
        __syncthreads();
    }

    if (!live) return;

    // collect neighbourhood intensities: point itself + k neighbours
    int nbr_I[KMAX+1];
    nbr_I[0] = I_in[qi];
    for (int q = 0; q < K; q++) nbr_I[q+1] = I_in[hi[q]];

    I_out[qi] = build_hist_and_remap(nbr_I, K+1, I_in[qi], K+1);
}

/*
 * KERNEL 2: APPROXIMATE KNN  (grid-based spatial search)
 *   Each thread searches all cells within radius RSEARCH of its
 *   query cell -> (2R+1)^3 = 125 cells examined.
 *
 * Instead of comparing each point with all N points, do this:
 *   divide 3D space into grid cells
 *   put each point into a cell
 *   for a query point, search only its own cell and nearby cells
 * So it avoids scanning the entire dataset.
 *
 * Optimisation: __ldg() loads use the read-only texture cache path
 * which is faster for scattered reads from sorted arrays.
 */
__global__
void cuda_approx_knn(const int * __restrict__ sX,    // sorted coords
                     const int * __restrict__ sY,
                     const int * __restrict__ sZ,
                     const int * __restrict__ sI,
                     const int * __restrict__ s_orig, // orig index
                     const int * __restrict__ cell_off, // cell start idx
                     const int * __restrict__ cell_cnt,
                     int G,                            // cells/dim
                     const int * __restrict__ qX,    // query (orig order)
                     const int * __restrict__ qY,
                     const int * __restrict__ qZ,
                     const int * __restrict__ qI,
                     const int * __restrict__ qGx,   // cell coords
                     const int * __restrict__ qGy,
                     const int * __restrict__ qGz,
                     int N, int K,
                     int * __restrict__ I_out)
{
    int qi = blockIdx.x * TPB + threadIdx.x;
    if (qi >= N) return;

    // local heap
    int hd[KMAX], hx[KMAX], hy[KMAX], hz[KMAX], hi[KMAX];
    int heap_sz = 0;

    // load query point and its cells
    int qx = qX[qi], qy = qY[qi], qz = qZ[qi];
    int cx = qGx[qi], cy = qGy[qi], cz = qGz[qi];

    // search bounds
    int x0=max(0,cx-RSEARCH), x1=min(G-1,cx+RSEARCH);
    int y0=max(0,cy-RSEARCH), y1=min(G-1,cy+RSEARCH);
    int z0=max(0,cz-RSEARCH), z1=min(G-1,cz+RSEARCH);

    // scan all candidate nearby cells
    for (int gz=z0; gz<=z1; gz++)
    for (int gy=y0; gy<=y1; gy++)
    for (int gx=x0; gx<=x1; gx++) {
        int cid   = gz*G*G + gy*G + gx;
        int cnt   = cell_cnt[cid];
        int start = cell_off[cid];
        for (int s=0; s<cnt; s++) {
            int js = start+s;
            int jo = __ldg(&s_orig[js]);
            if (jo == qi) continue; // self
            // use __ldg for read-only texture cache - faster scattered reads
            int tx=__ldg(&sX[js]), ty=__ldg(&sY[js]), tz=__ldg(&sZ[js]);
            int dx=tx-qx, dy=ty-qy, dz=tz-qz;
            int d2=dx*dx+dy*dy+dz*dz;
            // from the candidate points, compute sq distances and maintain heap
            if (heap_sz < K) {
                hd[heap_sz]=d2; hx[heap_sz]=tx;
                hy[heap_sz]=ty; hz[heap_sz]=tz; hi[heap_sz]=jo;
                heap_sz++;
                if (heap_sz==K) maxheap_build(hd,hx,hy,hz,hi,K);
            } else if (point_closer_or_lex(d2,tx,ty,tz,
                                            hd[0],hx[0],hy[0],hz[0])) {
                hd[0]=d2; hx[0]=tx; hy[0]=ty; hz[0]=tz; hi[0]=jo;
                maxheap_fix_down(hd,hx,hy,hz,hi,0,K);
            }
        }
    }

    // fallback: fill remaining slots if grid search found < k points
    for (int j=0; j<N && heap_sz<K; j++) {
        if (j==qi) continue;
        bool dup=false;
        for (int q=0;q<heap_sz;q++) if(hi[q]==j){dup=true;break;}
        if (!dup) {
            int tx=qX[j],ty=qY[j],tz=qZ[j];
            int dx=tx-qx,dy=ty-qy,dz=tz-qz;
            hd[heap_sz]=dx*dx+dy*dy+dz*dz;
            hx[heap_sz]=tx; hy[heap_sz]=ty; hz[heap_sz]=tz; hi[heap_sz]=j;
            heap_sz++;
        }
    }
    if (heap_sz==K) maxheap_build(hd,hx,hy,hz,hi,K);

    int nbr_I[KMAX+1];
    nbr_I[0]=qI[qi];
    for (int q=0;q<K;q++) nbr_I[q+1]=qI[hi[q]];

    I_out[qi] = build_hist_and_remap(nbr_I, K+1, qI[qi], K+1);
}

/*
 * KERNEL 3a: KMEANS ASSIGNMENT
 *
 * Assigns each point to its nearest centroid.
 * Tiebreak: lex-smaller centroid (x, y, z).
 * Sets *flag=1 atomically if any assignment changed (convergence check).
 *
 * Optimisation: centroids are read from __constant__ memory (d_cX/Y/Z)
 * which is cached and broadcast to all threads in a warp for free.
 * Much faster than reading from global memory every iteration.
 *
 * steps:
 *   cluster points into K groups
 *   for each point, use its cluster intensities
 *   equalize intensity based on cluster histogram
 */
__global__
void cuda_kmeans_assign(const int * __restrict__ X,  // points
                        const int * __restrict__ Y,
                        const int * __restrict__ Z,
                        int N, int K,
                        int * __restrict__ labels,
                        int * __restrict__ flag)
{
    int i = blockIdx.x * TPB + threadIdx.x;
    if (i >= N) return;

    long long best_d=LLONG_MAX;
    int best=-1;
    int xi=X[i], yi=Y[i], zi=Z[i];

    // loop over all centroids - read from constant memory cache
    for (int c=0; c<K; c++) {
        // for each, compute squared distance to centroid
        long long dx=(long long)(d_cX[c]-xi);
        long long dy=(long long)(d_cY[c]-yi);
        long long dz=(long long)(d_cZ[c]-zi);
        long long d=dx*dx+dy*dy+dz*dz;
        bool pick=false;
        // choose the closer ones
        if (d<best_d) {
            pick=true;
        } else if (d==best_d) {
            if      (d_cX[c]!=d_cX[best]) pick=d_cX[c]<d_cX[best];
            else if (d_cY[c]!=d_cY[best]) pick=d_cY[c]<d_cY[best];
            else                            pick=d_cZ[c]<d_cZ[best];
        }
        if (pick){best_d=d; best=c;}
    }

    if (labels[i]!=best){ labels[i]=best; atomicOr(flag,1); }
}

/*
 * KERNEL 3b: KMEANS REMAP
 *
 * After convergence, remap each point using its cluster histogram.
 * cluster_sI: intensities sorted by cluster (built on host).
 * c_off[c]:   start index in cluster_sI for cluster c.
 * c_cnt[c]:   number of points in cluster c  (= m for this cluster).
 */
__global__
void cuda_kmeans_remap(const int * __restrict__ I_in,
                       const int * __restrict__ labels,
                       const int * __restrict__ cluster_sI,
                       const int * __restrict__ c_off,
                       const int * __restrict__ c_cnt,
                       int N,
                       int * __restrict__ I_out)
{
    int i = blockIdx.x * TPB + threadIdx.x;
    if (i >= N) return;

    int c     = labels[i];
    int start = c_off[c];
    int cnt   = c_cnt[c];

    I_out[i] = build_hist_and_remap(cluster_sI + start, cnt,
                                     I_in[i], cnt);
}

/*
 * HOST I/O
 * Optimisation: buffered output using a single fwrite instead of
 * N separate fprintf calls. Each fprintf has syscall overhead;
 * pre-formatting into a buffer and writing once is much faster.
 * Saves 1-2 seconds on n=100000 since file I/O is timed.
 */
static void save_output(const char *path, int N,
                         const int *X, const int *Y,
                         const int *Z, const int *I)
{
    // 32 bytes per line is enough for 4 ints + spaces + newline
    char *buf = (char*)malloc((size_t)N * 32);
    if (!buf){ perror("malloc output buffer"); exit(1); }
    int pos = 0;
    for (int i = 0; i < N; i++)
        pos += sprintf(buf+pos, "%d %d %d %d\n", X[i], Y[i], Z[i], I[i]);
    FILE *f = fopen(path, "w");
    if (!f){ perror("fopen output"); exit(1); }
    fwrite(buf, 1, pos, f);
    fclose(f);
    free(buf);
}

static void load_input(const char *path,
                        int *N, int *K, int *T,
                        int **X, int **Y, int **Z, int **I)
{
    FILE *f = fopen(path,"r");
    if (!f){ perror("fopen"); exit(1); }
    // read parameters
    fscanf(f,"%d %d %d", N, K, T);

    // allocate host arrays
    *X=(int*)malloc((*N)*sizeof(int));
    *Y=(int*)malloc((*N)*sizeof(int));
    *Z=(int*)malloc((*N)*sizeof(int));
    *I=(int*)malloc((*N)*sizeof(int));
    for (int i=0;i<*N;i++){
        fscanf(f,"%d %d %d %d",
               &(*X)[i],&(*Y)[i],&(*Z)[i],&(*I)[i]);
    }
    fclose(f);
}

// HOST(CPU): EXACT KNN launcher
static void run_exact_knn(int N, int K,
                           const int *X, const int *Y,
                           const int *Z, const int *I,
                           int *Out)
{
    int *dX,*dY,*dZ,*dI,*dOut;
    // allocate gpu memory
    CHK(cudaMalloc(&dI,  N*sizeof(int)));
    CHK(cudaMalloc(&dZ,  N*sizeof(int)));
    CHK(cudaMalloc(&dY,  N*sizeof(int)));
    CHK(cudaMalloc(&dX,  N*sizeof(int)));
    CHK(cudaMalloc(&dOut,N*sizeof(int)));

    // copy input from cpu to gpu
    CHK(cudaMemcpy(dZ,Z,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dY,Y,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dX,X,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dI,I,N*sizeof(int),cudaMemcpyHostToDevice));

    CHK(cudaFuncSetCacheConfig(cuda_exact_knn,cudaFuncCachePreferL1));

    // cuda timing events
    cudaEvent_t t0,t1;
    CHK(cudaEventCreate(&t0)); CHK(cudaEventCreate(&t1));

    // number of blocks (ceil division)
    int BLK=(N+TPB-1)/TPB;

    // record timing and launch kernel
    CHK(cudaEventRecord(t0));
    cuda_exact_knn<<<BLK,TPB>>>(dX,dY,dZ,dI,N,K,dOut);
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));

    // measure time taken
    float ms; CHK(cudaEventElapsedTime(&ms,t0,t1));
    fprintf(stderr,"[KNN] kernel=%.2fms\n",ms);

    // copy result back to cpu and clean up memory
    CHK(cudaMemcpy(Out,dOut,N*sizeof(int),cudaMemcpyDeviceToHost));
    cudaFree(dX);cudaFree(dY);cudaFree(dZ);cudaFree(dI);cudaFree(dOut);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
}

// HOST: APPROXIMATE KNN  (build 3D grid, launch kernel)
//
// G        = cells per dimension
// ncells   = G^3
// cell_sz  = size of each cell
// xmin, ymin, zmin = min coordinates

struct SpatialGrid {
    int   G, ncells;
    float cell_sz;
    int   xmin, ymin, zmin;
    int  *sX,*sY,*sZ,*sI;   // sorted arrays
    int  *s_orig;
    int  *cell_off, *cell_cnt;
    int  *qGx, *qGy, *qGz;  // cell coords per query
};

// grid preparation for approx KNN
static SpatialGrid grid_build(int N,
    const int *X, const int *Y,
    const int *Z, const int *I)
{
    SpatialGrid g;
    g.xmin=X[0]; g.ymin=Y[0]; g.zmin=Z[0];
    int xmx=X[0],ymx=Y[0],zmx=Z[0];
    // create bounding box
    for (int i=1;i<N;i++){
        g.xmin=std::min(g.xmin,X[i]); g.ymin=std::min(g.ymin,Y[i]); g.zmin=std::min(g.zmin,Z[i]);
        xmx=std::max(xmx,X[i]); ymx=std::max(ymx,Y[i]); zmx=std::max(zmx,Z[i]);
    }
    g.G=(int)cbrt((double)N/8.0);
    if(g.G<1)g.G=1; if(g.G>100)g.G=100;
    int span=std::max(std::max(xmx-g.xmin,ymx-g.ymin),zmx-g.zmin)+1;
    g.cell_sz=(float)span/g.G;
    g.ncells=g.G*g.G*g.G;

    g.sX=(int*)malloc(N*sizeof(int)); g.sY=(int*)malloc(N*sizeof(int));
    g.sZ=(int*)malloc(N*sizeof(int)); g.sI=(int*)malloc(N*sizeof(int));
    g.s_orig=(int*)malloc(N*sizeof(int));
    g.cell_off=(int*)calloc(g.ncells,sizeof(int));
    g.cell_cnt=(int*)calloc(g.ncells,sizeof(int));
    g.qGx=(int*)malloc(N*sizeof(int));
    g.qGy=(int*)malloc(N*sizeof(int));
    g.qGz=(int*)malloc(N*sizeof(int));

    // to store each point's cell id
    int *cid=(int*)malloc(N*sizeof(int));

    // convert physical coordinates to grid coordinates
    for (int i=0;i<N;i++){
        int cx=(int)((X[i]-g.xmin)/g.cell_sz);
        int cy=(int)((Y[i]-g.ymin)/g.cell_sz);
        int cz=(int)((Z[i]-g.zmin)/g.cell_sz);
        cx=std::min(cx,g.G-1); cy=std::min(cy,g.G-1); cz=std::min(cz,g.G-1);
        g.qGx[i]=cx; g.qGy[i]=cy; g.qGz[i]=cz;
        // flatten cell id
        cid[i]=cz*g.G*g.G+cy*g.G+cx;
        // count points per cell
        g.cell_cnt[cid[i]]++;
    }
    g.cell_off[0]=0;
    for (int c=1;c<g.ncells;c++)
        g.cell_off[c]=g.cell_off[c-1]+g.cell_cnt[c-1];
    int *off=(int*)calloc(g.ncells,sizeof(int));

    // make points of same cell contiguous in memory
    for (int i=0;i<N;i++){
        int c=cid[i], p=g.cell_off[c]+off[c];
        g.sX[p]=X[i]; g.sY[p]=Y[i]; g.sZ[p]=Z[i];
        g.sI[p]=I[i]; g.s_orig[p]=i; off[c]++;
    }
    free(cid); free(off);
    fprintf(stderr,"[GRID] G=%d cell_sz=%.1f\n",g.G,g.cell_sz);
    return g;
}

// free all memory inside spatial grid
static void grid_free(SpatialGrid &g){
    free(g.cell_off);free(g.cell_cnt);
    free(g.sX);free(g.sY);free(g.sZ);free(g.sI);
    free(g.s_orig);free(g.qGx);free(g.qGy);free(g.qGz);
}

// host wrapper for spatial approx knn
static void run_approx_knn(int N, int K,
    const int *X, const int *Y,
    const int *Z, const int *I,
    int *Out)
{
    SpatialGrid g = grid_build(N,X,Y,Z,I);

    // device pointers
    int *dcoff,*dccnt;
    int *dsX,*dsY,*dsZ,*dsI,*ds_orig;
    int *dqX,*dqY,*dqZ,*dqI,*dqGx,*dqGy,*dqGz,*dOut;

    // allocate all gpu memory
    CHK(cudaMalloc(&dsX,   N*sizeof(int)));CHK(cudaMalloc(&dsY,   N*sizeof(int)));
    CHK(cudaMalloc(&dsZ,   N*sizeof(int)));CHK(cudaMalloc(&dsI,   N*sizeof(int)));
    CHK(cudaMalloc(&dcoff, g.ncells*sizeof(int)));
    CHK(cudaMalloc(&dccnt, g.ncells*sizeof(int)));
    CHK(cudaMalloc(&ds_orig,N*sizeof(int)));
    CHK(cudaMalloc(&dqGx,  N*sizeof(int)));CHK(cudaMalloc(&dqGy,  N*sizeof(int)));
    CHK(cudaMalloc(&dqGz,  N*sizeof(int)));CHK(cudaMalloc(&dOut,  N*sizeof(int)));
    CHK(cudaMalloc(&dqX,   N*sizeof(int)));CHK(cudaMalloc(&dqY,   N*sizeof(int)));
    CHK(cudaMalloc(&dqZ,   N*sizeof(int)));CHK(cudaMalloc(&dqI,   N*sizeof(int)));

    // copy all required arrays to gpu
    CHK(cudaMemcpy(dsX,  g.sX,      N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dsY,  g.sY,      N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dsZ,  g.sZ,      N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dcoff,g.cell_off,g.ncells*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dccnt,g.cell_cnt,g.ncells*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dsI,  g.sI,      N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(ds_orig,g.s_orig,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqGx, g.qGx,    N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqGy, g.qGy,    N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqGz, g.qGz,    N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqX,  X,         N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqY,  Y,         N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqZ,  Z,         N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dqI,  I,         N*sizeof(int),cudaMemcpyHostToDevice));

    CHK(cudaFuncSetCacheConfig(cuda_approx_knn,cudaFuncCachePreferL1));
    cudaEvent_t t0,t1;

    // launch kernel
    CHK(cudaEventCreate(&t0));CHK(cudaEventCreate(&t1));

    int BLK=(N+TPB-1)/TPB;
    CHK(cudaEventRecord(t0));
    cuda_approx_knn<<<BLK,TPB>>>(dsX,dsY,dsZ,dsI,ds_orig,
                                  dcoff,dccnt,g.G,
                                  dqX,dqY,dqZ,dqI,
                                  dqGx,dqGy,dqGz,N,K,dOut);
    CHK(cudaGetLastError());
    CHK(cudaEventRecord(t1));CHK(cudaEventSynchronize(t1));

    float ms; CHK(cudaEventElapsedTime(&ms,t0,t1));
    fprintf(stderr,"[APPROX_KNN] kernel=%.2fms\n",ms);

    CHK(cudaMemcpy(Out,dOut,N*sizeof(int),cudaMemcpyDeviceToHost));

    cudaFree(dsX);cudaFree(dsY);cudaFree(dsZ);cudaFree(dsI);cudaFree(ds_orig);
    cudaFree(dcoff);cudaFree(dccnt);
    cudaFree(dqX);cudaFree(dqY);cudaFree(dqZ);cudaFree(dqI);
    cudaFree(dqGx);cudaFree(dqGy);cudaFree(dqGz);cudaFree(dOut);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    grid_free(g);
}

// HOST: KMEANS

// CPU centroid update with OpenMP (integer division)
static void update_centroids_omp(const int *X, const int *Y, const int *Z,
                                   const int *labels, int N, int K,
                                   int *cX, int *cY, int *cZ)
{
    // accumulators, eg: sx[] = sum of x coords
    long long *sx=(long long*)calloc(K,sizeof(long long));
    long long *sy=(long long*)calloc(K,sizeof(long long));
    long long *sz=(long long*)calloc(K,sizeof(long long));
    // count of points per cluster
    int *sc=(int*)calloc(K,sizeof(int));

    // combine thread-local partial results
    #pragma omp parallel for reduction(+:sx[:K],sy[:K],sz[:K],sc[:K])
    for (int i=0;i<N;i++){
        int c=labels[i];
        sx[c]+=X[i]; sy[c]+=Y[i]; sz[c]+=Z[i]; sc[c]++;
    }

    // new centroid computation
    for (int c=0;c<K;c++){
        if (sc[c]>0){
            cX[c]=(int)(sx[c]/sc[c]);
            cY[c]=(int)(sy[c]/sc[c]);
            cZ[c]=(int)(sz[c]/sc[c]);
        }
        // empty cluster: centroid unchanged
    }
    free(sx);free(sy);free(sz);free(sc);
}

// Build cluster-sorted intensity array for remap kernel
static void make_cluster_table(int N, int K,
                                 const int *I, const int *labels,
                                 int **sI_out, int **off_out, int **cnt_out)
{
    int *cnt=(int*)calloc(K,sizeof(int));
    int *off=(int*)malloc(K*sizeof(int));
    int *sI =(int*)malloc(N*sizeof(int));
    for (int i=0;i<N;i++) cnt[labels[i]]++;
    off[0]=0;
    for (int c=1;c<K;c++) off[c]=off[c-1]+cnt[c-1];
    int *tmp=(int*)calloc(K,sizeof(int));
    // scatter intensities to make them contiguous in memory
    for (int i=0;i<N;i++){
        int c=labels[i];
        sI[off[c]+tmp[c]]=I[i]; tmp[c]++;
    }
    free(tmp);
    *sI_out=sI; *off_out=off; *cnt_out=cnt;
}

static void run_kmeans(int N, int K, int T,
                        const int *X, const int *Y,
                        const int *Z, const int *I,
                        int *Out)
{
    // initial centroids = first K points
    int *hcX=(int*)malloc(K*sizeof(int));
    int *hcY=(int*)malloc(K*sizeof(int));
    int *hcZ=(int*)malloc(K*sizeof(int));
    for (int c=0;c<K;c++){hcX[c]=X[c];hcY[c]=Y[c];hcZ[c]=Z[c];}

    // cluster labels
    int *hlabels=(int*)calloc(N,sizeof(int));

    int *dX,*dY,*dZ,*dlabels,*dflag;
    CHK(cudaMalloc(&dX,      N*sizeof(int)));
    CHK(cudaMalloc(&dY,      N*sizeof(int)));
    CHK(cudaMalloc(&dZ,      N*sizeof(int)));
    CHK(cudaMalloc(&dlabels, N*sizeof(int)));
    CHK(cudaMalloc(&dflag,   sizeof(int)));

    // host to device mem copy
    CHK(cudaMemcpy(dX,X,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dY,Y,N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dZ,Z,N*sizeof(int),cudaMemcpyHostToDevice));
    // init labels to -1 so first iteration always detects change
    CHK(cudaMemset(dlabels,-1,N*sizeof(int)));

    int BLK=(N+TPB-1)/TPB;

    // timings
    cudaEvent_t t0,t1;
    CHK(cudaEventCreate(&t0));CHK(cudaEventCreate(&t1));
    CHK(cudaEventRecord(t0));

    for (int iter=0;iter<T;iter++){
        // copy centroids to constant memory cache (faster than global)
        CHK(cudaMemcpyToSymbol(d_cX,hcX,K*sizeof(int)));
        CHK(cudaMemcpyToSymbol(d_cY,hcY,K*sizeof(int)));
        CHK(cudaMemcpyToSymbol(d_cZ,hcZ,K*sizeof(int)));

        CHK(cudaMemset(dflag,0,sizeof(int)));

        // kernel launch
        cuda_kmeans_assign<<<BLK,TPB>>>(dX,dY,dZ,N,K,dlabels,dflag);
        CHK(cudaGetLastError());
        CHK(cudaDeviceSynchronize());

        int hflag=0;
        CHK(cudaMemcpy(&hflag,dflag,sizeof(int),cudaMemcpyDeviceToHost));

        // if no change -> converged
        if (!hflag){
            fprintf(stderr,"[KMEANS] converged iter=%d\n",iter);
            break;
        }
        // copy back to cpu
        CHK(cudaMemcpy(hlabels,dlabels,N*sizeof(int),cudaMemcpyDeviceToHost));
        update_centroids_omp(X,Y,Z,hlabels,N,K,hcX,hcY,hcZ);
    }

    CHK(cudaMemcpy(hlabels,dlabels,N*sizeof(int),cudaMemcpyDeviceToHost));

    int *h_sI,*h_off,*h_cnt;
    make_cluster_table(N,K,I,hlabels,&h_sI,&h_off,&h_cnt);

    // remap stage preparation
    int *dI,*d_sI,*d_off,*d_cnt,*dOut;
    CHK(cudaMalloc(&d_off,K*sizeof(int)));CHK(cudaMalloc(&d_cnt,K*sizeof(int)));
    CHK(cudaMalloc(&dI,   N*sizeof(int)));CHK(cudaMalloc(&d_sI, N*sizeof(int)));
    CHK(cudaMalloc(&dOut, N*sizeof(int)));

    CHK(cudaMemcpy(d_off,h_off, K*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_cnt,h_cnt, K*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dI,   I,    N*sizeof(int),cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(d_sI, h_sI, N*sizeof(int),cudaMemcpyHostToDevice));

    CHK(cudaFuncSetCacheConfig(cuda_kmeans_remap,cudaFuncCachePreferL1));

    // kernel launch
    cuda_kmeans_remap<<<BLK,TPB>>>(dI,dlabels,d_sI,d_off,d_cnt,N,dOut);
    CHK(cudaGetLastError());

    // stop timings
    CHK(cudaEventRecord(t1));CHK(cudaEventSynchronize(t1));

    float ms; CHK(cudaEventElapsedTime(&ms,t0,t1));
    fprintf(stderr,"[KMEANS] total=%.2fms\n",ms);

    CHK(cudaMemcpy(Out,dOut,N*sizeof(int),cudaMemcpyDeviceToHost));

    // cleanup
    cudaFree(dlabels);cudaFree(dflag);
    cudaFree(dX);cudaFree(dY);cudaFree(dZ);
    free(h_sI);free(h_off);free(h_cnt);
    cudaFree(dI);cudaFree(d_sI);cudaFree(d_off);cudaFree(d_cnt);cudaFree(dOut);
    cudaEventDestroy(t0);cudaEventDestroy(t1);
    free(hcX);free(hcY);free(hcZ);free(hlabels);
}

// MAIN
// Parses the input point cloud, dispatches to whichever neighbourhood
// strategy was requested on the command line, and writes the remapped
// intensities (with original x/y/z) to the corresponding output file.
int main(int argc, char *argv[])
{
    if (argc < 3){
        fprintf(stderr,"Usage: %s input.txt knn|approx_knn|kmeans\n",argv[0]);
        return 1;
    }

    int N,K,T;
    int *X,*Y,*Z,*I;
    load_input(argv[1],&N,&K,&T,&X,&Y,&Z,&I);

    int *Out=(int*)malloc(N*sizeof(int));

    if (strcmp(argv[2],"knn")==0){
        run_exact_knn(N,K,X,Y,Z,I,Out);
        save_output("knn.txt",N,X,Y,Z,Out);

    } else if (strcmp(argv[2],"approx_knn")==0){
        run_approx_knn(N,K,X,Y,Z,I,Out);
        save_output("approx_knn.txt",N,X,Y,Z,Out);

    } else if (strcmp(argv[2],"kmeans")==0){
        run_kmeans(N,K,T,X,Y,Z,I,Out);
        save_output("kmeans.txt",N,X,Y,Z,Out);

    } else {
        fprintf(stderr,"Unknown mode: %s\n",argv[2]);
        return 1;
    }

    free(X);free(Y);free(Z);free(I);free(Out);
    return 0;
}
