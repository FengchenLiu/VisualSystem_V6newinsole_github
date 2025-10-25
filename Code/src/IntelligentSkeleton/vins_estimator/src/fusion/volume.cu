#include "volume.h"


using Vec3fda = Eigen::Matrix<float, 3, 1, Eigen::DontAlign>;

#define HESSIAN_LENGTH                                                   21
#define BIAS_LENGTH                                                       6
#define LOSS_LENGTH                                                       1
#define HBL_LENGTH             (HESSIAN_LENGTH + BIAS_LENGTH + LOSS_LENGTH)
#define MAX_BLOCK_SIZE                                                  512

void eventCreate(Event *event) {
  gpuErrchk(cudaEventCreateWithFlags(&event->cpy_htd, eventFlags()));
  gpuErrchk(cudaEventCreateWithFlags(&event->compute, eventFlags()));
  gpuErrchk(cudaEventCreateWithFlags(&event->cpy_dth, eventFlags()));
}

void eventDestroy(Event *event) {
  gpuErrchk(cudaEventDestroy(event->cpy_htd));
  gpuErrchk(cudaEventDestroy(event->compute));
  gpuErrchk(cudaEventDestroy(event->cpy_dth));
}

void eventSynchronize(Event *event) {
  gpuErrchk(cudaEventSynchronize(event->cpy_dth));
}

template<unsigned int N, unsigned int M>
__device__
void AtomicAdd(matNxM<N, M> *A, const matNxM<N, M> dA) {
  const int n = N * M;
#pragma unroll 1
  for (int i = 0; i < n; ++i) {
    atomicAdd(&(A->entries[i]), dA.entries[i]);
  }
}

template<typename T, int SIZE>
static __device__ __forceinline__
void ReduceSum(volatile T* buffer)
{
  const int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
  T value = buffer[thread_id];

  if (SIZE >= 1024) {
    if (thread_id < 512) buffer[thread_id] = value = value + buffer[thread_id + 512];
    __syncthreads();
  }
  if (SIZE >= 512) {
    if (thread_id < 256) buffer[thread_id] = value = value + buffer[thread_id + 256];
    __syncthreads();
  }
  if (SIZE >= 256) {
    if (thread_id < 128) buffer[thread_id] = value = value + buffer[thread_id + 128];
    __syncthreads();
  }
  if (SIZE >= 128) {
    if (thread_id < 64) buffer[thread_id] = value = value + buffer[thread_id + 64];
    __syncthreads();
  }

  if (thread_id < 32) {
    if (SIZE >= 64) buffer[thread_id] = value = value + buffer[thread_id + 32];
    if (SIZE >= 32) buffer[thread_id] = value = value + buffer[thread_id + 16];
    if (SIZE >= 16) buffer[thread_id] = value = value + buffer[thread_id + 8];
    if (SIZE >= 8) buffer[thread_id] = value = value + buffer[thread_id + 4];
    if (SIZE >= 4) buffer[thread_id] = value = value + buffer[thread_id + 2];
    if (SIZE >= 2) buffer[thread_id] = value = value + buffer[thread_id + 1];
  }
}



__host__ __device__ __forceinline__ int GetID(int x, int y, int z, int dim_x, int dim_y)
{
  return x + y * dim_x + z * dim_x * dim_y;
}

__host__ __device__ __forceinline__ int GetRemainder(int x, int y)
{
  return (x >= 0) ? x % y : y - ((-x) % y);
}

__host__ __device__ __forceinline__ int GlobalPosToVoxelID(const float3 global_vertex, float ox, float oy, float oz, int wx, int wy, int wz, int dx, int dy, int dz, float vs)
{
  float3 local_point;
  local_point.x = global_vertex.x - (ox + wx * vs);
  local_point.y = global_vertex.y - (oy + wy * vs);
  local_point.z = global_vertex.z - (oz + wz * vs);

  int3 volume_id;
  volume_id.x = roundf(local_point.x / vs);
  volume_id.y = roundf(local_point.y / vs);
  volume_id.z = roundf(local_point.z / vs);

  int bx = GetRemainder(wx, dx);
  int by = GetRemainder(wy, dy);
  int bz = GetRemainder(wz, dz);

  int3 voxel_id;
  voxel_id.x = GetRemainder(volume_id.x + bx, dx);
  voxel_id.y = GetRemainder(volume_id.y + by, dy);
  voxel_id.z = GetRemainder(volume_id.z + bz, dz);

  return GetID(voxel_id.x, voxel_id.y, voxel_id.z, dx, dy);
}


__host__ __device__ __forceinline__ int3 GetVolumeID(int tid_x, int tid_y, int tid_z, int bx, int by, int bz, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int3 vid;
  vid.x = GetRemainder(tid_x + bx, grid_dim_x);
  vid.y = GetRemainder(tid_y + by, grid_dim_y);
  vid.z = GetRemainder(tid_z + bz, grid_dim_z);
  return vid;
}

__host__ __device__ __forceinline__ int GetVoxelID(int tid_x, int tid_y, int tid_z, int bx, int by, int bz, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int3 vid = GetVolumeID(tid_x, tid_y, tid_z, bx, by, bz, grid_dim_x, grid_dim_y, grid_dim_z);
  return GetID(vid.x, vid.y, vid.z, grid_dim_x, grid_dim_y);
}

__host__ __device__ __forceinline__ float3 GetVoxelPosition(int tid_x, int tid_y, int tid_z, int wx, int wy, int wz, float vs, float ox, float oy, float oz)
{
  float3 pos;
  pos.x = ox + (wx + tid_x) * vs + vs / 2;
  pos.y = oy + (wy + tid_y) * vs + vs / 2;
  pos.z = oz + (wz + tid_z) * vs + vs / 2;
  return pos;
}

__host__ __device__ __forceinline__ mat3x3 GetSkewMatrix(float3 p)
{
  mat3x3 skew;
  skew.ptr()[0] =    0; skew.ptr()[1] = -p.z; skew.ptr()[2] =  p.y; 
  skew.ptr()[3] =  p.z; skew.ptr()[4] =    0; skew.ptr()[5] = -p.x; 
  skew.ptr()[6] = -p.y; skew.ptr()[7] =  p.x; skew.ptr()[8] =    0; 
  return skew;
}

__host__ __device__ __forceinline__ mat3x1 VecToMat(float3 vec)
{
  mat3x1 mat;
  mat.ptr()[0] = vec.x;
  mat.ptr()[1] = vec.y;
  mat.ptr()[2] = vec.z;
  return mat;
}

__host__  __forceinline__ Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& v)
{

  Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const double d = v.norm();
  const double d2 = d*d;

  Eigen::Matrix3d skew;
  skew <<     0,  -v.z(),   v.y(),
          v.z(),       0,  -v.x(),
         -v.y(),   v.x(),       0;
  
  if(d<1e-7)
    return (I + skew + 0.5 * skew * skew);
  else
    return (I + skew*sin(d)/d + skew*skew*(1.0f-cos(d))/d2);

}

__global__ void SelectRegionKernel(uchar3 *segment_img, uchar *binary_img, uchar3 target, int height, int width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    binary_img[x + y * width] = 0;

    if(segment_img[x + y * width].x == target.x &&
       segment_img[x + y * width].y == target.y &&
       segment_img[x + y * width].z == target.z)
      binary_img[x + y * width] = 255;
  }
}

__global__ void ResetVolumeKernel(Voxel *voxels, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int pt_grid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_grid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if(pt_grid_x >= grid_dim_x || pt_grid_y >= grid_dim_y) return;

  for (int pt_grid_z = 0; pt_grid_z < grid_dim_z; pt_grid_z++) {
    int volumd_id = pt_grid_z * grid_dim_y * grid_dim_x + pt_grid_y * grid_dim_x + pt_grid_x;
    voxels[volumd_id].Reset();
  }
}

__global__ void IntegrateKernel(Voxel *voxels, uchar3 *color_map, float *depth_map, float *K, float *Twc,
                                  int grid_dim_x, int grid_dim_y, int grid_dim_z,
                                  int cur_warp_x, int cur_warp_y, int cur_warp_z,
                                  float grid_origin_x, float grid_origin_y, float grid_origin_z, 
                                  float voxel_size, int height, int width, float trunc_margin, 
                                  float max_depth, float min_depth)
{

  int pt_grid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int pt_grid_y = blockIdx.y * blockDim.y + threadIdx.y;

  if(pt_grid_x >= grid_dim_x || pt_grid_y >= grid_dim_y) return;

  for (int pt_grid_z = 0; pt_grid_z < grid_dim_z; pt_grid_z++) {
    // Convert voxel center from grid coordinates to base frame camera coordinates
    // float3 voxel_position = GetVoxelPosition(pt_grid_x, pt_grid_y, pt_grid_z, cur_warp_x, cur_warp_y, cur_warp_z, voxel_size, grid_origin_x, grid_origin_y, grid_origin_z);
    float3 voxel_position;
    voxel_position.x = grid_origin_x + (cur_warp_x + pt_grid_x) * voxel_size + voxel_size / 2;
    voxel_position.y = grid_origin_y + (cur_warp_y + pt_grid_y) * voxel_size + voxel_size / 2;
    voxel_position.z = grid_origin_z + (cur_warp_z + pt_grid_z) * voxel_size + voxel_size / 2;

    // Convert from base frame camera coordinates to current frame camera coordinates
    float tmp_pt[3] = {0};
    tmp_pt[0] = voxel_position.x - Twc[0 * 4 + 3];
    tmp_pt[1] = voxel_position.y - Twc[1 * 4 + 3];
    tmp_pt[2] = voxel_position.z - Twc[2 * 4 + 3];
    float pt_cam_x = Twc[0 * 4 + 0] * tmp_pt[0] + Twc[1 * 4 + 0] * tmp_pt[1] + Twc[2 * 4 + 0] * tmp_pt[2];
    float pt_cam_y = Twc[0 * 4 + 1] * tmp_pt[0] + Twc[1 * 4 + 1] * tmp_pt[1] + Twc[2 * 4 + 1] * tmp_pt[2];
    float pt_cam_z = Twc[0 * 4 + 2] * tmp_pt[0] + Twc[1 * 4 + 2] * tmp_pt[1] + Twc[2 * 4 + 2] * tmp_pt[2];

    if (pt_cam_z <= 0)
      continue;

    int pt_pix_x = roundf(K[0 * 3 + 0] * (pt_cam_x / pt_cam_z) + K[0 * 3 + 2]);
    int pt_pix_y = roundf(K[1 * 3 + 1] * (pt_cam_y / pt_cam_z) + K[1 * 3 + 2]);
    if (pt_pix_x < 0 || pt_pix_x >= width || pt_pix_y < 0 || pt_pix_y >= height)
      continue;

    uchar3 color_val = color_map[pt_pix_y * width + pt_pix_x];
    float depth_val = depth_map[pt_pix_y * width + pt_pix_x];

    if (depth_val <= min_depth || depth_val > max_depth)
      continue;

    float sdf = depth_val - pt_cam_z;

    // float weight = 2.0f / (depth_val * depth_val + 1.0f);
    // float weight = 1.0f;
    float weight = 2.0f;
    // float weight = 2.5f;
    // float weight = 3.0f;
    
    // int volumd_id = GetVoxelID(pt_grid_x, pt_grid_y, pt_grid_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z);
    int id_x = (pt_grid_x + cur_warp_x) >= 0 ? (pt_grid_x + cur_warp_x) % grid_dim_x : grid_dim_x - (-((pt_grid_x + cur_warp_x)) % grid_dim_x);
    int id_y = (pt_grid_y + cur_warp_y) >= 0 ? (pt_grid_y + cur_warp_y) % grid_dim_y : grid_dim_y - (-((pt_grid_y + cur_warp_y)) % grid_dim_y);
    int id_z = (pt_grid_z + cur_warp_z) >= 0 ? (pt_grid_z + cur_warp_z) % grid_dim_z : grid_dim_z - (-((pt_grid_z + cur_warp_z)) % grid_dim_z);
    int volumd_id = id_z * grid_dim_y * grid_dim_x + id_y * grid_dim_x + id_x;
    voxels[volumd_id].Combine(sdf, trunc_margin, color_val, weight, 4);
  }
}

__global__ void UpdateSegmentKernel(Voxel *voxels, ushort *index_map, float *depth_map, float *K, float *Twc,
                                    int grid_dim_x, int grid_dim_y, int grid_dim_z,
                                    int cur_warp_x, int cur_warp_y, int cur_warp_z,
                                    float grid_origin_x, float grid_origin_y, float grid_origin_z, 
                                    float voxel_size, int height, int width, float trunc_margin, 
                                    float max_depth, float min_depth)
{
  int tx = threadIdx.x + blockIdx.x * blockDim.x;
  int ty = threadIdx.y + blockIdx.y * blockDim.y;

  if(tx >= width || ty >= height)
    return;

  int image_id = tx + ty * width;

  ushort index = index_map[image_id];

  float depth_value = depth_map[image_id];
  if (depth_value > max_depth || depth_value < min_depth)
    return;
  
  float3 vertex;

  vertex.x = (tx - K[2]) * depth_value / K[0];
  vertex.y = (ty - K[5]) * depth_value / K[4];
  vertex.z = depth_value;

  float3 global_vertex;
  global_vertex.x = Twc[0] * vertex.x + Twc[1] * vertex.y + Twc[2] * vertex.z + Twc[3];
  global_vertex.y = Twc[4] * vertex.x + Twc[5] * vertex.y + Twc[6] * vertex.z + Twc[7];
  global_vertex.z = Twc[8] * vertex.x + Twc[9] * vertex.y + Twc[10] * vertex.z + Twc[11];

  float3 local_point;
  local_point.x = global_vertex.x - (grid_origin_x + cur_warp_x * voxel_size);
  local_point.y = global_vertex.y - (grid_origin_y + cur_warp_y * voxel_size);
  local_point.z = global_vertex.z - (grid_origin_z + cur_warp_z * voxel_size);

  int3 volume_id;
  volume_id.x = roundf(local_point.x / voxel_size);
  volume_id.y = roundf(local_point.y / voxel_size);
  volume_id.z = roundf(local_point.z / voxel_size);

  int bottom_x = GetRemainder(cur_warp_x, grid_dim_x);
  int bottom_y = GetRemainder(cur_warp_y, grid_dim_y);
  int bottom_z = GetRemainder(cur_warp_z, grid_dim_z);

  int3 voxel_id;
  voxel_id.x = GetRemainder(volume_id.x + bottom_x, grid_dim_x);
  voxel_id.y = GetRemainder(volume_id.y + bottom_y, grid_dim_y);
  voxel_id.z = GetRemainder(volume_id.z + bottom_z, grid_dim_z);

  const int &x = voxel_id.x, &y = voxel_id.y, &z = voxel_id.z;
  const int xp = (x+1) % grid_dim_x, xn = (x-1) % grid_dim_x,
            yp = (y+1) % grid_dim_y, yn = (y-1) % grid_dim_y,
            zp = (z+1) % grid_dim_z, zn = (z-1) % grid_dim_z;

  voxels[GetID(x, y, z, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(xp, y, z, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(x, yp, z, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(x, y, zp, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(xn, y, z, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(x, yn, z, grid_dim_x, grid_dim_y)].Update(index);
  voxels[GetID(x, y, zn, grid_dim_x, grid_dim_y)].Update(index);
}



__global__ void ComputeVertexMapKernel(float *depth_map, float3 *vertex_map, int height, int width, float *K, float max_depth, float min_depth)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= width || y >= height)
    return;

  int id = x + y * width;

  float depth_value = depth_map[id];
  if (depth_value > max_depth || depth_value < min_depth) { 
    depth_value = 0.f; 
  }

  vertex_map[id].x = (x - K[2]) * depth_value / K[0];
  vertex_map[id].y = (y - K[5]) * depth_value / K[4];
  vertex_map[id].z = depth_value;
}

__global__ void ComputeNormalMapKernel(float3 *vertex_map, float3 *normal_map, int height, int width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  int id = x + y * width;
  const Vec3fda left = Vec3fda(vertex_map[id - 1].x, vertex_map[id - 1].y, vertex_map[id - 1].z);
  const Vec3fda right = Vec3fda(vertex_map[id + 1].x, vertex_map[id + 1].y, vertex_map[id + 1].z);
  const Vec3fda upper = Vec3fda(vertex_map[id - width].x, vertex_map[id - width].y, vertex_map[id - width].z);
  const Vec3fda lower = Vec3fda(vertex_map[id + width].x, vertex_map[id + width].y, vertex_map[id + width].z);
  const Vec3fda center = Vec3fda(vertex_map[id].x, vertex_map[id].y, vertex_map[id].z);

  Vec3fda normal;

  if(center.z() == 0 || left.z() == 0 || right.z() == 0 || upper.z() == 0 || lower.z() == 0)
    normal = Vec3fda(0, 0, 0);
  else {
    Vec3fda hor(left.x() - right.x(), left.y() - right.y(), left.z() - right.z());
    Vec3fda ver(upper.x() - lower.x(), upper.y() - lower.y(), upper.z() - lower.z());

    normal = hor.cross(ver);
    normal.normalize();

    if (normal.z() > 0)
      normal *= -1;
  }

  normal_map[id] = make_float3(normal.x(), normal.y(), normal.z());
}


__global__ void ReleaseXKernel(Voxel *voxels, int bottom, int delta, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int z = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int top = bottom + delta;
  top = (top >= 0) ? top % grid_dim_x : grid_dim_x - ((-top) % grid_dim_x);
  const bool warp = (top != bottom + delta);

  if (z < grid_dim_z && y < grid_dim_y) {
    for(int x = 0; x < grid_dim_x; x++) {
      if(!warp ? (x >= bottom && x < top) : (x >= bottom || x < top)) {
        int id = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y;
        voxels[id].Reset();
      }
    }
  }
}

__global__ void ReleaseYKernel(Voxel *voxels, int bottom, int delta, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int z = threadIdx.y + blockIdx.y * blockDim.y;

  int top = bottom + delta;
  top = (top >= 0) ? top % grid_dim_y : grid_dim_y - ((-top) % grid_dim_y);
  const bool warp = (top != bottom + delta);

  if (x < grid_dim_x && z < grid_dim_z) {
    for(int y = 0; y < grid_dim_y; y++) {
      if(!warp ? (y >= bottom && y < top) : (y >= bottom || y < top)) {
        int id = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y;
        voxels[id].Reset();
      }
    }
  }

}


__global__ void ReleaseZKernel(Voxel *voxels, int bottom, int delta, int grid_dim_x, int grid_dim_y, int grid_dim_z)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  int top = bottom + delta;
  top = (top >= 0) ? top % grid_dim_z : grid_dim_z - ((-top) % grid_dim_z);
  const bool warp = (top != bottom + delta);

  if (x < grid_dim_x && y < grid_dim_y) {
    for(int z = 0; z < grid_dim_z; z++) {
      if(!warp ? (z >= bottom && z < top) : (z >= bottom || z < top)) {
        int id = x + y * grid_dim_x + z * grid_dim_x * grid_dim_y;
        voxels[id].Reset();
      }
    }
  }
}


__global__ void ExtractSurfaceKernel(Voxel *voxels, BGRPoint* surface, int *surface_num, float voxel_size,
                                    int grid_dim_x, int grid_dim_y, int grid_dim_z, int bottom_x, int bottom_y, int bottom_z,
                                    int offset_x, int offset_y, int offset_z, float grid_origin_x, float grid_origin_y, float grid_origin_z)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= grid_dim_x || y >= grid_dim_y) return;

  for (int z = 0; z < grid_dim_z; z+=1) {
    
    
    int id = GetID(x, y, z, grid_dim_x, grid_dim_y);
    voxels[id].is_surface = 0;
    
    int xp = (x+1) % grid_dim_x;
    int yp = (y+1) % grid_dim_y;
    int zp = (z+1) % grid_dim_z;

    if(xp == bottom_x || yp == bottom_y || zp == bottom_z) {
      
      continue;

      if(std::abs(voxels[id].tsdf) < 0.5 && voxels[id].weight > 0.1f) {
      // if(std::abs(voxels[id].tsdf) < 0.5) {
        int index = atomicAdd(surface_num, 1);
        int3 voxel_gid;
        voxel_gid.x = (x < bottom_x) ? (x + offset_x + grid_dim_x) : (x + offset_x);
        voxel_gid.y = (y < bottom_y) ? (y + offset_y + grid_dim_y) : (y + offset_y);
        voxel_gid.z = (z < bottom_z) ? (z + offset_z + grid_dim_z) : (z + offset_z);

        float pt_base_x = grid_origin_x + voxel_gid.x * voxel_size + voxel_size * 0.5;
        float pt_base_y = grid_origin_y + voxel_gid.y * voxel_size + voxel_size * 0.5;
        float pt_base_z = grid_origin_z + voxel_gid.z * voxel_size + voxel_size * 0.5;

        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, voxels[id].color, voxels[id].plane_id);
      }
        
    }

    int id_xp = GetID(xp, y, z, grid_dim_x, grid_dim_y);
    int id_yp = GetID(x, yp, z, grid_dim_x, grid_dim_y);
    int id_zp = GetID(x, y, zp, grid_dim_x, grid_dim_y);

    const float tsdf = voxels[id].tsdf;
    const float weight = voxels[id].weight;
    const uchar3 color = voxels[id].color;

    if(std::abs(tsdf) >= 0.98 || weight < 0.1f) continue;
    // if(std::abs(tsdf) >= 0.98) continue;
    
    const float tsdf_xp = voxels[id_xp].tsdf;
    const float tsdf_yp = voxels[id_yp].tsdf;
    const float tsdf_zp = voxels[id_zp].tsdf;

    const float weight_xp = voxels[id_xp].weight;
    const float weight_yp = voxels[id_yp].weight;
    const float weight_zp = voxels[id_zp].weight;

    if(weight_xp < 0.1f || weight_yp < 0.1f || weight_zp < 0.1f) continue;

    const bool is_surface_x = ((tsdf > 0) && (tsdf_xp < 0)) || ((tsdf < 0) && (tsdf_xp > 0));
    const bool is_surface_y = ((tsdf > 0) && (tsdf_yp < 0)) || ((tsdf < 0) && (tsdf_yp > 0));
    const bool is_surface_z = ((tsdf > 0) && (tsdf_zp < 0)) || ((tsdf < 0) && (tsdf_zp > 0));

    if (is_surface_x || is_surface_y || is_surface_z) {
      float3 normal;
      normal.x = (tsdf_xp - tsdf);
      normal.y = (tsdf_yp - tsdf);
      normal.z = (tsdf_zp - tsdf);
      if(norm(normal) == 0)
        continue;

      normal = normalize(normal);

      int index = atomicAdd(surface_num, 1);

      int3 voxel_gid;
      voxel_gid.x = (x < bottom_x) ? (x + offset_x + grid_dim_x) : (x + offset_x);
      voxel_gid.y = (y < bottom_y) ? (y + offset_y + grid_dim_y) : (y + offset_y);
      voxel_gid.z = (z < bottom_z) ? (z + offset_z + grid_dim_z) : (z + offset_z);

      float pt_base_x = grid_origin_x + voxel_gid.x * voxel_size + voxel_size * 0.5;
      float pt_base_y = grid_origin_y + voxel_gid.y * voxel_size + voxel_size * 0.5;
      float pt_base_z = grid_origin_z + voxel_gid.z * voxel_size + voxel_size * 0.5;

      float delta_x = abs(tsdf) / abs(tsdf_xp - tsdf) * voxel_size;
      float delta_y = abs(tsdf) / abs(tsdf_yp - tsdf) * voxel_size;
      float delta_z = abs(tsdf) / abs(tsdf_zp - tsdf) * voxel_size;

      // if (is_surface_x) {
      //   pt_base_x = pt_base_x + delta_x;
      //   surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color);
      // }
      // if (is_surface_y) {
      //   pt_base_y = pt_base_y + delta_y;
      //   surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color);
      // }
      // if (is_surface_z) {
      //   pt_base_z = pt_base_z + delta_z;
      //   surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color);
      // }

      if(is_surface_x && !is_surface_y && !is_surface_z) {
        pt_base_x = pt_base_x + delta_x;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_xp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(!is_surface_x && is_surface_y && !is_surface_z) {
        pt_base_y = pt_base_y + delta_y;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_yp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(!is_surface_x && !is_surface_y && is_surface_z) {
        pt_base_z = pt_base_z + delta_z;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_zp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(is_surface_x && is_surface_y && !is_surface_z) {
        pt_base_x = pt_base_x + delta_x;
        pt_base_y = pt_base_y + delta_y;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_xp].is_surface = true;
        // voxels[id_yp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(is_surface_x && !is_surface_y && is_surface_z) {
        pt_base_x = pt_base_x + delta_x;
        pt_base_z = pt_base_z + delta_z;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_xp].is_surface = true;
        // voxels[id_zp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(!is_surface_x && is_surface_y && is_surface_z) {
        pt_base_y = pt_base_y + delta_y;
        pt_base_z = pt_base_z + delta_z;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        // voxels[id].is_surface = true;
        // voxels[id_yp].is_surface = true;
        // voxels[id_zp].is_surface = true;

        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
      else if(is_surface_x && is_surface_y && is_surface_z) {
        pt_base_x = pt_base_x + delta_x;
        pt_base_y = pt_base_y + delta_y;
        pt_base_z = pt_base_z + delta_z;
        surface[index++].Set(pt_base_x, pt_base_y, pt_base_z, color, voxels[id].plane_id);
        voxels[id].is_surface = 1;
        voxels[id_xp].is_surface = 1;
        voxels[id_yp].is_surface = 1;
        voxels[id_zp].is_surface = 1;
      }
    }
  }
}

__global__ void VerticalSearchKernel(Voxel *voxels, unsigned char *labels, unsigned short *ids, 
                                     const float* Twc, int x2d, int y2d, int grid_dim_x, int grid_dim_y)
{
  int z = threadIdx.x + blockIdx.x * blockDim.x;

  Voxel &voxel = voxels[GetID(x2d, y2d, z, grid_dim_x, grid_dim_y)];
  if(voxel.tsdf < 0) {
    labels[z] = 0;
    if(voxel.plane_cnt > 0)
      ids[z] = voxel.plane_id;
    else 
      ids[z] = 0;
  }
  else {
    labels[z] = 1;
  }

}

inline bool InterpolateTrilinearyHost(Voxel *voxels, int3 &voxel_id, float& tsdf, float3 global_point, int bx, int by, int bz, 
                                                      float ox, float oy, float oz, int wx, int wy, int wz,
                                                      int dx, int dy, int dz, float voxel_size)
{
  float3 local_point;
  local_point.x = global_point.x - (ox + wx * voxel_size);
  local_point.y = global_point.y - (oy + wy * voxel_size);
  local_point.z = global_point.z - (oz + wz * voxel_size);

  int3 volume_id;
  volume_id.x = roundf(local_point.x / voxel_size) - 1;
  volume_id.y = roundf(local_point.y / voxel_size) - 1;
  volume_id.z = roundf(local_point.z / voxel_size) - 1;

  voxel_id.x = GetRemainder(volume_id.x + bx, dx);
  voxel_id.y = GetRemainder(volume_id.y + by, dy);
  voxel_id.z = GetRemainder(volume_id.z + bz, dz);

  float a = (local_point.x - ((volume_id.x + 0.5) * voxel_size)) / voxel_size;
  float b = (local_point.y - ((volume_id.y + 0.5) * voxel_size)) / voxel_size;
  float c = (local_point.z - ((volume_id.z + 0.5) * voxel_size)) / voxel_size;

  const int &x = voxel_id.x, &y = voxel_id.y, &z = voxel_id.z;
  int xp = (x+1) % dx, yp = (y+1) % dy, zp = (z+1) % dz;
  if(xp == bx || yp == by || zp == bz)
    return false;

  tsdf = voxels[GetID(x, y, z, dx, dy)].tsdf * (1 - a) * (1 - b) * (1 - c) +
         voxels[GetID(x, y, zp, dx, dy)].tsdf * (1 - a) * (1 - b) * c +
         voxels[GetID(x, yp, z, dx, dy)].tsdf * (1 - a) * b * (1 - c) +
         voxels[GetID(x, yp, zp, dx, dy)].tsdf * (1 - a) * b * c +
         voxels[GetID(xp, y, z, dx, dy)].tsdf* a * (1 - b) * (1 - c) +
         voxels[GetID(xp, y, zp, dx, dy)].tsdf * a * (1 - b) * c +
         voxels[GetID(xp, yp, z, dx, dy)].tsdf * a * b * (1 - c) +
         voxels[GetID(xp, yp, zp, dx, dy)].tsdf * a * b * c;

  return true;
}

__device__ __forceinline__ bool InterpolateTrilineary(Voxel *voxels, int3 &voxel_id, float& tsdf, float3 global_point, int bx, int by, int bz, 
                                                      float ox, float oy, float oz, int wx, int wy, int wz,
                                                      int dx, int dy, int dz, float voxel_size)
{
  float3 local_point;
  local_point.x = global_point.x - (ox + wx * voxel_size);
  local_point.y = global_point.y - (oy + wy * voxel_size);
  local_point.z = global_point.z - (oz + wz * voxel_size);

  int3 volume_id;
  volume_id.x = roundf(local_point.x / voxel_size) - 1;
  volume_id.y = roundf(local_point.y / voxel_size) - 1;
  volume_id.z = roundf(local_point.z / voxel_size) - 1;

  voxel_id.x = GetRemainder(volume_id.x + bx, dx);
  voxel_id.y = GetRemainder(volume_id.y + by, dy);
  voxel_id.z = GetRemainder(volume_id.z + bz, dz);

  float a = (local_point.x - ((volume_id.x + 0.5) * voxel_size)) / voxel_size;
  float b = (local_point.y - ((volume_id.y + 0.5) * voxel_size)) / voxel_size;
  float c = (local_point.z - ((volume_id.z + 0.5) * voxel_size)) / voxel_size;

  const int &x = voxel_id.x, &y = voxel_id.y, &z = voxel_id.z;
  int xp = (x+1) % dx, yp = (y+1) % dy, zp = (z+1) % dz;
  if(xp == bx || yp == by || zp == bz)
    return false;

  tsdf = voxels[GetID(x, y, z, dx, dy)].tsdf * (1 - a) * (1 - b) * (1 - c) +
         voxels[GetID(x, y, zp, dx, dy)].tsdf * (1 - a) * (1 - b) * c +
         voxels[GetID(x, yp, z, dx, dy)].tsdf * (1 - a) * b * (1 - c) +
         voxels[GetID(x, yp, zp, dx, dy)].tsdf * (1 - a) * b * c +
         voxels[GetID(xp, y, z, dx, dy)].tsdf* a * (1 - b) * (1 - c) +
         voxels[GetID(xp, y, zp, dx, dy)].tsdf * a * (1 - b) * c +
         voxels[GetID(xp, yp, z, dx, dy)].tsdf * a * b * (1 - c) +
         voxels[GetID(xp, yp, zp, dx, dy)].tsdf * a * b * c;

  return true;
}

__global__ void RayCastingKernel(Voxel *voxels, float *K, float *virtual_depth, float3 *virtual_vertex, 
                                  float3 *virtual_normal, ushort *virtual_index, float *Twc, 
                                  int grid_dim_x, int grid_dim_y, int grid_dim_z,
                                  int cur_warp_x, int cur_warp_y, int cur_warp_z, 
                                  float grid_origin_x, float grid_origin_y, float grid_origin_z, 
                                  float voxel_size, int height, int width, float trunc_margin, 
                                  float max_depth, float min_depth)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= width || y >= height)
    return;

  int image_id = x + y * width;
  virtual_depth[image_id] = 0.0;
  virtual_vertex[image_id] = make_float3(0, 0, 0);
  virtual_normal[image_id] = make_float3(0, 0, 0);
  virtual_index[image_id] = 0;

  const float3 ray_dir_c{(x - K[2]) / K[0], (y - K[5]) / K[4], 1.0f};

  float3 ray_start{Twc[3], Twc[7], Twc[11]};
  float3 ray_dir;
  ray_dir.x = Twc[0] * ray_dir_c.x + Twc[1] * ray_dir_c.y + Twc[2] * ray_dir_c.z;
  ray_dir.y = Twc[4] * ray_dir_c.x + Twc[5] * ray_dir_c.y + Twc[6] * ray_dir_c.z;
  ray_dir.z = Twc[8] * ray_dir_c.x + Twc[9] * ray_dir_c.y + Twc[10] * ray_dir_c.z;

  int bottom_x = GetRemainder(cur_warp_x, grid_dim_x);
  int bottom_y = GetRemainder(cur_warp_y, grid_dim_y);
  int bottom_z = GetRemainder(cur_warp_z, grid_dim_z);

  float current_depth = 0.05f, prev_depth = 0.05f;
  float3 current_point, local_point;
  int3 voxel_id;
  int vec_id;
  float prev_tsdf = 1.0f, cur_tsdf = 1.0f;
  unsigned char result = 0;
  float3 normal = make_float3(0, 0, 0);

  while(current_depth < max_depth) {

    current_point = ray_dir * current_depth + ray_start;
    prev_tsdf = cur_tsdf;
    bool res = InterpolateTrilineary(voxels, voxel_id, cur_tsdf, current_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
    if(!res)
      return;
    
    vec_id = GetID(voxel_id.x, voxel_id.y, voxel_id.z, grid_dim_x, grid_dim_y);

    if(prev_tsdf < 0.0f && cur_tsdf > 0.0f) {
      result = 1;
      break;
    }
    else if(prev_tsdf > 0.0f && cur_tsdf < 0.0f) {
      
      float near_depth = current_depth + (cur_tsdf) / (prev_tsdf - cur_tsdf) * voxel_size;
      current_depth = near_depth;
      current_point = ray_dir * current_depth + ray_start;
      result = 2;
      int &vx = voxel_id.x, &vy = voxel_id.y, &vz = voxel_id.z;
      int vxp = (vx + 1) % grid_dim_x, vyp = (vy + 1) % grid_dim_y, vzp = (vz + 1) % grid_dim_z;
      if(vxp == bottom_x || vyp == bottom_y || vzp == bottom_z || vx == bottom_x || vy == bottom_y || vz == bottom_z)
        break;
      
      bool normal_result = false;
      float3 pt_point, mt_point;
      float pt_tsdf, mt_tsdf;

      // compute normal x
      pt_point = current_point + make_float3(voxel_size * 0.5, 0, 0);
      mt_point = current_point - make_float3(voxel_size * 0.5, 0, 0);

      normal_result = InterpolateTrilineary(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;
      normal_result = InterpolateTrilineary(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;

      normal.x = (pt_tsdf - mt_tsdf);

      // compute normal y
      pt_point = current_point + make_float3(0, voxel_size * 0.5, 0);
      mt_point = current_point - make_float3(0, voxel_size * 0.5, 0);

      normal_result = InterpolateTrilineary(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;
      normal_result = InterpolateTrilineary(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;

      normal.y = (pt_tsdf - mt_tsdf);
      
      // compute normal z
      pt_point = current_point + make_float3(0, 0, voxel_size * 0.5);
      mt_point = current_point - make_float3(0, 0, voxel_size * 0.5);

      normal_result = InterpolateTrilineary(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;
      normal_result = InterpolateTrilineary(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                    grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
      if(!normal_result)
        break;

      normal.z = (pt_tsdf - mt_tsdf);

      break;
    }

    float delta;
    if(voxels[vec_id].weight < 0.1 || cur_tsdf > 0.98f) {
      delta = trunc_margin / 2;
    }
    else {
      delta = voxel_size;
    }
    prev_depth = current_depth;
    current_depth += delta;
  }

  if (result == 2) {
    
    int vid = GlobalPosToVoxelID(current_point, grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);

    virtual_depth[image_id] = current_depth;
    virtual_vertex[image_id] = current_point;

    if(normal.x == 0 && normal.y == 0 && normal.z == 0) {
      virtual_normal[image_id] = make_float3(0, 0, 0);
      virtual_index[image_id] = 0;      
    }
    else {
      normalize(normal);
      virtual_normal[image_id] = normal;
      virtual_index[image_id] = voxels[vid].plane_id;
    }
      
  } else {
    virtual_depth[image_id] = 0.0;
    virtual_vertex[image_id] = make_float3(0, 0, 0);
    virtual_normal[image_id] = make_float3(0, 0, 0);
    virtual_index[image_id] = 0;    
  }
}

__global__ void ResetMaskKernel(unsigned char *mask, float *render_depth, int height, int width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < width && y < height) {
    int image_id = x + y * width;
    mask[image_id] = (unsigned char)0;
    render_depth[image_id] = 0.0f;
  }

}

__global__ void ResetBlockDataKernel(double *block_data, double *sum_data, int length)
{
  if(threadIdx.x < length && blockIdx.x < HBL_LENGTH) {
    block_data[length * blockIdx.x + threadIdx.x] = 0.0f;
    sum_data[blockIdx.x] = 0.0f;
  }
}

__global__ void AssociateKernel(float3 *host_vertex_map, float3 *host_normal_map, float3 *frame_vertex_map, float *Twc, float *Tcw_host, float *K, float *render_depth,
                                unsigned char *mask, float3 *residual_vertex, float3 *residual_host_vertex, float3 *residual_host_normal, float *residual_host_depth, 
                                int *match_num, float dist_thres, int height, int width, int divide, float max_depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x % DIVIDE == 0 && y % DIVIDE == 0 && x < width && y < height) {
    int image_id = x + y * width;
    mask[image_id] = (unsigned char)0;
    float3 vertex = frame_vertex_map[image_id];

    if(!(vertex.x == 0 && vertex.y == 0 && vertex.z == 0)) {
      
      float3 global_vertex;
      global_vertex.x = Twc[0] * vertex.x + Twc[1] * vertex.y + Twc[2] * vertex.z + Twc[3];
      global_vertex.y = Twc[4] * vertex.x + Twc[5] * vertex.y + Twc[6] * vertex.z + Twc[7];
      global_vertex.z = Twc[8] * vertex.x + Twc[9] * vertex.y + Twc[10] * vertex.z + Twc[11];

      float3 vertex_cam;
      vertex_cam.x = Tcw_host[0] * global_vertex.x + Tcw_host[1] * global_vertex.y + Tcw_host[2] * global_vertex.z + Tcw_host[3];
      vertex_cam.y = Tcw_host[4] * global_vertex.x + Tcw_host[5] * global_vertex.y + Tcw_host[6] * global_vertex.z + Tcw_host[7];
      vertex_cam.z = Tcw_host[8] * global_vertex.x + Tcw_host[9] * global_vertex.y + Tcw_host[10] * global_vertex.z + Tcw_host[11];

      if (vertex_cam.z > 0 && vertex_cam.z < max_depth) {
        int host_x = roundf(K[0 * 3 + 0] * (vertex_cam.x / vertex_cam.z) + K[0 * 3 + 2]);
        int host_y = roundf(K[1 * 3 + 1] * (vertex_cam.y / vertex_cam.z) + K[1 * 3 + 2]);
        if (host_x >= 0 && host_x < width && host_y >= 0 && host_y < height) {
          int host_image_id = host_x + host_y * width;
          float3 host_vertex = host_vertex_map[host_image_id];
          float3 host_normal = host_normal_map[host_image_id];

          if(!(host_vertex.x == 0 && host_vertex.y == 0 && host_vertex.z == 0) && !(host_normal.x == 0 && host_normal.y == 0 && host_normal.z == 0)) {
            float residual = dot(host_normal, global_vertex - host_vertex);
            render_depth[host_image_id] = vertex_cam.z;
            if(fabs(residual) <= dist_thres) {
              mask[image_id] = (unsigned char)255;
              int index = atomicAdd(match_num, 1);
              residual_vertex[index] = vertex;
              residual_host_vertex[index] = host_vertex;
              residual_host_normal[index] = host_normal;
              residual_host_depth[index] = vertex_cam.z;
            }
          }
        }
      }        
    }
  }
} 

// __global__ void AssociateColorKernel(float3 *host_vertex_map, uchar3 *host_seg_map, float3 *frame_vertex_map, float *Twc, float *Tcw_host, float *K, float *render_depth,
//                                      unsigned char *mask, float3 *residual_vertex, float3 *residual_host_vertex, uchar3 *residual_host_seg, float *residual_depth, 
//                                      int *match_num, float dist_thres, int height, int width, int divide, float max_depth)
// {
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;

//   if(x % DIVIDE == 0 && y % DIVIDE == 0 && x < width && y < height) {
//     int image_id = x + y * width;
//     mask[image_id] = (unsigned char)0;
//     float3 vertex = frame_vertex_map[image_id];

//     if(!(vertex.x == 0 && vertex.y == 0 && vertex.z == 0)) {
      
//       float3 global_vertex;
//       global_vertex.x = Twc[0] * vertex.x + Twc[1] * vertex.y + Twc[2] * vertex.z + Twc[3];
//       global_vertex.y = Twc[4] * vertex.x + Twc[5] * vertex.y + Twc[6] * vertex.z + Twc[7];
//       global_vertex.z = Twc[8] * vertex.x + Twc[9] * vertex.y + Twc[10] * vertex.z + Twc[11];

//       float3 vertex_cam;
//       vertex_cam.x = Tcw_host[0] * global_vertex.x + Tcw_host[1] * global_vertex.y + Tcw_host[2] * global_vertex.z + Tcw_host[3];
//       vertex_cam.y = Tcw_host[4] * global_vertex.x + Tcw_host[5] * global_vertex.y + Tcw_host[6] * global_vertex.z + Tcw_host[7];
//       vertex_cam.z = Tcw_host[8] * global_vertex.x + Tcw_host[9] * global_vertex.y + Tcw_host[10] * global_vertex.z + Tcw_host[11];

//       if (vertex_cam.z > 0 && vertex_cam.z < max_depth) {
//         int host_x = roundf(K[0 * 3 + 0] * (vertex_cam.x / vertex_cam.z) + K[0 * 3 + 2]);
//         int host_y = roundf(K[1 * 3 + 1] * (vertex_cam.y / vertex_cam.z) + K[1 * 3 + 2]);
//         if (host_x >= 0 && host_x < width && host_y >= 0 && host_y < height) {
//           int host_image_id = host_x + host_y * width;
//           float3 host_vertex = host_vertex_map[host_image_id];
//           uchar3 host_seg = host_seg_map[host_image_id];

//           if(!(host_vertex.x == 0 && host_vertex.y == 0 && host_vertex.z == 0) && !(host_normal.x == 0 && host_normal.y == 0 && host_normal.z == 0)) {
//             float residual = dot(host_normal, global_vertex - host_vertex);
//             render_depth[host_image_id] = vertex_cam.z;
//             if(fabs(residual) <= dist_thres) {
//               mask[image_id] = (unsigned char)255;
//               int index = atomicAdd(match_num, 1);
//               residual_vertex[index] = vertex;
//               residual_host_vertex[index] = host_vertex;
//               residual_host_seg[index] = host_seg;
//               residual_depth[index] = vertex.z;
//             }
//           }
//         }
//       }        
//     }
//   }
// } 



__global__ void EvaluateKernel(float3 *host_vertex_map, float3 *host_normal_map, float3 *frame_vertex_map, float *Twc, float *render_depth, 
                                float *Tcw_host, float *K, unsigned char *mask, double *block_data, int *block_num, 
                                float dist_thres, int height, int width, int grid_size, float max_depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  mat6x6 H;
  mat6x1 b;
  float loss;
  H.setZero();
  b.setZero();
  loss = 0.0;
  int cnt = 0;
  bool valid_match = false;


  if(x < width && y < height) {
    int image_id = x + y * width;
    mask[image_id] = (unsigned char)0;
    float3 vertex = frame_vertex_map[image_id];

    if(!(vertex.x == 0 && vertex.y == 0 && vertex.z == 0)) {
      
      float3 global_vertex;
      global_vertex.x = Twc[0] * vertex.x + Twc[1] * vertex.y + Twc[2] * vertex.z + Twc[3];
      global_vertex.y = Twc[4] * vertex.x + Twc[5] * vertex.y + Twc[6] * vertex.z + Twc[7];
      global_vertex.z = Twc[8] * vertex.x + Twc[9] * vertex.y + Twc[10] * vertex.z + Twc[11];

      float3 vertex_cam;
      vertex_cam.x = Tcw_host[0] * global_vertex.x + Tcw_host[1] * global_vertex.y + Tcw_host[2] * global_vertex.z + Tcw_host[3];
      vertex_cam.y = Tcw_host[4] * global_vertex.x + Tcw_host[5] * global_vertex.y + Tcw_host[6] * global_vertex.z + Tcw_host[7];
      vertex_cam.z = Tcw_host[8] * global_vertex.x + Tcw_host[9] * global_vertex.y + Tcw_host[10] * global_vertex.z + Tcw_host[11];

      if (vertex_cam.z > 0 && vertex_cam.z < max_depth) {
        int host_x = roundf(K[0 * 3 + 0] * (vertex_cam.x / vertex_cam.z) + K[0 * 3 + 2]);
        int host_y = roundf(K[1 * 3 + 1] * (vertex_cam.y / vertex_cam.z) + K[1 * 3 + 2]);
        if (host_x >= 0 && host_x < width && host_y >= 0 && host_y < height) {
          int host_image_id = host_x + host_y * width;
          float3 host_vertex = host_vertex_map[host_image_id];
          float3 host_normal = host_normal_map[host_image_id];

          if(!(host_vertex.x == 0 && host_vertex.y == 0 && host_vertex.z == 0) && !(host_normal.x == 0 && host_normal.y == 0 && host_normal.z == 0)) {
            float residual = dot(host_normal, global_vertex - host_vertex);
            render_depth[host_image_id] = vertex_cam.z;
            if(fabs(residual) <= dist_thres) {
              mask[image_id] = (unsigned char)255;
              cnt = 1;
              mat3x3 Rwc;
              Rwc.ptr()[0] = Twc[0]; Rwc.ptr()[1] = Twc[1]; Rwc.ptr()[2] = Twc[2];
              Rwc.ptr()[3] = Twc[4]; Rwc.ptr()[4] = Twc[5]; Rwc.ptr()[5] = Twc[6];
              Rwc.ptr()[6] = Twc[8]; Rwc.ptr()[7] = Twc[9]; Rwc.ptr()[8] = Twc[10];

              mat1x3 host_normal_tp = VecToMat(host_normal).getTranspose();
              mat1x3 jacobian_rotation = -host_normal_tp * Rwc * GetSkewMatrix(vertex);
              mat1x3 jacobian_translation = host_normal_tp;

              float weight = 1 / (0.02 * vertex_cam.z);

              mat1x6 jacobian;
              jacobian.setBlock(jacobian_rotation, 0, 0);
              jacobian.setBlock(jacobian_translation, 0, 3);

              H = jacobian.getTranspose() * weight * jacobian;
              b = -jacobian.getTranspose() * weight * residual;
              loss = weight * residual * residual;
            }
          }
        }
      }        
    }
  }
  
  // assert(loss >= 0);
  __syncthreads();
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  __shared__ double smemlf[32 * 32];
  __shared__ int smemi[32 * 32];
  int shift = 0;
  for(int i = 0; i < 6; i++) {
    for(int j = i; j < 6; j++) {
      __syncthreads();
      smemlf[tid] = H.entries2D[i][j];
      __syncthreads();

      ReduceSum<double, 32 * 32>(smemlf);
      __syncthreads();
      if(tid == 0) {
        block_data[grid_size * shift + gridDim.x * blockIdx.y + blockIdx.x] = smemlf[0];
        shift++;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    __syncthreads();
    smemlf[tid] = b.entries[i];
    __syncthreads();

    ReduceSum<double, 32 * 32>(smemlf);
    __syncthreads();
    if(tid == 0) {
      block_data[grid_size * shift + gridDim.x * blockIdx.y + blockIdx.x] = smemlf[0];
      shift++;
    }
  }


  __syncthreads();
  smemlf[tid] = loss;
  assert(smemlf[tid] >= 0);
  __syncthreads();

  ReduceSum<double, 32 * 32>(smemlf);
  __syncthreads();

  if(tid == 0) {
    block_data[grid_size * shift + gridDim.x * blockIdx.y + blockIdx.x] = smemlf[0];
    shift++;
  }
  __syncthreads();
  smemi[tid] = cnt;
  __syncthreads();

  ReduceSum<int, 32 * 32>(smemi);
  __syncthreads();
  if(tid == 0) {
    block_num[gridDim.x * blockIdx.y + blockIdx.x] = smemi[0];
    assert(block_num[gridDim.x * blockIdx.y + blockIdx.x] >= 0);
  }
} 

__global__ void ReductionKernel(double *block_data, double *sum_data, int *block_num, int *sum_num, int length)
{
  double sum = block_data[length * blockIdx.x + threadIdx.x];
  // int num = block_num[threadIdx.x];
  __shared__ double smemlf[MAX_BLOCK_SIZE];
  smemlf[threadIdx.x] = (threadIdx.x > length) ? 0 : sum;
  __syncthreads();

  ReduceSum<double, MAX_BLOCK_SIZE>(smemlf);

  if(threadIdx.x == 0)
    sum_data[blockIdx.x] = smemlf[0];
}

__global__ void ComputePointToVolumeKernel(Voxel *voxels, float3 *frame_vertex_map, float *Twc, unsigned char *mask, int3 bottom, float3 origin, int3 warp, int3 dim, float vs, 
                                           float *K, int *match_num, float depth_thres, int height, int width)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  // if(x % 4 != 3 || y % 4 != 3)
  //   return;

  int image_id = x + y * width;
  float3 vertex = frame_vertex_map[image_id];
  mask[image_id] = (unsigned char)0;

  float3 global_vertex;
  global_vertex.x = Twc[0] * vertex.x + Twc[1] * vertex.y + Twc[2] * vertex.z + Twc[3];
  global_vertex.y = Twc[4] * vertex.x + Twc[5] * vertex.y + Twc[6] * vertex.z + Twc[7];
  global_vertex.z = Twc[8] * vertex.x + Twc[9] * vertex.y + Twc[10] * vertex.z + Twc[11];

  int3 voxel_id;
  float tsdf;
  bool res = InterpolateTrilineary(voxels, voxel_id, tsdf, global_vertex, bottom.x, bottom.y, bottom.z, origin.x, origin.y, origin.z, warp.x, warp.y, warp.z, dim.x, dim.y, dim.z, vs);
  // if(fabs(res * vs) < depth_thres)
  //   mask[image_id] = (unsigned char)255;

  if(tsdf < 0.02)
    mask[image_id] = (unsigned char)255;
}

Volume::~Volume()
{
  gpuErrchk(cudaDeviceSynchronize());

  gpuErrchk(cudaFree(d_cam_K_in_));
  gpuErrchk(cudaFree(voxels));
  gpuErrchk(cudaFree(surface));
  gpuErrchk(cudaFree(virtual_vertex_map));
  gpuErrchk(cudaFree(virtual_normal_map));
  gpuErrchk(cudaFree(virtual_index_map));
  gpuErrchk(cudaFree(d_virtual_depth_));
}

__global__ void ReleaseSurfaceKernel(BGRPoint* surface, int size)
{
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if(id >= size) 
    return;
  surface[id].Set(0, 0, 0, make_uchar3(0, 0, 0), 0);
}

void Volume::Init(int _height, int _width, int _grid_dim_x, int _grid_dim_y, int _grid_dim_z, float _voxel_size, float _max_depth, float _min_depth, const float* cam_K)
{
  height = _height;
  width = _width;

  max_depth = _max_depth;
  min_depth = _min_depth;
  grid_dim_x = _grid_dim_x;
  grid_dim_y = _grid_dim_y;
  grid_dim_z = _grid_dim_z;
  voxel_size = _voxel_size;

  printf("[Volume::Init] Volume Params: \n");
  printf("Height: %d, Width: %d\n", height, width);
  printf("MaxDepth: %f, MinDepth: %f\n", max_depth, min_depth);
  printf("GridDimX: %d, GridDimY: %d, GridDimZ: %d\n", grid_dim_x, grid_dim_y, grid_dim_z);
  printf("VoxelSize: %f\n", voxel_size);

  assert(grid_dim_x%2 == 0);
  assert(grid_dim_y%2 == 0);
  assert(grid_dim_z%2 == 0);

  grid_origin_x = -(grid_dim_x/2) * voxel_size;
  grid_origin_y = -(grid_dim_y/2) * voxel_size;
  grid_origin_z = -(grid_dim_z/2) * voxel_size;

  cur_warp_x = 0;
  cur_warp_y = 0;
  cur_warp_z = 0;

  gpuErrchk(cudaMallocManaged((void **)&voxels, grid_dim_x * grid_dim_y * grid_dim_z * sizeof(Voxel)));
  gpuErrchk(cudaMallocManaged((void **)&surface, grid_dim_x * grid_dim_y * grid_dim_z * sizeof(BGRPoint)));

  // gpuErrchk(cudaMallocManaged((void **)&virtual_vertex_map, height * width * sizeof(float3)));
  // gpuErrchk(cudaMallocManaged((void **)&virtual_normal_map, height * width * sizeof(float3)));
  // gpuErrchk(cudaMallocManaged((void **)&virtual_index_map, height * width * sizeof(ushort)));
  // gpuErrchk(cudaMallocManaged((void **)&d_virtual_depth_, height * width * sizeof(float)));

  gpuErrchk(cudaMalloc((void **)&virtual_vertex_map, height * width * sizeof(float3)));
  gpuErrchk(cudaMalloc((void **)&virtual_normal_map, height * width * sizeof(float3)));
  gpuErrchk(cudaMalloc((void **)&virtual_index_map, height * width * sizeof(ushort)));
  gpuErrchk(cudaMalloc((void **)&d_virtual_depth_, height * width * sizeof(float)));

  // int num_pixels = height * width;
  gpuErrchk(cudaMalloc(&d_cam_K_in_, sizeof(float) * 9));
  gpuErrchk(cudaMemcpy(d_cam_K_in_, cam_K, sizeof(float) * 9, cudaMemcpyHostToDevice));
} 

void Volume::Reset()
{
  std::unique_lock<std::mutex> lock(volume_mutex);
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

}

void Volume::UpdateTSDF(const uchar3 *bgr, const float* depth, const float* Twc)
{
  int num_pixels = height * width;

  uchar3 *d_bgr_in; 
  float *d_depth_in, *d_Twc_in;
  gpuErrchk(cudaMalloc(&d_bgr_in, sizeof(uchar3) * num_pixels));
  gpuErrchk(cudaMalloc(&d_depth_in, sizeof(float) * num_pixels));
  gpuErrchk(cudaMalloc(&d_Twc_in, sizeof(float) * 16));

  gpuErrchk(cudaMemcpy(d_bgr_in, bgr, sizeof(uchar3) * num_pixels, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_depth_in, depth, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_Twc_in, Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));
  
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

  IntegrateKernel<<<grid_dim, block_dim>>>(voxels, d_bgr_in, d_depth_in, d_cam_K_in_, d_Twc_in, 
                                            grid_dim_x, grid_dim_y, grid_dim_z, 
                                            cur_warp_x, cur_warp_y, cur_warp_z,
                                            grid_origin_x, grid_origin_y, grid_origin_z, 
                                            voxel_size, height, width, 5 * voxel_size, max_depth, min_depth);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaFree(d_bgr_in));
  gpuErrchk(cudaFree(d_depth_in));
  gpuErrchk(cudaFree(d_Twc_in));

}

void Volume::UpdateSegment(const ushort *index, const float* depth, const float* Twc)
{
  int num_pixels = height * width;

  ushort *d_index_img;
  float *d_depth_img, *d_Twc;
  gpuErrchk(cudaMalloc((void**)&d_index_img, sizeof(ushort) * num_pixels));
  gpuErrchk(cudaMalloc((void**)&d_depth_img, sizeof(float) * num_pixels));
  gpuErrchk(cudaMalloc((void**)&d_Twc, sizeof(float) * 16));
  gpuErrchk(cudaMemcpy(d_index_img, index, sizeof(ushort) * num_pixels, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_depth_img, depth, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(d_Twc, Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(32, 32, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);
  UpdateSegmentKernel<<<grid_dim, block_dim>>>(voxels, d_index_img, d_depth_img, d_cam_K_in_, d_Twc, 
                                               grid_dim_x, grid_dim_y, grid_dim_z, cur_warp_x, cur_warp_y, cur_warp_z,
                                               grid_origin_x, grid_origin_y, grid_origin_z, voxel_size, height, width, 
                                               5 * voxel_size, max_depth, min_depth);
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaFree(d_depth_img));
  gpuErrchk(cudaFree(d_index_img));
  gpuErrchk(cudaFree(d_Twc));

}



void Volume::ReleaseXPlus(int dx)
{
  assert(dx > 0);
  int delta = abs(dx);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_z, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

  int bottom = cur_warp_x >= 0 ? cur_warp_x % grid_dim_x : grid_dim_x - ((-cur_warp_x) % grid_dim_x);

  double start = cv::getTickCount();
  ReleaseXKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
}

void Volume::ReleaseXMinus(int dx)
{
  assert(dx < 0);
  int delta = abs(dx);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_z, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

  int top = cur_warp_x >= 0 ? cur_warp_x % grid_dim_x : grid_dim_x - ((-cur_warp_x) % grid_dim_x);
  int bottom = top + dx;

  if(bottom < 0) {
    bottom = grid_dim_x + bottom;
  }

  double start = cv::getTickCount();
  ReleaseXKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
}

void Volume::ReleaseYPlus(int dy)
{
  assert(dy > 0);
  int delta = abs(dy);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_z, block_dim.y);

  int bottom = cur_warp_y >= 0 ? cur_warp_y % grid_dim_y : grid_dim_y - ((-cur_warp_y) % grid_dim_y);

  ReleaseYKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
  
}

void Volume::ReleaseYMinus(int dy)
{
  assert(dy < 0);
  int delta = abs(dy);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_z, block_dim.y);

  int top = cur_warp_y >= 0 ? cur_warp_y % grid_dim_y : grid_dim_y - ((-cur_warp_y) % grid_dim_y);
  int bottom = top + dy;

  if(bottom < 0) {
    bottom = grid_dim_y + bottom;
  }
  ReleaseYKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
  
}

void Volume::ReleaseZPlus(int dz)
{
  assert(dz > 0);
  int delta = abs(dz);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

  int bottom = cur_warp_z >= 0 ? cur_warp_z % grid_dim_z : grid_dim_z - ((-cur_warp_z) % grid_dim_z);

  ReleaseZKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
  
}

void Volume::ReleaseZMinus(int dz)
{
  assert(dz < 0);
  int delta = abs(dz);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(grid_dim_x, block_dim.x);
  grid_dim.y = DivUp(grid_dim_y, block_dim.y);

  int top = cur_warp_z >= 0 ? cur_warp_z % grid_dim_z : grid_dim_z - ((-cur_warp_z) % grid_dim_z);
  int bottom = top + dz;

  if(bottom < 0) {
    bottom = grid_dim_z + bottom;
  }
  ReleaseZKernel<<<grid_dim, block_dim>>>(voxels, bottom, delta, grid_dim_x, grid_dim_y, grid_dim_z);
  gpuErrchk(cudaDeviceSynchronize());
  
}


void Volume::Move(int vx, int vy, int vz)
{

  int dx = vx - cur_warp_x;
  int dy = vy - cur_warp_y;
  int dz = vz - cur_warp_z;

  if(dx > 0)
    ReleaseXPlus(dx);
  else if(dx < 0)
    ReleaseXMinus(dx);

  if(dy > 0)
    ReleaseYPlus(dy);
  else if(dy < 0)
    ReleaseYMinus(dy);

  if(dz > 0)
    ReleaseZPlus(dz);
  else if(dz < 0)
    ReleaseZMinus(dz);

  cur_warp_x = vx;
  cur_warp_y = vy;
  cur_warp_z = vz;
  gpuErrchk(cudaDeviceSynchronize());

}

void Volume::ExtractSurface()
{

  { 
    int grid_dim = DivUp(surface_num, 32);
    int block_dim = 32;
    ReleaseSurfaceKernel<<<grid_dim, block_dim>>>(surface, surface_num);
    surface_num = 0;
    match_num = 0;
  }

  {
    dim3 grid_dim(1, 1, 1);
    dim3 block_dim(16, 16, 1);
    grid_dim.x = DivUp(grid_dim_x, block_dim.x);
    grid_dim.y = DivUp(grid_dim_y, block_dim.y);

    int bottom_x = GetRemainder(cur_warp_x, grid_dim_x);
    int bottom_y = GetRemainder(cur_warp_y, grid_dim_y);
    int bottom_z = GetRemainder(cur_warp_z, grid_dim_z);

    int offset_x = cur_warp_x - bottom_x;
    int offset_y = cur_warp_y - bottom_y;
    int offset_z = cur_warp_z - bottom_z;

    ExtractSurfaceKernel<<<grid_dim, block_dim>>>(voxels, surface, &surface_num, voxel_size, 
                                                  grid_dim_x, grid_dim_y, grid_dim_z, 
                                                  bottom_x, bottom_y, bottom_z, offset_x, offset_y, offset_z,
                                                  grid_origin_x, grid_origin_y, grid_origin_z);
    gpuErrchk(cudaDeviceSynchronize());
  }
}

void Volume::RayCasting(const float* Twc, cv::Mat& depth, cv::Mat& normal, cv::Mat& vertex, cv::Mat& index)
{
  std::unique_lock<std::mutex> lock(render_mutex);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(16, 16, 1);
  grid_dim.x = DivUp(width, block_dim.x);
  grid_dim.y = DivUp(height, block_dim.y);

  float *d_Twc_in;
  gpuErrchk(cudaMalloc((void**)&d_Twc_in, sizeof(float) * 16));
  gpuErrchk(cudaMemcpy(d_Twc_in, Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));

  RayCastingKernel<<<grid_dim, block_dim>>>(voxels, d_cam_K_in_, d_virtual_depth_, virtual_vertex_map, virtual_normal_map, virtual_index_map, d_Twc_in, 
                                            grid_dim_x, grid_dim_y, grid_dim_z, cur_warp_x, cur_warp_y, cur_warp_z,
                                            grid_origin_x, grid_origin_y, grid_origin_z, voxel_size, height, width, 5 * voxel_size, 4.0f, min_depth);
  
  gpuErrchk(cudaDeviceSynchronize());
  gpuErrchk(cudaFree(d_Twc_in));
  render_ok = true;
}

void Volume::GetGroundPos(const float *Twc, unsigned short &plane_id, float &gx, float &gy, float& gz, float &nx, float &ny, float &nz)
{
  float3 global_vertex = make_float3(Twc[3], Twc[7], Twc[11]);

  float3 local_point;
  local_point.x = global_vertex.x - (grid_origin_x + cur_warp_x * voxel_size);
  local_point.y = global_vertex.y - (grid_origin_y + cur_warp_y * voxel_size);
  local_point.z = global_vertex.z - (grid_origin_z + cur_warp_z * voxel_size);

  int3 volume_id;
  volume_id.x = roundf(local_point.x / voxel_size);
  volume_id.y = roundf(local_point.y / voxel_size);
  volume_id.z = roundf(local_point.z / voxel_size);

  int bottom_x = GetRemainder(cur_warp_x, grid_dim_x);
  int bottom_y = GetRemainder(cur_warp_y, grid_dim_y);
  int bottom_z = GetRemainder(cur_warp_z, grid_dim_z);

  int x2d = GetRemainder(volume_id.x + bottom_x, grid_dim_x);
  int y2d = GetRemainder(volume_id.y + bottom_y, grid_dim_y);
  int z2d = GetRemainder(volume_id.z + bottom_z, grid_dim_z);

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(grid_dim_z, 1, 1);

  unsigned char *labels;
  unsigned short *ids;
  gpuErrchk(cudaMallocManaged((void**)&labels, sizeof(unsigned char) * grid_dim_z));
  gpuErrchk(cudaMallocManaged((void**)&ids, sizeof(unsigned short) * grid_dim_z));

  VerticalSearchKernel<<<grid_dim, block_dim>>>(voxels, labels, ids, Twc, x2d, y2d, grid_dim_x, grid_dim_y);
  gpuErrchk(cudaDeviceSynchronize());

  assert(labels[z2d] == 1);
  int z = z2d;
  while(labels[z] == 1 && z > 0) {
    z--;
  }
  if(z == 0) {
    plane_id = 0;
    gx = 0; gy = 0; gz = 0;
    gpuErrchk(cudaFree(labels));
    gpuErrchk(cudaFree(ids));
    return;
  }

  plane_id = ids[z];
  int id_down = GetID(x2d, y2d, z, grid_dim_x, grid_dim_y);
  int id_up = GetID(x2d, y2d, z+1, grid_dim_x, grid_dim_y);
  float offset = fabs(voxels[id_up].tsdf) / (fabs(voxels[id_up].tsdf) + fabs(voxels[id_down].tsdf));
  gx = Twc[3];
  gy = Twc[7];
  gz = Twc[11] - (z2d - z - 1 + offset) * voxel_size;


  // compute normal x
  float3 global_point = make_float3(gx, gy, gz), pt_point, mt_point;
  bool normal_result;
  float pt_tsdf, mt_tsdf;
  int3 voxel_id;

  pt_point = global_point + make_float3(voxel_size * 0.5, 0, 0);
  mt_point = global_point - make_float3(voxel_size * 0.5, 0, 0);

  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;
  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;

  nx = (pt_tsdf - mt_tsdf);

  // compute normal y
  pt_point = global_point + make_float3(0, voxel_size * 0.5, 0);
  mt_point = global_point - make_float3(0, voxel_size * 0.5, 0);

  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;
  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;

  ny = (pt_tsdf - mt_tsdf);

  // compute normal z
  pt_point = global_point + make_float3(0, 0, voxel_size * 0.5);
  mt_point = global_point - make_float3(0, 0, voxel_size * 0.5);

  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, pt_tsdf, pt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;
  normal_result = InterpolateTrilinearyHost(voxels, voxel_id, mt_tsdf, mt_point, bottom_x, bottom_y, bottom_z, 
                                grid_origin_x, grid_origin_y, grid_origin_z, cur_warp_x, cur_warp_y, cur_warp_z, grid_dim_x, grid_dim_y, grid_dim_z, voxel_size);
  // if(!normal_result)
  //   break;

  nz = (pt_tsdf - mt_tsdf);

  gpuErrchk(cudaFree(labels));
  gpuErrchk(cudaFree(ids));
}

// bool Volume::Register(const cv::Mat& cur_depth_map, Eigen::Matrix4d& Twc, const Eigen::Matrix4d& Twc_host)
// {

//   std::unique_lock<std::mutex> lock(render_mutex);
//   int num_pixels = height * width;

//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Tcw_host_r = Twc_host.inverse().cast<float>();

//   memcpy(h_cur_Twc, Twcr.data(), sizeof(float) * 16);

//   float* render_depth;
//   gpuErrchk(cudaMallocManaged((void **)&render_depth, sizeof(float) * num_pixels));

//   // std::cout << "origin Twc: " << std::endl << Twc << std::endl;
//   // grid block dim definition
//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(32, 32, 1);
//   grid_dim.x = DivUp(width, block_dim.x);
//   grid_dim.y = DivUp(height, block_dim.y);

//   // reduce sum variables
//   double *block_data;
//   double *sum_data;
//   int *block_num, *sum_num;
//   int grid_size = grid_dim.x * grid_dim.y;
//   gpuErrchk(cudaMallocManaged((void **)&block_data, sizeof(double) * grid_size * HBL_LENGTH));
//   gpuErrchk(cudaMallocManaged((void **)&sum_data, sizeof(double) * HBL_LENGTH));
//   gpuErrchk(cudaMallocManaged((void **)&block_num, sizeof(int) * grid_size));
//   gpuErrchk(cudaMallocManaged((void **)&sum_num, sizeof(int)));

//   gpuErrchk(cudaMemcpy(d_cur_Twc, h_cur_Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_host_Tcw, Tcw_host_r.data(), sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_depth_in_, cur_depth_map.data, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));

//   int match_num = 0;
//   bool is_converged = false;
//   Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Twc_step = Twc;
//   const int max_iter = 10;

//   ComputeVertexMapKernel<<<grid_dim, block_dim>>>(d_depth_in_, vertex_map, height, width, d_cam_K_in_, max_depth, min_depth);

//   for(int iter = 0; iter < max_iter; iter++) {
//     ResetMaskKernel<<<grid_dim, block_dim>>>(d_mask_, render_depth, height, width);
//     // gpuErrchk(cudaThreadSynchronize());
//     ResetBlockDataKernel<<<HBL_LENGTH, MAX_BLOCK_SIZE>>>(block_data, sum_data, grid_size);
//     EvaluateKernel<<<grid_dim, block_dim>>>(d_host_vertex_, d_host_normal_, vertex_map, d_cur_Twc, render_depth, d_host_Tcw, d_cam_K_in_, 
//                                               d_mask_, block_data, block_num, 0.02, height, width, grid_size, max_depth);
//     ReductionKernel<<<HBL_LENGTH, MAX_BLOCK_SIZE>>>(block_data, sum_data, block_num, sum_num, grid_size);
//     gpuErrchk(cudaDeviceSynchronize());

//     Eigen::Matrix<double, 6, 6> A;
//     Eigen::Matrix<double, 6, 1> b;
//     A << sum_data[0], sum_data[1], sum_data[2],  sum_data[3], sum_data[4], sum_data[5],
//         sum_data[1], sum_data[6], sum_data[7],  sum_data[8], sum_data[9], sum_data[10],
//         sum_data[2], sum_data[7], sum_data[11], sum_data[12], sum_data[13], sum_data[14],
//         sum_data[3], sum_data[8], sum_data[12],  sum_data[15], sum_data[16], sum_data[17],
//         sum_data[4], sum_data[9], sum_data[13],  sum_data[16], sum_data[18], sum_data[19],
//         sum_data[5], sum_data[10], sum_data[14],  sum_data[17], sum_data[19], sum_data[20];

//     b << sum_data[21], sum_data[22], sum_data[23], sum_data[24], sum_data[25], sum_data[26];
//     double loss = sum_data[27];

//     Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
//     double singular_ratio = svd.singularValues()(5) / svd.singularValues()(4);

//     if(singular_ratio < 0.15) {
//       DEBUG_PRINT("[ICP Step] singular_ratio: %lf, Degeneration occurs!\n", singular_ratio);
//       break;
//     }
      
//     std::cout << "eigenvalues: " << svd.singularValues().transpose() << std::endl;

//     auto x = A.ldlt().solve(b);
//     auto dr = x.head<3>();
//     auto dt = x.tail<3>();
//     Twc_step.block<3, 3>(0, 0) *= ExpSO3(dr);
//     Twc_step.block<3, 1>(0, 3) += dt;
//     gpuErrchk(cudaDeviceSynchronize());

//     DEBUG_PRINT("Iter: %d, Loss: %lf\n", iter, loss);
//     if(dr.norm() < 0.002 && dt.norm() < 0.002) {
//       DEBUG_PRINT("[ICP Step] dr: %lf, dt: %lf, Return!\n", dr.norm(), dt.norm());
//       is_converged = true;
//       break;
//     }

//     if(iter == max_iter - 1) {
//       is_converged = true;
//       break;
//     }

//     Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twc_step_f = Twc_step.cast<float>();
//     // memcpy(h_cur_Twc, Twcf.data(), sizeof(float) * 16);
//     gpuErrchk(cudaMemcpy(d_cur_Twc, Twc_step_f.data(), sizeof(float) * 16, cudaMemcpyHostToDevice));
//     gpuErrchk(cudaDeviceSynchronize());
//   }
//   gpuErrchk(cudaDeviceSynchronize());

//   Twc = Twc_step;
//   Twcr = Twc.cast<float>();
//   memcpy(h_cur_Twc, Twcr.data(), sizeof(float) * 16); // for visualization

//   return is_converged;
//   // cv::Mat mask_img(height, width, CV_8U);
//   // gpuErrchk(cudaMemcpy(mask_img.data, d_mask_, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost));

//   // cv::Mat render_depth_map(height, width, CV_32F);
//   // memcpy(render_depth_map.data, render_depth, sizeof(float) * height * width);
//   // render_depth_map.setTo(cv::Scalar(4.0), render_depth_map > 4.0);

//   // cv::Mat virtual_depth_8u;
//   // render_depth_map.convertTo(virtual_depth_8u, CV_8U, 255. / 4.0);
//   // cv::Mat virtual_depth;
//   // cv::applyColorMap(virtual_depth_8u, virtual_depth, cv::COLORMAP_JET);
//   // virtual_depth.setTo(cv::Scalar(0, 0, 0), virtual_depth_8u == 0);
//   // cv::imshow("virtual_depth", virtual_depth);
//   // cv::imshow("mask", mask_img);
//   // cv::waitKey(1);
//   // printf("[Register::Align] match_num: %d\n", match_num);
// }


// void Volume::Associate(const cv::Mat& cur_depth_map, const cv::Mat& host_normal_map, const Eigen::Matrix4d& Twc, const Eigen::Matrix4d& Twc_host)
// {
//   match_num = 0;
//   int num_pixels = height * width;
//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Tcw_host_r = Twc_host.inverse().cast<float>();

//   float* render_depth;
//   gpuErrchk(cudaMallocManaged((void **)&render_depth, sizeof(float) * num_pixels));

//   float3 *host_normal;
//   gpuErrchk(cudaMallocManaged((void **)&host_normal, sizeof(float3) * num_pixels));

//   memcpy(h_cur_Twc, Twcr.data(), sizeof(float) * 16);

//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(32, 32, 1);
//   grid_dim.x = DivUp(width, block_dim.x);
//   grid_dim.y = DivUp(height, block_dim.y);

//   gpuErrchk(cudaMemcpy(d_cur_Twc, h_cur_Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_host_Tcw, Tcw_host_r.data(), sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_depth_in_, cur_depth_map.data, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(host_normal, host_normal_map.data, sizeof(float3) * num_pixels, cudaMemcpyHostToDevice));

//   ComputeVertexMapKernel<<<grid_dim, block_dim>>>(d_depth_in_, vertex_map, height, width, d_cam_K_in_, max_depth, min_depth);
//   gpuErrchk(cudaDeviceSynchronize());

//   AssociateKernel<<<grid_dim, block_dim>>>(d_host_vertex_, d_host_normal_, vertex_map, d_cur_Twc, d_host_Tcw, d_cam_K_in_, render_depth,
//                                            d_mask_, residual_vertex, residual_host_vertex, residual_host_normal, residual_depth, &match_num,
//                                            0.05, height, width, DIVIDE, max_depth);
  
//   gpuErrchk(cudaDeviceSynchronize());
//   printf("[matchnum] %d\n", match_num);

//   // gpuErrchk(cudaFree(&host_normal));

//   cv::Mat mask_img(height, width, CV_8U);
//   gpuErrchk(cudaMemcpy(mask_img.data, d_mask_, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost));
//   cv::Mat render_depth_map(height, width, CV_32F);
//   memcpy(render_depth_map.data, render_depth, sizeof(float) * height * width);
//   render_depth_map.setTo(cv::Scalar(max_depth), render_depth_map > max_depth);

//   cv::Mat virtual_depth_8u;
//   render_depth_map.convertTo(virtual_depth_8u, CV_8U, 255. / max_depth);
//   cv::Mat virtual_depth;
//   cv::applyColorMap(virtual_depth_8u, virtual_depth, cv::COLORMAP_JET);
//   virtual_depth.setTo(cv::Scalar(0, 0, 0), virtual_depth_8u == 0);
//   cv::imshow("virtual_depth", virtual_depth);
//   cv::imshow("mask", mask_img);
//   cv::waitKey(1);

// }

// void Volume::AssociateColor(const cv::Mat& cur_depth_map, const cv::Mat& cur_seg_map, const Eigen::Matrix4d& Twc, const Eigen::Matrix4d& Twc_host)
// {
//   match_num = 0;
//   int num_pixels = height * width;
//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Twcr = Twc.cast<float>();
//   Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Tcw_host_r = Twc_host.inverse().cast<float>();

//   float* render_depth;
//   gpuErrchk(cudaMallocManaged((void **)&render_depth, sizeof(float) * num_pixels));

//   memcpy(h_cur_Twc, Twcr.data(), sizeof(float) * 16);

//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(32, 32, 1);
//   grid_dim.x = DivUp(width, block_dim.x);
//   grid_dim.y = DivUp(height, block_dim.y);

//   gpuErrchk(cudaMemcpy(d_cur_Twc, h_cur_Twc, sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_host_Tcw, Tcw_host_r.data(), sizeof(float) * 16, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(d_depth_in_, cur_depth_map.data, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));
//   gpuErrchk(cudaMemcpy(residual_seg, cur_seg_map.data, sizeof(float) * num_pixels, cudaMemcpyHostToDevice));

//   ComputeVertexMapKernel<<<grid_dim, block_dim>>>(d_depth_in_, vertex_map, height, width, d_cam_K_in_, max_depth, min_depth);
//   gpuErrchk(cudaDeviceSynchronize());


//   // AssociateColorKernel(d_host_vertex_, d_host_normal_, vertex_map, d_cur_Twc, d_host_Tcw, d_cam_K_in_, render_depth,
//   //                                    d_mask_, residual_vertex, residual_host_vertex, residual_seg, residual_depth, 
//   //                                    &match_num, 0.03, height, width, DIVIDE, max_depth);
  
//   gpuErrchk(cudaDeviceSynchronize());
//   printf("[matchnum] %d\n", match_num);


//   cv::Mat mask_img(height, width, CV_8U);
//   gpuErrchk(cudaMemcpy(mask_img.data, d_mask_, sizeof(unsigned char) * num_pixels, cudaMemcpyDeviceToHost));
//   cv::Mat render_depth_map(height, width, CV_32F);
//   memcpy(render_depth_map.data, render_depth, sizeof(float) * height * width);
//   render_depth_map.setTo(cv::Scalar(max_depth), render_depth_map > max_depth);

//   cv::Mat virtual_depth_8u;
//   render_depth_map.convertTo(virtual_depth_8u, CV_8U, 255. / max_depth);
//   cv::Mat virtual_depth;
//   cv::applyColorMap(virtual_depth_8u, virtual_depth, cv::COLORMAP_JET);
//   virtual_depth.setTo(cv::Scalar(0, 0, 0), virtual_depth_8u == 0);
//   cv::imshow("virtual_depth", virtual_depth);
//   cv::imshow("mask", mask_img);
//   cv::waitKey(1);
// }


// void Volume::RenderHostFrame(const float* Twc)
// {

// }

// void Volume::IntegrateAsync(const uchar3 *bgr, const float* depth, const float* Twc)
// {

// }