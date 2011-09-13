/*
	SimpleGL SDK example modified to prepare for a simple 
	Spring-mass-damper model implementation.
*/

/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

 /* This example demonstrates how to use the Cuda OpenGL bindings with the
  * runtime API.
  * Device code.
  */

#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_

// Cuda utilities
#include <cuda/uint_util.hcu>
#include <cuda/float_util.hcu>
#include <cuda/float_util_device.hcu>

__global__ void msd_initialize_kernel( float4 *dataPtr, float3 offset, uint3 dims )
{
	// Index in position array
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	if( idx<prod(dims) ){

		// 3D coordinate of current particle. 
		const uint3 co = idx_to_co( idx, dims );

		// Output
		float3 pos = uintd_to_floatd(co);
		pos /= uintd_to_floatd(dims);
		pos = pos*2.0f - make_float3(1.0f, 1.0f, 1.0f);
		pos += offset;

		dataPtr[idx] = make_float4( pos.x, pos.y, pos.z, 1.0f );
	}
}

//#define K 0.008f // Spring stiffness
//#define L 0.3f // Rest length (shouldnt be a global constant)

#define K 0.02f // Spring stiffness
#define L 0.001f // Rest length (shouldnt be a global constant)

__device__ float3 spring_force_rest_1(const float3 Xi, const float3 Xj){
  // Optimisation opportunity: Reuse parameters for data, ie. Xi for Xij and Xj.x for normXij
  float3 Xij = Xi - Xj;
  float normXij = sqrtf((Xij.x * Xij.x) +(Xij.y * Xij.y) + (Xij.z * Xij.z));

  return ((L - normXij) * K) * (Xij / normXij);
}

#define SQRT_2 sqrtf(2*L*L) // sqrt ((L)^2 +(L)^2) Replace with constant
__device__ float3 spring_force_rest_2(const float3 Xi, const float3 Xj){
  // Optimisation opportunity: Reuse parameters for data, ie. Xi for Xij and Xj.x for normXij
  float3 Xij = Xi - Xj;
  float normXij = sqrtf((Xij.x * Xij.x) +(Xij.y * Xij.y) + (Xij.z * Xij.z));

  return ((SQRT_2 - normXij) * K) * (Xij / normXij);
}

#define SQRT_3 sqrtf(3*L*L) // sqrt (L^2 +L^2+L^2) Replace with constant
__device__ float3 spring_force_rest_3(const float3 Xi, const float3 Xj){
  // Optimisation opportunity: Reuse parameters for data, ie. Xi for Xij and Xj.x for normXij
  float3 Xij = Xi - Xj;
  float normXij = sqrtf((Xij.x * Xij.x) +(Xij.y * Xij.y) + (Xij.z * Xij.z));

  return ((SQRT_3 - normXij) * K) * (Xij / normXij);
}

__device__ inline uint3 co_move(uint3 foo, const int x, const int y, const int z){
  foo.x += x;
  foo.y += y;
  foo.z += z;
  return foo;
}

#define ValueAtOffset(x,y,z) (crop_last_dim(_cur_pos[co_to_idx(co_move(co, x, y, z), dims)]))

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to implement numerical integration. EXTEND WITH MSD SYSTEM.
//! @param pos  vertex positiond in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void msd_kernel(const float4 *_old_pos, const float4 *_cur_pos, float4 *_new_pos, const uint3 dims, float _a, float dt)
{
	// Index in position array
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if ( idx<prod(dims) ){
      
      // 3D coordinate of current particle. 
      // Can be offset to access neighbors. E.g.: upIdx = co_to_idx(co-make_uint3(0,1,0), dims). <-- Be sure to take speciel care for border cases!
      const uint3 co = idx_to_co( idx, dims );

      if ((co.x == 0 && co.y == 0) || (co.x == 0 && co.z == 0) || (co.y == 0 && co.z == 0) ||
          (co.x == 0 && co.y == dims.y-1) || (co.x == 0 && co.z == dims.z-1) || (co.y == 0 && co.z == dims.z-1) ||
          (co.x == dims.x-1 && co.y == 0) || (co.x == dims.x-1 && co.z == 0) || (co.y == dims.y-1 && co.z == 0) ||
          (co.x == dims.x-1 && co.y == dims.y-1) || (co.x == dims.x-1 && co.z == dims.z-1) || (co.y == dims.y-1 && co.z == dims.z-1) ) {
        _new_pos[idx] = _cur_pos[idx];
      }
      	  else
      {

		// Time step size
  		//const float dt = 0.1f;
        //const float dt = 0.5f;

		// Get the two previous positions
		const float3 old_pos = crop_last_dim(_old_pos[idx]);
		const float3 cur_pos = crop_last_dim(_cur_pos[idx]);

		// Accelerate (constant gravity)
		//const float _a = -0.0008f;
		//const float _a = -0.0018f;
		float3 a = make_float3( 0.0f, _a, 0.0f );
        
#if 1
        // Sqrt(1) neighbors: There are 6 of these
        if (co.x > 0)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(-1,0,0));
        if (co.x < dims.x-1)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(1,0,0));
        if (co.y > 0)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(0,-1,0));
        if (co.y < dims.y-1)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(0,1,0));
        if (co.z > 0)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(0,0,-1));
        if (co.z < dims.z-1)
          a += spring_force_rest_1(cur_pos, ValueAtOffset(0,0,1));
#endif

#if 1
        // Sqrt(2) neighbors: There are 12 of these
        if (co.x > 0 && co.y > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(-1,-1,0));
        if (co.x > 0 && co.y < dims.y-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(-1, 1,0));
        if (co.x < dims.x-1 && co.y > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset( 1,-1,0));
        if (co.x < dims.x-1 && co.y < dims.y-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset( 1, 1,0));

        if (co.x > 0 && co.z > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(-1,0,-1));
        if (co.x > 0 && co.z < dims.z-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(-1,0, 1));
        if (co.x < dims.x-1 && co.z > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset( 1,0,-1));
        if (co.x < dims.x-1 && co.z < dims.z-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset( 1,0, 1));

        if (co.y > 0 && co.z > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(0,-1,-1));
        if (co.y > 0 && co.z < dims.z-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(0,-1, 1));
        if (co.y < dims.y-1 && co.z > 0)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(0, 1,-1));
        if (co.y < dims.y-1 && co.z < dims.z-1)
          a += spring_force_rest_2(cur_pos, ValueAtOffset(0, 1, 1));
#endif

#if 1
        // Sqrt(3) neighbors: There are 8 of these.
	if (co.x > 0 & co.y > 0 & co.z > 0)
          a += spring_force_rest_3(cur_pos, ValueAtOffset(-1,-1,-1));
	if (co.x > 0 & co.y > 0 & co.z < dims.z-1)
          a += spring_force_rest_3(cur_pos, ValueAtOffset(-1,-1, 1));
	if (co.x > 0 & co.y < dims.y-1 & co.z < dims.z-1)
          a += spring_force_rest_3(cur_pos, ValueAtOffset(-1, 1, 1));
	if (co.x < dims.x-1 & co.y < dims.y-1 & co.z < dims.z-1)
          a += spring_force_rest_3(cur_pos, ValueAtOffset( 1, 1, 1));

	if (co.x < dims.x-1 & co.y > 0 & co.z > 0)
          a += spring_force_rest_3(cur_pos, ValueAtOffset( 1,-1,-1));
	if (co.x < dims.x-1 & co.y < dims.y-1 & co.z > 0)
          a += spring_force_rest_3(cur_pos, ValueAtOffset( 1, 1,-1));

	if (co.x < dims.x-1 & co.y > 0 & co.z < dims.z-1)
          a += spring_force_rest_3(cur_pos, ValueAtOffset( 1,-1, 1));

	if (co.x > 0 & co.y < dims.y-1 & co.z > 0)
          a += spring_force_rest_3(cur_pos, ValueAtOffset(-1, 1,-1));
#endif

#if 0
          // Integrate acceleration (forward Euler) to find velocity
          const float3 cur_v = (cur_pos-old_pos)/dt;
          const float3 new_v = cur_v + dt*a; // v'=a
          
          // Integrate velocity (forward Euler) to find new particle position
          float3 new_pos = cur_pos + dt*(new_v); // pos'=v
#endif

#if 0
          // Verlet integration
          float3 new_pos = 2 * cur_pos - old_pos + a * dt * dt;
#endif

#if 1
          // Verlet integration with dampening
          #define DAMP 1.2f
          float3 new_pos = (2 - DAMP) * cur_pos - (1 - DAMP) * old_pos + a * dt * dt;
#endif

          if (new_pos.y < 0.0f)
        	  new_pos.y = 0.0f;

          // Output
          _new_pos[idx] = make_float4( new_pos.x, new_pos.y, new_pos.z, 1.0f );
      }
	}
}

#endif // #ifndef _SIMPLEGL_KERNEL_H_
