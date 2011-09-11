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


//Calculte the Euclidian distance in a 3D space
/* Old version
__device__ float norm(float3 i, float3 j) {
  return sqrtf((i.x - j.x) * (i.x - j.x) + (i.y - j.y) * (i.y - j.y) + (i.z - j.z) * (i.z - j.z));
}
*/

//Calculte the Euclidian distance in a 3D space
__device__ inline float norm(float3 i, float3 j)
{
	return sqrtf(powf((i.x - j.x),2) + powf((i.y - j.y),2) + powf((i.z - j.z),2));
}

__device__ float3 springForce(const float3 Xi, const float3 Xj)
{
	// Spring stiffness, k
	//const float k = 0.005f;
	const float k = 0.008f;
	// Initial length of spring, l
	//const float l = 1.0f;
	const float l = 0.05f;

	float normXij = norm(Xi, Xj);

	return ( k * (l - normXij) / normXij ) * (Xi - Xj);;
}

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

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to implement numerical integration. EXTEND WITH MSD SYSTEM.
//! @param pos  vertex positiond in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void msd_kernel( float4 *_old_pos, float4 *_cur_pos, float4 *_new_pos, uint3 dims )
{
	// Index in position array
	const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	
	if( idx<prod(dims) ) {

		const uint3 co = idx_to_co( idx, dims );

	  // We assume border  particles are stuck
	  if ((co.x == 0 && co.y == 0) || (co.x == 0 && co.z == 0) || (co.y == 0 && co.z == 0) ||
	      (co.x == 0 && co.y == dims.y-1) || (co.x == 0 && co.z == dims.z-1) || (co.y == 0 && co.z == dims.z-1) ||
	      (co.x == dims.x-1 && co.y == 0) || (co.x == dims.x-1 && co.z == 0) || (co.y == dims.y-1 && co.z == 0) ||
	      (co.x == dims.x-1 && co.y == dims.y-1) || (co.x == dims.x-1 && co.z == dims.z-1) || (co.y == dims.y-1 && co.z == dims.z-1) ) {
		  _new_pos[idx] = _cur_pos[idx];

		} else {

		  // Time step size
		  const float dt = 0.1f;

		  /* Old version
		  // Spring stiffness, k
		  const float k = 0.01f;
		  // Initial length of spring, l
		  const float l = 1.0f;
		  */

		  // 3D coordinate of current particle. 
		  // Can be offset to access neighbors. E.g.: upIdx = co_to_idx(co-make_uint3(0,1,0), dims). <-- Be sure to take speciel care for border cases!
		  
		  //If we are on the border we take the force to ourselves, thus anialating the force to the particle that does not exist, during the calculation of accumulated force.
		  unsigned int upIdx; 
		  unsigned int downIdx;
		  unsigned int leftIdx;
		  unsigned int rightIdx;
		  unsigned int backIdx;
		  unsigned int forthIdx;

		  if (co.x == 0) {
		    rightIdx = idx;
		  } else {
		    rightIdx = co_to_idx( co - make_uint3(1,0,0), dims);
		  }
		  if (co.x == dims.x-1) {
		    leftIdx = idx;
		  } else {
		    leftIdx = co_to_idx( co + make_uint3(1,0,0), dims);
		  }
		  if (co.y == 0) {
		    upIdx = idx;
		  } else {
		    upIdx = co_to_idx( co - make_uint3(0,1,0), dims);
		  }
		  if (co.y == dims.y-1) {
		    downIdx = idx;
		  } else {
		    downIdx = co_to_idx( co + make_uint3(0,1,0), dims);
		  }
		  if (co.z == 0) {
		    backIdx = idx;
		  } else {
		    backIdx = co_to_idx( co - make_uint3(0,0,1), dims);
		  }
		  if (co.z == dims.z-1) {
		    forthIdx = idx;
		  } else {
		    forthIdx = co_to_idx( co + make_uint3(0,0,1), dims);
		  }

		  // Get the two previous positions
		  const float3 old_pos = crop_last_dim(_old_pos[idx]);
		  const float3 cur_pos = crop_last_dim(_cur_pos[idx]);

		  // Calculate force for each of the particles length 1 away. That is for the 6 nearest neighbors.
		  // fx is force on the x axis, that is the force of the particle to the right plus the force of the particle to the left

		  float3 fnew = springForce(cur_pos, crop_last_dim(_cur_pos[upIdx]))
				        + springForce(cur_pos, crop_last_dim(_cur_pos[leftIdx]))
				        + springForce(cur_pos, crop_last_dim(_cur_pos[rightIdx]))
				        //+ springForce(cur_pos, crop_last_dim(_cur_pos[backIdx])) // Doesn't work ?
				        //+ springForce(cur_pos, crop_last_dim(_cur_pos[forthIdx])) // Doesn't work ?
				        + springForce(cur_pos, crop_last_dim(_cur_pos[downIdx]));

		  /* Old version
		  float3 fnew = (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[rightIdx])))/ norm(cur_pos, crop_last_dim(_cur_pos[rightIdx]))) * (cur_pos - crop_last_dim(_cur_pos[rightIdx]) ) +
		    (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[leftIdx]))) / norm(cur_pos, crop_last_dim(_cur_pos[leftIdx])) ) * (cur_pos - crop_last_dim(_cur_pos[leftIdx])) +
		    (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[upIdx]))) / norm(cur_pos, crop_last_dim(_cur_pos[upIdx]))) * (cur_pos - crop_last_dim(_cur_pos[upIdx]) ) + 
		    (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[downIdx]))) / norm(cur_pos, crop_last_dim(_cur_pos[downIdx]))) * (cur_pos - crop_last_dim(_cur_pos[downIdx]) ) +
		    (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[forthIdx]))) / norm(cur_pos, crop_last_dim(_cur_pos[forthIdx]))) * (cur_pos - crop_last_dim(_cur_pos[forthIdx]) ) + 
		    (k * (l - norm(cur_pos, crop_last_dim(_cur_pos[backIdx]))) / norm(cur_pos, crop_last_dim(_cur_pos[backIdx])) ) * (cur_pos - crop_last_dim(_cur_pos[backIdx]) ) ;
          */

		  // Accelerate (constant gravity)
		  const float _a = -0.0008f;
		  const float3 a = make_float3( fnew.x, fnew.y + _a, fnew.z );
		  //const float3 a = make_float3( 0, _a, 0 );

		  // Integrate acceleration (forward Euler) to find velocity
		  const float3 cur_v = (cur_pos-old_pos)/dt;
		  const float3 new_v = cur_v + dt*a; // v'=a

		  // Integrate velocity (forward Euler) to find new particle position
		  float3 new_pos = cur_pos + dt*new_v; // pos'=v

		  // Implement a "floor"
		  if( new_pos.y < 0.0f)
		    new_pos.y =0.0f;

		  // Output
		  _new_pos[idx] = make_float4( new_pos.x, new_pos.y, new_pos.z, 1.0f );
		}
	}
}


#endif // #ifndef _SIMPLEGL_KERNEL_H_
