// CS 61C Fall 2015 Project 4

// include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <x86intrin.h>
#endif

// include OpenMP
#if !defined(_MSC_VER)
#include <pthread.h>
#endif
#include <omp.h>

//#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <math.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

#define MAX(a, b) ((a>b)? a:b)
#define MIN(a, b) ((a<b)? a:b)

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

int displacementOptimized(int dx, int dy)
{
	return dx * dx + dy * dy;
}

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	int unrollBound1 = 2 * featureWidth + 1;
	int unrollBound8 = (unrollBound1)/8 * 8;
	int unrollBound4 = (unrollBound1)/4 * 4;
	
	__m128 r0 = _mm_setzero_ps(); 

	int totalArea = imageWidth * imageHeight;
	int depthIndex = 0;
	for (; depthIndex < totalArea / 16 * 16; depthIndex += 16)
    {
    	_mm_storeu_ps(depth + depthIndex, r0);
    	_mm_storeu_ps(depth + depthIndex + 4, r0);
    	_mm_storeu_ps(depth + depthIndex + 8, r0);
    	_mm_storeu_ps(depth + depthIndex + 12, r0);
    }

    for (; depthIndex < totalArea; depthIndex++)
	{
		depth[depthIndex] = 0;
	}

	#pragma omp parallel for
	/* The two outer for loops iterate through each pixel */

	for (int y = featureHeight; y < imageHeight - featureHeight; y++)
	{
		for (int x = featureWidth; x < imageWidth - featureWidth; x++)
		{	
			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			int dyStart = MAX(featureHeight - y, -1 * maximumDisplacement);
			int dxStart = MAX(featureWidth - x, -1 * maximumDisplacement);
			int dyBound = MIN(imageHeight - featureHeight - y - 1, maximumDisplacement);
			int dxBound = MIN(imageWidth - featureWidth - x - 1, maximumDisplacement);

			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			for (int dy = dyStart; dy <= dyBound; dy++)
			{
				for (int dx = dxStart; dx <= dxBound; dx++)
				{
					float squaredDifference = 0;
					float S[4];
					__m128 sum = _mm_setzero_ps();

					// Unroll by 8
					// printf("\ndy = %d, dx = %d\n", dy,dx);
					// printf("\nUnroll by 8:\n");
					// for (int boxX = 0; boxX < unrollBound8; boxX+=8)
					// {
					// 	int leftCol = x + boxX - featureWidth;
					// 	int rightCol = leftCol + dx;
					// 	for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					// 	{
							
					// 		int leftRow = y + boxY;
					// 		int leftIndex = leftRow * imageWidth + leftCol; 
					// 		int rightIndex = (leftRow + dy) * imageWidth + rightCol;
			
					// 		__m128 v1 = _mm_loadu_ps(leftIndex + left); 
					// 		__m128 v2 = _mm_loadu_ps(rightIndex + right); 
					// 		v1 = _mm_sub_ps(v1, v2);
					// 		v1 = _mm_mul_ps(v1, v1);
					// 		sum = _mm_add_ps(v1, sum);

					// 		v1 = _mm_loadu_ps(leftIndex + 4  + left); 
					// 		v2 = _mm_loadu_ps(rightIndex + 4 + right); 
					// 		v1 = _mm_sub_ps(v1, v2);
					// 		v1 = _mm_mul_ps(v1, v1);
					// 		sum = _mm_add_ps(v1, sum);
					// 		printf("Left: (%d,%d) ", x + boxX - featureWidth, y + boxY);
					// 		printf("Right: (%d,%d)\n", leftCol + dx, leftRow + dy);
					// 	}
					// }

					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{	
						for (int boxX = 0; boxX < unrollBound8; boxX+=8)
						{
							int leftCol = x + boxX - featureWidth;
							int rightCol = leftCol + dx;
							int leftRow = y + boxY;
							int leftIndex = leftRow * imageWidth + leftCol; 
							int rightIndex = (leftRow + dy) * imageWidth + rightCol;
			
							__m128 v1 = _mm_loadu_ps(leftIndex + left); 
							__m128 v2 = _mm_loadu_ps(rightIndex + right); 
							v1 = _mm_sub_ps(v1, v2);
							v1 = _mm_mul_ps(v1, v1);
							sum = _mm_add_ps(v1, sum);

							v1 = _mm_loadu_ps(leftIndex + 4  + left); 
							v2 = _mm_loadu_ps(rightIndex + 4 + right); 
							v1 = _mm_sub_ps(v1, v2);
							v1 = _mm_mul_ps(v1, v1);
							sum = _mm_add_ps(v1, sum);
							// printf("Left: (%d,%d) ", x + boxX - featureWidth, y + boxY);
							// printf("Right: (%d,%d)\n", leftCol + dx, leftRow + dy);
						}
					}
					
					// Unroll by 4
					//printf("\nUnroll by 4:\n");
					for (int boxX = unrollBound8; boxX < unrollBound4; boxX+=4)
					{
						int leftCol = x + boxX - featureWidth;
						int rightCol = leftCol + dx;
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
						{ 
							__m128 v1 = _mm_loadu_ps(left + (y + boxY) * imageWidth + leftCol); 
							__m128 v2 = _mm_loadu_ps(right + (y + dy + boxY) * imageWidth + rightCol); 
							v1 = _mm_sub_ps(v1, v2);
							v1 = _mm_mul_ps(v1, v1);
							sum = _mm_add_ps(v1, sum);
						}
					}

					//Unrool by 1
					// printf("\nUnroll by 1:\n");
					for (int boxX = unrollBound4; boxX < unrollBound1; boxX++)
					{
						for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
						{
							int leftX = x + boxX - featureWidth;
							int leftY = y + boxY;
							float difference = left[leftY * imageWidth + leftX] - right[(leftY + dy) * imageWidth + leftX + dx];
							squaredDifference += difference * difference;
						}
					}

					_mm_store_ps(S, sum);
					squaredDifference += S[0] + S[1] + S[2] +S[3];

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || (minimumSquaredDifference > squaredDifference) || ((minimumSquaredDifference == squaredDifference) && (displacementOptimized(dx, dy) < displacementOptimized(minimumDx, minimumDy))))
					{
						minimumSquaredDifference = squaredDifference;
						minimumDx = dx;
						minimumDy = dy;
					}
				}
			}

			/* 
			Set the value in the depth map. 
			If max displacement is equal to 0, the depth value is just 0.
			*/
			if (minimumSquaredDifference != -1 && (maximumDisplacement != 0))
			{
				depth[y * imageWidth + x] = displacementNaive(minimumDx, minimumDy);
			}
		}
	}
}
