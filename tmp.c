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

#include <math.h>
#include <stdbool.h>
#include <stdio.h>

#include "calcDepthOptimized.h"
#include "calcDepthNaive.h"

/* DO NOT CHANGE ANYTHING ABOVE THIS LINE. */

void calcDepthOptimized(float *depth, float *left, float *right, int imageWidth, int imageHeight, int featureWidth, int featureHeight, int maximumDisplacement)
{
	
	#pragma omp parallel for
	/* The two outer for loops iterate through each pixel */
	for (int y = 0; y < imageHeight; y++)
	{
		int yMulImageWidth = y * imageWidth;
		for (int x = 0; x < imageWidth; x++)
		{	

			/* Set the depth to 0 if looking at edge of the image where a feature box cannot fit. */
			if ((y < featureHeight) || (y >= imageHeight - featureHeight) || (x < featureWidth) || (x >= imageWidth - featureWidth))
			{
				depth[yMulImageWidth + x] = 0;
				continue;
			}

			float minimumSquaredDifference = -1;
			int minimumDy = 0;
			int minimumDx = 0;

			int dNDxDxDy = displacementNaive(minimumDx, minimumDy);

			int YMulImageWidthPlusX = y * imageWidth + x;
			/* Iterate through all feature boxes that fit inside the maximum displacement box. 
			   centered around the current pixel. */
			for (int dy = -maximumDisplacement; dy <= maximumDisplacement; dy++)
			{
				int yPlusDy = y + dy;
				
				for (int dx = -maximumDisplacement; dx <= maximumDisplacement; dx++)
				{
					int xPlusDx = x + dx;
					/* Skip feature boxes that dont fit in the displacement box. */
					if (yPlusDy - featureHeight < 0 || yPlusDy + featureHeight >= imageHeight || xPlusDx - featureWidth < 0 || xPlusDx + featureWidth >= imageWidth)
					{
						continue;
					}

					float squaredDifference = 0;

					/* Sum the squared difference within a box of +/- featureHeight and +/- featureWidth. */
					for (int boxY = -featureHeight; boxY <= featureHeight; boxY++)
					{
						// for (int boxX = -featureWidth; boxX <= featureWidth; boxX++)
						// {
						// 	int leftX = x + boxX;
						// 	int leftY = y + boxY;
						// 	int rightX = x + dx + boxX;
						// 	int rightY = y + dy + boxY;

						// 	float difference = left[leftY * imageWidth + leftX] - right[rightY * imageWidth + rightX];
						// 	squaredDifference += difference * difference;
						// }


						float S[4];
					    // load entire matrix into SSE vectors
						__m128 vSum = _mm_set_ps(0,0,0,0); 
						__m128 v1;
						__m128 v2;

						int boxXRange = featureWidth - 3;
						int boxX = -featureWidth;

						int leftYMulimageWidth = (y + boxY) * imageWidth;
						int rightYMulimageWidth = (y + boxY + dy) * imageWidth;


						for (; boxX <= boxXRange; boxX += 4)
					    {
					    	int leftX = x + boxX;

					    	v1 = _mm_loadu_ps(left + leftYMulimageWidth + leftX);
					    	v2 = _mm_loadu_ps(right + rightYMulimageWidth + leftX + dx);
					    	v2 = _mm_sub_ps(v1, v2);
					    	v2 = _mm_mul_ps(v2, v2);
							vSum = _mm_add_ps(vSum, v2);
					    }

						_mm_store_ps(S, vSum);
						squaredDifference += S[0] + S[1] + S[2] +S[3];

					    for (; boxX <= featureWidth; boxX++)
						{
							int leftX = x + boxX;
							float difference = left[leftYMulimageWidth + leftX] - right[rightYMulimageWidth + leftX + dx];
							squaredDifference += difference * difference;
						}

					}

					/* 
					Check if you need to update minimum square difference. 
					This is when either it has not been set yet, the current
					squared displacement is equal to the min and but the new
					displacement is less, or the current squared difference
					is less than the min square difference.
					*/
					if ((minimumSquaredDifference == -1) || ((minimumSquaredDifference == squaredDifference) && (displacementNaive(dx, dy) < dNDxDxDy)) || (minimumSquaredDifference > squaredDifference))
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
			if (minimumSquaredDifference != -1)
			{
				if (maximumDisplacement == 0)
				{
					depth[YMulImageWidthPlusX] = 0;
				}
				else
				{
					depth[YMulImageWidthPlusX] = displacementNaive(minimumDx, minimumDy);
				}
			}
			else
			{
				depth[YMulImageWidthPlusX] = 0;
			}
		}
	}
}
