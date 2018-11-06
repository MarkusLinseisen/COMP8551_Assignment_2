#pragma once

#include <math.h>

class serially {
	public:
		static void grayscale(const unsigned int* A, unsigned int* B, size_t width, size_t height);
	private:
		static float charToFloat(unsigned char c);
		static unsigned char floatToChar(float f);
		static float encode_sRGB(float f);
		static float decode_sRGB(float f);
		static const float RED_COEFFICIENT;
		static const float GREEN_COEFFICIENT;
		static const float BLUE_COEFFICIENT;
};