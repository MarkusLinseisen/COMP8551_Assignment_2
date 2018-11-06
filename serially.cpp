#include "serially.h"

const float serially::RED_COEFFICIENT = 0.2126f;
const float serially::GREEN_COEFFICIENT = 0.7152f;
const float serially::BLUE_COEFFICIENT = 0.0722f;

inline float serially::charToFloat(unsigned char c) {
	return c / 255.0f;
}

inline unsigned char serially::floatToChar(float f) {
	return round(f * 255.0f);
}

inline float serially::encode_sRGB(float f) {
	if (f <= 0.0031308f) {
		return 12.92f * f;
	} else {
		return 1.055f * pow(f, 1.0f / 2.4f) - 0.055;
	}
}

inline float serially::decode_sRGB(float f) {
	if (f <= 0.04045f) {
		return f / 12.92f;
	} else {
		return pow((f + 0.055f) / 1.055f, 2.4f);
	}
}

void serially::grayscale(const unsigned int * A, unsigned int * B, size_t width, size_t height) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			size_t index = x + y * width;

			union Color {
				unsigned int u;
				unsigned char c[4];
			} color = { A[index] };

			float linear_luma = decode_sRGB(charToFloat(color.c[0])) * RED_COEFFICIENT + decode_sRGB(charToFloat(color.c[1])) * GREEN_COEFFICIENT + decode_sRGB(charToFloat(color.c[2])) * BLUE_COEFFICIENT;
			unsigned char luma = floatToChar(encode_sRGB(linear_luma));

			color.c[0] = luma;
			color.c[1] = luma;
			color.c[2] = luma;

			B[index] = color.u;
		}
	}
}
