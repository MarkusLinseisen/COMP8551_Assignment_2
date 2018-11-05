#define RED_COEFFICIENT 0.2126f
#define GREEN_COEFFICIENT 0.7152f
#define BLUE_COEFFICIENT 0.0722f

inline float charToFloat(unsigned char c) {
	return c / 255.0f;
}

inline unsigned char floatToChar(float f) {
	return round(f * 255.0f);
}

inline float encode_sRGB(float f) {
	if (f <= 0.0031308f) {
		return 12.92f * f;
	} else {
		return 1.055f * pow(f, 1.0f / 2.4f) - 0.055;
	}
}

inline float decode_sRGB(float f) {
	if (f <= 0.04045f) {
		return f / 12.92f;
	} else {
		return pow((f + 0.055f) / 1.055f, 2.4f);
	}
}

void kernel grayscale(global const unsigned int* A, global unsigned int* B) {
    size_t index = get_global_id(0);

	union Color {
		unsigned int u;
		unsigned char c[4];
	} color = {A[index]};

	float linear_luma = decode_sRGB(charToFloat(color.c[0])) * RED_COEFFICIENT + decode_sRGB(charToFloat(color.c[1])) * GREEN_COEFFICIENT + decode_sRGB(charToFloat(color.c[2])) * BLUE_COEFFICIENT;
	unsigned char luma = floatToChar(encode_sRGB(linear_luma));
	
	color.c[0] = luma;
	color.c[1] = luma;
	color.c[2] = luma;

	B[index] = color.u;
}