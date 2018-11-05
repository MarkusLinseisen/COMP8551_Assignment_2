#define RED_COEFFICIENT 0.30f
#define GREEN_COEFFICIENT 0.59f
#define BLUE_COEFFICIENT 0.11f

inline unsigned char encode_gamma(float f) {
	return pow(f, 0.4545f) * 255;
}

inline float decode_gamma(unsigned char c) {
	return pow(c / 255.0f, 2.2f);
}

void kernel grayscale(global const unsigned int* A, global unsigned int* B) {
    size_t index = get_global_id(0);

	union Color {
		unsigned int u;
		unsigned char c[4];
	} color = {A[index]};

	float linear_luma = decode_gamma(color.c[0]) * RED_COEFFICIENT + decode_gamma(color.c[1]) * GREEN_COEFFICIENT + decode_gamma(color.c[2]) * BLUE_COEFFICIENT;
	unsigned char luma = encode_gamma(linear_luma);
	
	color.c[0] = luma;
	color.c[1] = luma;
	color.c[2] = luma;

	B[index] = color.u;
}