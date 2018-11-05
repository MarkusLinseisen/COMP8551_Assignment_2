#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdlib.h>

#include "CL/cl.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

using namespace std;

int main() {
	// load image from file
	int width, height;
	unsigned int *input_image = (unsigned int*)stbi_load("./images/in.png", &width, &height, NULL, STBI_rgb_alpha);
	if (input_image == NULL) {
		cout << "image could not be read from file" << endl;
		return 1;
	}
	unsigned int* output_image = new unsigned int[width * height];
	cout << "read image from file" << endl;

	// load kernel from file
	ifstream kernel_file("kernel.cl");
	if (!kernel_file) {
		cout << "kernel could not be read from file" << endl;
		return 1;
	}
	string kernel_source(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	kernel_file.close();
	cout << "read kernel from file" << endl;

	// get device
	cl::Device device = cl::Device::getDefault();

	// create OpenCL context
	cl::Context context(device);
	cout << "created OpenCL context" << endl;

	// build program
	cl::Program program(context, kernel_source);
	program.build();
	cout << "built program" << endl;

	auto start = chrono::high_resolution_clock::now();

	// generate buffers
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, sizeof(unsigned int) * width * height);
	cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * width * height);
	cout << "generated buffers" << endl;

	// create command queue
	cl::CommandQueue queue(context, device);

	// fill buffer
	queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, sizeof(unsigned int) * width * height, input_image);

	// run kernel
	cl::Kernel kernel(program, "grayscale");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height));

	// get result
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(unsigned int) * width * height, output_image);

	auto end = chrono::high_resolution_clock::now();
	cout << "OpenCL grayscale took : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n";

	stbi_write_png("./images/out.png", width, height, STBI_rgb_alpha, output_image, STBI_rgb_alpha * width);
	stbi_image_free(input_image);

	cin.get();
	return 0;
}