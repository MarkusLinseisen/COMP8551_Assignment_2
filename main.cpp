#include <chrono>
#include <fstream>
#include <iostream>
#include <stdlib.h>

#include "CL/cl.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "serially.h"

using namespace std;

void Serial(unsigned int * input_image, unsigned int * output_image, unsigned int width, unsigned int height) {
	auto start = chrono::high_resolution_clock::now();
	serially::grayscale(input_image, output_image, width, height);
	auto end = chrono::high_resolution_clock::now();
	cout << "Serial took : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";
}

void OpenCL_GPU(unsigned int * input_image, unsigned int * output_image, unsigned int width, unsigned int height) {
	// load kernel from file
	ifstream kernel_file("kernel.cl");
	if (!kernel_file) {
		cout << "kernel could not be read from file" << endl;
		return;
	}
	string kernel_source(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	kernel_file.close();

	// get cpu device with access to most memory
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Device device;
	uint64_t max = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		std::vector<cl::Device> devices;
		platforms[i].getDevices(CL_DEVICE_TYPE_GPU, &devices);
		for (size_t n = 0; n < devices.size(); n++) {
			auto memSize = devices[n].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			if (memSize > max) {
				max = memSize;
				device = devices[n];
			}
		}
	}
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

	// create OpenCL context
	cl::Context context(device);

	// build program
	cl::Program program(context, kernel_source);
	if (program.build() != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return;
	}

	auto start = chrono::high_resolution_clock::now();

	// generate buffers
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * width * height, input_image);
	cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * width * height);

	// create command queue
	cl::CommandQueue queue(context, device);

	// run kernel
	cl::Kernel kernel(program, "grayscale");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height));

	// get result
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(unsigned int) * width * height, output_image);

	auto end = chrono::high_resolution_clock::now();
	cout << "OpenCL GPU took : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";
}

void OpenCL_CPU(unsigned int * input_image, unsigned int * output_image, unsigned int width, unsigned int height) {
	// load kernel from file
	ifstream kernel_file("kernel.cl");
	if (!kernel_file) {
		cout << "kernel could not be read from file" << endl;
		return;
	}
	string kernel_source(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	kernel_file.close();

	// get gpu device with access to most memory
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Device device;
	uint64_t max = 0;
	for (size_t i = 0; i < platforms.size(); i++) {
		std::vector<cl::Device> devices;
		platforms[i].getDevices(CL_DEVICE_TYPE_CPU, &devices);
		for (size_t n = 0; n < devices.size(); n++) {
			auto memSize = devices[n].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
			if (memSize > max) {
				max = memSize;
				device = devices[n];
			}
		}
	}
	std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

	// create OpenCL context
	cl::Context context(device);

	// build program
	cl::Program program(context, kernel_source);
	if (program.build() != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
		return;
	}

	auto start = chrono::high_resolution_clock::now();

	// generate buffers
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * width * height, input_image);
	cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * width * height);

	// create command queue
	cl::CommandQueue queue(context, device);

	// run kernel
	cl::Kernel kernel(program, "grayscale");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	queue.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height));

	// get result
	queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(unsigned int) * width * height, output_image);

	auto end = chrono::high_resolution_clock::now();
	cout << "OpenCL CPU took : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";
}

void OpenCL_GPU_CPU(unsigned int * input_image, unsigned int * output_image, unsigned int width, unsigned int height) {
	// load kernel from file
	ifstream kernel_file("kernel.cl");
	if (!kernel_file) {
		cout << "kernel could not be read from file" << endl;
		return;
	}
	string kernel_source(istreambuf_iterator<char>(kernel_file), (istreambuf_iterator<char>()));
	kernel_file.close();

	// get devices from first platform that has 2 devices
	// presumably this will be a CPU and GPU pair except on devices utilizing CrossFire or SLI
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	std::vector<cl::Device> devices;
	for (size_t i = 0; i < platforms.size(); i++) {
		platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		if (devices.size() == 2) {
			break;
		}
	}
	std::cout << "Using device: " << devices[0].getInfo<CL_DEVICE_NAME>() << "\n";
	std::cout << "Using device: " << devices[1].getInfo<CL_DEVICE_NAME>() << "\n";

	// create OpenCL context
	cl::Context context(devices);

	// build program
	cl::Program program(context, kernel_source);
	if (program.build() != CL_SUCCESS) {
		for (size_t i = 0; i < devices.size(); i++) {
			std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[i]) << "\n";
		}
		return;
	}

	auto start = chrono::high_resolution_clock::now();

	// generate buffers
	cl::Buffer input_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned int) * width * height, input_image);
	cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(unsigned int) * width * height);

	// create command queue
	cl::CommandQueue queue_gpu(context, devices[0]);
	cl::CommandQueue queue_cpu(context, devices[1]);

	// run kernel
	cl::Kernel kernel(program, "grayscale");
	kernel.setArg(0, input_buffer);
	kernel.setArg(1, output_buffer);
	queue_gpu.enqueueNDRangeKernel(kernel, 0, cl::NDRange(width * height / 2));
	queue_cpu.enqueueNDRangeKernel(kernel, width * height / 2, cl::NDRange(width * height / 2));

	// get result
	queue_cpu.enqueueReadBuffer(output_buffer, CL_FALSE, sizeof(unsigned int) * width * height / 2, sizeof(unsigned int) * width * height / 2, output_image + (width * height / 2));
	queue_gpu.enqueueReadBuffer(output_buffer, CL_FALSE, 0, sizeof(unsigned int) * width * height / 2, output_image);

	auto end = chrono::high_resolution_clock::now();
	cout << "OpenCL GPU+CPU took : " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds\n\n";
}

int main() {
	// creat input "array"
	int width, height;
	unsigned int *input_image = (unsigned int*)stbi_load("./images/in.png", &width, &height, NULL, STBI_rgb_alpha);
	if (input_image == NULL) {
		cout << "image could not be read from file" << endl;
		return 1;
	}

	// grayscale serially
	unsigned int* output_image_serial = new unsigned int[width * height];
	Serial(input_image, output_image_serial, width, height);
	stbi_write_png("./images/out_Serial.png", width, height, STBI_rgb_alpha, output_image_serial, STBI_rgb_alpha * width);
	free(output_image_serial);

	// grayscale OpenCL CPU
	unsigned int* output_image_CL_CPU = new unsigned int[width * height];
	OpenCL_CPU(input_image, output_image_CL_CPU, width, height);
	stbi_write_png("./images/out_CL_CPU.png", width, height, STBI_rgb_alpha, output_image_CL_CPU, STBI_rgb_alpha * width);
	free(output_image_CL_CPU);

	// grayscale OpenCL GPU
	unsigned int* output_image_CL_GPU = new unsigned int[width * height];
	OpenCL_GPU(input_image, output_image_CL_GPU, width, height);
	stbi_write_png("./images/out_CL_GPU.png", width, height, STBI_rgb_alpha, output_image_CL_GPU, STBI_rgb_alpha * width);
	free(output_image_CL_GPU);

	// grayscale OpenCL GPU + CPU
	unsigned int* output_image_CL_GPU_CPU = new unsigned int[width * height];
	OpenCL_GPU_CPU(input_image, output_image_CL_GPU_CPU, width, height);
	stbi_write_png("./images/out_CL_GPU+CPU.png", width, height, STBI_rgb_alpha, output_image_CL_GPU_CPU, STBI_rgb_alpha * width);
	free(output_image_CL_GPU_CPU);

	stbi_image_free(input_image);

	cin.get();
	return 0;
}