// IMGUI_OpenGL_GLFW_Fixed.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

//Device_Status
////////////////////////////////////////////////////////////////////////////////
#include "Device_Status\Device_Status.h"

//CUDA_VBO
////////////////////////////////////////////////////////////////////////////////

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics includes
#include <helper_gl.h>

// includes, cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
#include <timer.h>               // timing functions

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check
#include <helper_cuda_gl.h>      // helper functions for CUDA/GL interop

//IMGUI GLFW
////////////////////////////////////////////////////////////////////////////////
// ImGui - standalone example application for Glfw + OpenGL 2, using fixed pipeline
// If you are new to ImGui, see examples/README.txt and documentation at the top of imgui.cpp.

#include <imgui.h>
#include "imgui_impl_glfw.h"
#include <stdio.h>
#include <GLFW/glfw3.h>

#include <math.h>           // sqrtf, powf, cosf, sinf, floorf, ceilf
#define IM_ARRAYSIZE(_ARR)  ((int)(sizeof(_ARR)/sizeof(*_ARR)))



//CUDA_VBO
////////////////////////////////////////////////////////////////////////////////

// GL functionality
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags);
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void Render_CUDA_VBO();

// Cuda functionality
void runCuda(struct cudaGraphicsResource **vbo_resource);

//CUDA_VBO
////////////////////////////////////////////////////////////////////////////////

// constants
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;

// vbo variables
GLuint vbo;
struct cudaGraphicsResource *cuda_vbo_resource;
void *d_vbo_buffer = NULL;

float g_fAnim = 0.0;
#define MAX(a,b) ((a > b) ? a : b)


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void simple_vbo_kernel(float4 *pos, unsigned int width, unsigned int height, float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float)width;
	float v = y / (float)height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width + x] = make_float4(u, w, v, 1.0f);
}


void launch_kernel(float4 *pos, unsigned int mesh_width,
	unsigned int mesh_height, float time)
{
	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	simple_vbo_kernel << < grid, block >> >(pos, mesh_width, mesh_height, time);
}

bool checkHW(char *name, const char *gpuType, int dev)
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	strcpy(name, deviceProp.name);

	if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
	{
		return true;
	}
	else
	{
		return false;
	}
}

int findGraphicsGPU(char *name)
{
	int nGraphicsGPU = 0;
	int deviceCount = 0;
	bool bFoundGraphics = false;
	char firstGraphicsName[256], temp[256];

	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess)
	{
		printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
		printf("> FAILED sample finished, exiting...\n");
		exit(EXIT_FAILURE);
	}

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{
		printf("> There are no device(s) supporting CUDA\n");
		return false;
	}
	else
	{
		printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
	}

	for (int dev = 0; dev < deviceCount; ++dev)
	{
		bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
		printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

		if (bGraphics)
		{
			if (!bFoundGraphics)
			{
				strcpy(firstGraphicsName, temp);
			}

			nGraphicsGPU++;
		}
	}

	if (nGraphicsGPU)
	{
		strcpy(name, firstGraphicsName);
	}
	else
	{
		strcpy(name, "this hardware");
	}

	return nGraphicsGPU;
}


////////////////////////////////////////////////////////////////////////////////
//! Initialize CUDA_VBO
////////////////////////////////////////////////////////////////////////////////
bool init_CUDA_VBO()
{
	// initialize necessary OpenGL extensions
	if (!isGLVersionSupported(2, 0))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	char *name_GraphicsGPU = new char[100];
	int m_GraphicsGPU = findGraphicsGPU(name_GraphicsGPU);

	if (m_GraphicsGPU)
	{
		printf("Find GraphicsGPU %d \n", m_GraphicsGPU);
		printf("%s \n", name_GraphicsGPU);

		cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());

		// create VBO
		createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

		// run the cuda part
		runCuda(&cuda_vbo_resource);

		return true;

	}

	return false;
}


////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
	size_t num_bytes;
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,
		*vbo_resource));
	//printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

	// execute the kernel
	//    dim3 block(8, 8, 1);
	//    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	//    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

	launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

	// unmap buffer object
	checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif


////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBO(GLuint *vbo, struct cudaGraphicsResource **vbo_res,
	unsigned int vbo_res_flags)
{
	assert(vbo);

	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// register this buffer object with CUDA
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

	SDK_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBO(GLuint *vbo, struct cudaGraphicsResource *vbo_res)
{

	// unregister this buffer object with CUDA
	checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

	glBindBuffer(1, *vbo);
	glDeleteBuffers(1, vbo);

	*vbo = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Render_CUDA_VBO callback
////////////////////////////////////////////////////////////////////////////////
void Render_CUDA_VBO()
{
	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	// Disable the vbo
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}



//IMGUI glfw
////////////////////////////////////////////////////////////////////////////////

static void ShowHelpMarker(const char* desc)
{
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(450.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

static void error_callback(int error, const char* description)
{
	fprintf(stderr, "Error %d: %s\n", error, description);
}

//UTF-8到GB2312的转换
char* U2G(const char* utf8)
{
	int len = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
	wchar_t* wstr = new wchar_t[len + 1];
	memset(wstr, 0, len + 1);
	MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wstr, len);
	len = WideCharToMultiByte(CP_ACP, 0, wstr, -1, NULL, 0, NULL, NULL);
	char* str = new char[len + 1];
	memset(str, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wstr, -1, str, len, NULL, NULL);
	if (wstr) delete[] wstr;
	return str;
}

//GB2312到UTF-8的转换
char* G2U(char* gb2312)
{
	int len = MultiByteToWideChar(CP_ACP, 0, gb2312, -1, NULL, 0);
	wchar_t* wstr = new wchar_t[len + 1];
	memset(wstr, 0, len + 1);
	MultiByteToWideChar(CP_ACP, 0, gb2312, -1, wstr, len);
	len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
	char* str = new char[len + 1];
	memset(str, 0, len + 1);
	WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
	memcpy(gb2312, str, len + 1);

	if (wstr) delete[] wstr;
	if (str) delete[] str;

	return gb2312;
}

int main(int, char**)
{
	/*
	short m1;
	unsigned short m3;
	float m4;

	m1 = 0xf00f;
	m3 = 0xf00f;

	m1 = m3;
	m4 = m1;
	m4 = m3;
	*/


	// Setup window
	glfwSetErrorCallback(error_callback);
	if (!glfwInit())
		return 1;
	GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGui OpenGL2 example", NULL, NULL);
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1); // Enable vsync

	//Device_Status
	Disk_Struct* m_Disk_Struct;
	CDevice_Status* m_CDevice_Status = new CDevice_Status();

	// Init CUDA VBO
	if (init_CUDA_VBO())
	{
		printf("Init CUDA_VBO Success!");
	}

	// Setup ImGui binding
	ImGui_ImplGlfwGL2_Init(window, true);

	// Load Fonts
	// (there is a default font, this is only if you want to change it. see extra_fonts/README.txt for more details)
	ImGuiIO& io = ImGui::GetIO();
	//io.Fonts->AddFontDefault();
	io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\simkai.ttf", 15.0f, NULL, io.Fonts->GetGlyphRangesChinese());
	//io.Fonts->AddFontFromFileTTF("../../extra_fonts/DroidSans.ttf", 16.0f);
	//io.Fonts->AddFontFromFileTTF("../../extra_fonts/ProggyClean.ttf", 13.0f);
	//io.Fonts->AddFontFromFileTTF("../../extra_fonts/ProggyTiny.ttf", 10.0f);
	//io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf", 18.0f, NULL, io.Fonts->GetGlyphRangesJapanese());

	bool show_test_window = true;
	bool show_overlay_window = true;
	bool show_software_window = true;
	ImVec4 clear_color = ImColor(114, 144, 154);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplGlfwGL2_NewFrame();

		//4. lixiaoguang
		{
			//main frame
			{
				ImGui::Text("Welcome Sensor 3D");
				ImGui::ColorEdit3("clear color", (float*)&clear_color);
				if (ImGui::Button("show_overlay_window")) show_overlay_window ^= 1;
				if (ImGui::Button("show_software_window")) show_software_window ^= 1;
				if (ImGui::Button("show_test_window")) show_test_window ^= 1;
				//ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
			}

			//overlay fixed
			if (show_overlay_window)
			{
				ImGui::SetNextWindowPos(ImVec2(10, 10));
				ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(0.0f, 0.0f, 0.0f, 0.3f));
				if (!ImGui::Begin("Welcome Sensor 3D", &show_overlay_window, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings))
				{
					ImGui::End();
				}
				else
				{
					char* m_SystemTime = new char[m_CDevice_Status->Get_CharLength()];
					m_CDevice_Status->Get_SystemTime(m_SystemTime);
					ImGui::Text("%s\n", m_SystemTime);

					ImGui::Separator();
					ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
					ImGui::Text("Mouse Position: (%.1f,%.1f)", ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
					ImGui::End();

				}

				ImGui::PopStyleColor();
			}

			if (show_software_window)
			{
				ImGui::Begin("Software Window", &show_software_window, ImGuiWindowFlags_AlwaysAutoResize);
				ImGui::Text("Hello from Plots window!");
				
				char buf[256];

				//CPU 使用率
				///////////////////////////////////////////////////////////////////////

				//Device_Status
				sprintf_s(buf, 256, "CPU 使用率: %d\n", m_CDevice_Status->Get_Cpu_Usage());
				ImGui::Text(G2U(buf));

				// Create a dummy array of contiguous float values to plot
				// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
				static float values_CPU[256] = { 0 };
				static int values_offset_CPU = 0;

				values_CPU[values_offset_CPU] = m_CDevice_Status->Get_Cpu_Usage();
				values_offset_CPU = (values_offset_CPU + 1) % IM_ARRAYSIZE(values_CPU);
				ImGui::PlotLines("", values_CPU, IM_ARRAYSIZE(values_CPU), values_offset_CPU, NULL, 0.0f, 100.0f, ImVec2(200, 80));
				
				ImGui::Separator();

				//Memory 使用率
				///////////////////////////////////////////////////////////////////////

				//Device_Status
				sprintf_s(buf, 256, "Memory 使用率: %d\n", m_CDevice_Status->Get_Memory_Usage());
				ImGui::Text(G2U(buf));

				// Create a dummy array of contiguous float values to plot
				// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
				static float values_Memory[256] = { 0 };
				static int values_offset_Memory = 0;

				values_Memory[values_offset_Memory] = m_CDevice_Status->Get_Memory_Usage();
				values_offset_Memory = (values_offset_Memory + 1) % IM_ARRAYSIZE(values_Memory);
				ImGui::PlotHistogram("", values_Memory, IM_ARRAYSIZE(values_Memory), values_offset_Memory, NULL, 0.0f, 100.0f, ImVec2(200, 80));

				ImGui::Separator();

				//Disk 使用率
				///////////////////////////////////////////////////////////////////////
				
				//Device_Status
				int Disk_Count;
				m_Disk_Struct = m_CDevice_Status->Get_Disks_Usage(&Disk_Count);

				// Typically we would use ImVec2(-1.0f,0.0f) to use all available width, or ImVec2(width,0.0f) for a specified width. ImVec2(0.0f,0.0f) uses ItemWidth.
				for (size_t j = 0; j < Disk_Count; j++)
				{
					sprintf_s(buf, 256, "分区 %s 使用率:\n", m_Disk_Struct[j].Disk_Name);
					ImGui::Text(G2U(buf));
					ImGui::SameLine();
					ImGui::ProgressBar(m_Disk_Struct[j].Usage_Percent, ImVec2(100.0f, 20.0f));
				}
				
				//分区详细信息
				///////////////////////////////////////////////////////////////////////

				sprintf_s(buf, 256, "分区详细信息");
				if (ImGui::TreeNode(G2U(buf)))
				{

					for (size_t j = 0; j < Disk_Count; j++)
					{
						sprintf_s(buf, 256, "分区 ID: %s, 总量: %3d GB, 可用空间: %3d GB, 使用率: %.2f\n", m_Disk_Struct[j].Disk_Name, m_Disk_Struct[j].total_memory, m_Disk_Struct[j].available_memory, m_Disk_Struct[j].Usage_Percent * 100);
						ImGui::Text(G2U(buf));
					}

					ImGui::TreePop();
				}
				ImGui::Separator();


				///////////////////////////////////////////////////////////////////////
				
				ImGui::End();
			}
			

			//test window
			if (show_test_window)
			{
				//ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
				//ImGui::ShowTestWindow(&show_test_window);

				ImGui::Begin("show_test_window", &show_test_window);
				ImGui::Text("Hello show_test_window!");

				///////////////////////////////////////////////////////////////////////

				// Basic columns
				if (ImGui::TreeNode("Basic"))
				{
					//ImGui::Text("Without border:");
					//ImGui::Columns(3, "mycolumns3", false);  // 3-ways, no border
					//ImGui::Separator();
					//for (int n = 0; n < 14; n++)
					//{
					//	char label[32];
					//	sprintf(label, "Item %d", n);
					//	if (ImGui::Selectable(label)) {}
					//	//if (ImGui::Button(label, ImVec2(-1,0))) {}
					//	ImGui::NextColumn();
					//}
					//ImGui::Columns(1);
					//ImGui::Separator();

					ImGui::Text("With border:");
					ImGui::Columns(4, "mycolumns"); // 4-ways, with border
					ImGui::Separator();

					ImGui::Text("ID"); ImGui::NextColumn();
					ImGui::Text("Name"); ImGui::NextColumn();
					ImGui::Text("Path"); ImGui::NextColumn();
					ImGui::Text("Flags"); ImGui::NextColumn();
					ImGui::Separator();

					const char* names[3] = { "One", "Two", "Three" };
					const char* paths[3] = { "/path/one", "/path/two", "/path/three" };
					static int selected = -1;
					for (int i = 0; i < 3; i++)
					{
						char label[32];
						sprintf(label, "%04d", i);
						if (ImGui::Selectable(label, selected == i, ImGuiSelectableFlags_SpanAllColumns))
						{
							selected = i;
						}
							
						ImGui::NextColumn();
						ImGui::Text(names[i]); ImGui::NextColumn();
						ImGui::Text(paths[i]); ImGui::NextColumn();
						ImGui::Text("...."); ImGui::NextColumn();
					}
					ImGui::Columns(1);
					ImGui::Separator();
					ImGui::TreePop();
				}

				///////////////////////////////////////////////////////////////////////

				ImGui::End();
			}

		}


		// Rendering
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		glClear(GL_COLOR_BUFFER_BIT);

		// Render_CUDA_VBO
		g_fAnim += 0.01f;
		Render_CUDA_VBO();

		//glUseProgram(0); // You may want this if using this code in an OpenGL 3+ context where shaders may be bound
		
		ImGui::Render();
		glfwSwapBuffers(window);
	}

	// Cleanup
	ImGui_ImplGlfwGL2_Shutdown();
	glfwTerminate();

	return 0;
}


