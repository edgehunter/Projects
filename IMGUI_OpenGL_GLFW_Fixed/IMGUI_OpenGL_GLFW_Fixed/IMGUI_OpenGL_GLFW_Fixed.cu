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
	bool show_another_window = false;
	bool show_overlay_window = true;
	bool show_plots_window = true;
	bool show_software_window = true;
	ImVec4 clear_color = ImColor(114, 144, 154);

	// Main loop
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		ImGui_ImplGlfwGL2_NewFrame();

		/*

		// 1. Show a simple window
		// Tip: if we don't call ImGui::Begin()/ImGui::End() the widgets appears in a window automatically called "Debug"
		{
			static float f = 0.0f;
			ImGui::Text("Welcome Sensor 3D");
			//ImGui::SliderFloat("float", &f, 0.0f, 1.0f);
			ImGui::ColorEdit3("clear color", (float*)&clear_color);
			if (ImGui::Button("Test Window")) show_test_window ^= 1;
			if (ImGui::Button("Another Window")) show_another_window ^= 1;
			ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

		}

		// 2. Show another simple window, this time using an explicit Begin/End pair
		if (show_another_window)
		{
			ImGui::Begin("Another Window", &show_another_window);
			ImGui::Text("Hello from another window!");
			ImGui::End();
		}

		// 3. Show the ImGui test window. Most of the sample code is in ImGui::ShowTestWindow()
		if (show_test_window)
		{
			ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
			ImGui::ShowTestWindow(&show_test_window);
		}

		*/

		//4. lixiaoguang
		{
			//main frame
			{
				ImGui::Text("Welcome Sensor 3D");
				ImGui::ColorEdit3("clear color", (float*)&clear_color);
				if (ImGui::Button("show_overlay_window")) show_overlay_window ^= 1;
				if (ImGui::Button("show_software_window")) show_software_window ^= 1;
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

				///////////////////////////////////////////////////////////////////////

				//Device_Status
				char buf[256];
				sprintf_s(buf, 256, "CPU 使用率: %d\n", m_CDevice_Status->Get_Cpu_Usage());
				ImGui::Text(G2U(buf));

				//if (ImGui::TreeNode("CPU Usage"))
				{
					// Create a dummy array of contiguous float values to plot
					// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
					static float values_CPU[256] = { 0 };
					static int values_offset_CPU = 0;

					values_CPU[values_offset_CPU] = m_CDevice_Status->Get_Cpu_Usage();
					values_offset_CPU = (values_offset_CPU + 1) % IM_ARRAYSIZE(values_CPU);
					ImGui::PlotLines("", values_CPU, IM_ARRAYSIZE(values_CPU), values_offset_CPU, NULL, 0.0f, 100.0f, ImVec2(0, 80));
				}
				
				ImGui::Separator();
				///////////////////////////////////////////////////////////////////////

				sprintf_s(buf, 256, "Memory 使用率: %d\n", m_CDevice_Status->Get_Memory_Usage());
				ImGui::Text(G2U(buf));

				//if (ImGui::TreeNode("Memory Usage"))
				{
					// Create a dummy array of contiguous float values to plot
					// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
					static float values_Memory[256] = { 0 };
					static int values_offset_Memory = 0;

					values_Memory[values_offset_Memory] = m_CDevice_Status->Get_Memory_Usage();
					values_offset_Memory = (values_offset_Memory + 1) % IM_ARRAYSIZE(values_Memory);
					ImGui::PlotHistogram("", values_Memory, IM_ARRAYSIZE(values_Memory), values_offset_Memory, NULL, 0.0f, 100.0f, ImVec2(0, 80));
				}

				ImGui::Separator();
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
					ImGui::ProgressBar(m_Disk_Struct[j].Usage_Percent, ImVec2(0.0f, 0.0f));
				}


				sprintf_s(buf, 256, "分区详细信息");
				if (ImGui::TreeNode(G2U(buf)))
				{

					for (size_t j = 0; j < Disk_Count; j++)
					{
						sprintf_s(buf, 256, "分区 ID: %s, 总量: %3d GB, 可用空间: %3d GB, 使用率: %.2f\n", m_Disk_Struct[j].Disk_Name, m_Disk_Struct[j].total_memory, m_Disk_Struct[j].available_memory, m_Disk_Struct[j].Usage_Percent * 100);
						ImGui::Text(G2U(buf));
					}
				}


				ImGui::Separator();
				///////////////////////////////////////////////////////////////////////
				
				ImGui::End();
			}
			
			///////////////////////////////////////////////////////////////////////
			
			//plots window
			if (show_plots_window)
			{
				ImGui::Begin("Plots Window", &show_plots_window, ImGuiWindowFlags_AlwaysAutoResize);
				ImGui::Text("Hello from Plots window!");


				if (ImGui::TreeNode("Plots widgets"))
				{
					static bool animate = true;
					ImGui::Checkbox("Animate", &animate);

					static float arr[] = { 0.6f, 0.1f, 1.0f, 0.5f, 0.92f, 0.1f, 0.2f };
					ImGui::PlotLines("Frame Times", arr, IM_ARRAYSIZE(arr));

					// Create a dummy array of contiguous float values to plot
					// Tip: If your float aren't contiguous but part of a structure, you can pass a pointer to your first float and the sizeof() of your structure in the Stride parameter.
					static float values[90] = { 0 };
					static int values_offset = 0;
					static float refresh_time = 0.0f;
					if (!animate || refresh_time == 0.0f)
						refresh_time = ImGui::GetTime();
					while (refresh_time < ImGui::GetTime()) // Create dummy data at fixed 60 hz rate for the demo
					{
						static float phase = 0.0f;
						values[values_offset] = cosf(phase);
						values_offset = (values_offset + 1) % IM_ARRAYSIZE(values);
						phase += 0.10f*values_offset;
						refresh_time += 1.0f / 60.0f;
					}
					ImGui::PlotLines("Lines", values, IM_ARRAYSIZE(values), values_offset, "avg 0.0", -1.0f, 1.0f, ImVec2(0, 80));
					ImGui::PlotHistogram("Histogram", arr, IM_ARRAYSIZE(arr), 0, NULL, 0.0f, 1.0f, ImVec2(0, 80));

					// Use functions to generate output
					// FIXME: This is rather awkward because current plot API only pass in indices. We probably want an API passing floats and user provide sample rate/count.
					struct Funcs
					{
						static float Sin(void*, int i) { return sinf(i * 0.1f); }
						static float Saw(void*, int i) { return (i & 1) ? 1.0f : -1.0f; }
					};
					static int func_type = 0, display_count = 70;
					ImGui::Separator();
					ImGui::PushItemWidth(100); ImGui::Combo("func", &func_type, "Sin\0Saw\0"); ImGui::PopItemWidth();
					ImGui::SameLine();
					ImGui::SliderInt("Sample count", &display_count, 1, 400);
					float(*func)(void*, int) = (func_type == 0) ? Funcs::Sin : Funcs::Saw;
					ImGui::PlotLines("Lines", func, NULL, display_count, 0, NULL, -1.0f, 1.0f, ImVec2(0, 80));
					ImGui::PlotHistogram("Histogram", func, NULL, display_count, 0, NULL, -1.0f, 1.0f, ImVec2(0, 80));
					ImGui::Separator();

					// Animate a simple progress bar
					static float progress = 0.0f, progress_dir = 1.0f;
					if (animate)
					{
						progress += progress_dir * 0.4f * ImGui::GetIO().DeltaTime;
						if (progress >= +1.1f) { progress = +1.1f; progress_dir *= -1.0f; }
						if (progress <= -0.1f) { progress = -0.1f; progress_dir *= -1.0f; }
					}

					// Typically we would use ImVec2(-1.0f,0.0f) to use all available width, or ImVec2(width,0.0f) for a specified width. ImVec2(0.0f,0.0f) uses ItemWidth.
					ImGui::ProgressBar(progress, ImVec2(0.0f, 0.0f));
					ImGui::SameLine(0.0f, ImGui::GetStyle().ItemInnerSpacing.x);
					ImGui::Text("Progress Bar");

					float progress_saturated = (progress < 0.0f) ? 0.0f : (progress > 1.0f) ? 1.0f : progress;
					char buf[32];
					sprintf(buf, "%d/%d", (int)(progress_saturated * 1753), 1753);
					ImGui::ProgressBar(progress, ImVec2(0.f, 0.f), buf);
					ImGui::TreePop();
				}
				if (ImGui::TreeNode("Color/Picker Widgets"))
				{
					static ImVec4 color = ImColor(114, 144, 154, 200);

					static bool hdr = false;
					static bool alpha_preview = true;
					static bool alpha_half_preview = false;
					static bool options_menu = true;
					ImGui::Checkbox("With HDR", &hdr); ImGui::SameLine(); ShowHelpMarker("Currently all this does is to lift the 0..1 limits on dragging widgets.");
					ImGui::Checkbox("With Alpha Preview", &alpha_preview);
					ImGui::Checkbox("With Half Alpha Preview", &alpha_half_preview);
					ImGui::Checkbox("With Options Menu", &options_menu); ImGui::SameLine(); ShowHelpMarker("Right-click on the individual color widget to show options.");
					int misc_flags = (hdr ? ImGuiColorEditFlags_HDR : 0) | (alpha_half_preview ? ImGuiColorEditFlags_AlphaPreviewHalf : (alpha_preview ? ImGuiColorEditFlags_AlphaPreview : 0)) | (options_menu ? 0 : ImGuiColorEditFlags_NoOptions);

					ImGui::Text("Color widget:");
					ImGui::SameLine(); ShowHelpMarker("Click on the colored square to open a color picker.\nCTRL+click on individual component to input value.\n");
					ImGui::ColorEdit3("MyColor##1", (float*)&color, misc_flags);

					ImGui::Text("Color widget HSV with Alpha:");
					ImGui::ColorEdit4("MyColor##2", (float*)&color, ImGuiColorEditFlags_HSV | misc_flags);

					ImGui::Text("Color widget with Float Display:");
					ImGui::ColorEdit4("MyColor##2f", (float*)&color, ImGuiColorEditFlags_Float | misc_flags);

					ImGui::Text("Color button with Picker:");
					ImGui::SameLine(); ShowHelpMarker("With the ImGuiColorEditFlags_NoInputs flag you can hide all the slider/text inputs.\nWith the ImGuiColorEditFlags_NoLabel flag you can pass a non-empty label which will only be used for the tooltip and picker popup.");
					ImGui::ColorEdit4("MyColor##3", (float*)&color, ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel | misc_flags);

					ImGui::Text("Color button with Custom Picker Popup:");
					static bool saved_palette_inited = false;
					static ImVec4 saved_palette[32];
					static ImVec4 backup_color;
					if (!saved_palette_inited)
						for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
							ImGui::ColorConvertHSVtoRGB(n / 31.0f, 0.8f, 0.8f, saved_palette[n].x, saved_palette[n].y, saved_palette[n].z);
					bool open_popup = ImGui::ColorButton("MyColor##3b", color, misc_flags);
					ImGui::SameLine();
					open_popup |= ImGui::Button("Palette");
					if (open_popup)
					{
						ImGui::OpenPopup("mypicker");
						backup_color = color;
					}
					if (ImGui::BeginPopup("mypicker"))
					{
						// FIXME: Adding a drag and drop example here would be perfect!
						ImGui::Text("MY CUSTOM COLOR PICKER WITH AN AMAZING PALETTE!");
						ImGui::Separator();
						ImGui::ColorPicker4("##picker", (float*)&color, misc_flags | ImGuiColorEditFlags_NoSidePreview | ImGuiColorEditFlags_NoSmallPreview);
						ImGui::SameLine();
						ImGui::BeginGroup();
						ImGui::Text("Current");
						ImGui::ColorButton("##current", color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40));
						ImGui::Text("Previous");
						if (ImGui::ColorButton("##previous", backup_color, ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_AlphaPreviewHalf, ImVec2(60, 40)))
							color = backup_color;
						ImGui::Separator();
						ImGui::Text("Palette");
						for (int n = 0; n < IM_ARRAYSIZE(saved_palette); n++)
						{
							ImGui::PushID(n);
							if ((n % 8) != 0)
								ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.y);
							if (ImGui::ColorButton("##palette", saved_palette[n], ImGuiColorEditFlags_NoPicker | ImGuiColorEditFlags_NoTooltip, ImVec2(20, 20)))
								color = ImVec4(saved_palette[n].x, saved_palette[n].y, saved_palette[n].z, color.w); // Preserve alpha!
							ImGui::PopID();
						}
						ImGui::EndGroup();
						ImGui::EndPopup();
					}

					ImGui::Text("Color button only:");
					ImGui::ColorButton("MyColor##3b", *(ImVec4*)&color, misc_flags, ImVec2(80, 80));

					ImGui::Text("Color picker:");
					static bool alpha = true;
					static bool alpha_bar = true;
					static bool side_preview = true;
					static bool ref_color = false;
					static ImVec4 ref_color_v(1.0f, 0.0f, 1.0f, 0.5f);
					static int inputs_mode = 2;
					static int picker_mode = 0;
					ImGui::Checkbox("With Alpha", &alpha);
					ImGui::Checkbox("With Alpha Bar", &alpha_bar);
					ImGui::Checkbox("With Side Preview", &side_preview);
					if (side_preview)
					{
						ImGui::SameLine();
						ImGui::Checkbox("With Ref Color", &ref_color);
						if (ref_color)
						{
							ImGui::SameLine();
							ImGui::ColorEdit4("##RefColor", &ref_color_v.x, ImGuiColorEditFlags_NoInputs | misc_flags);
						}
					}
					ImGui::Combo("Inputs Mode", &inputs_mode, "All Inputs\0No Inputs\0RGB Input\0HSV Input\0HEX Input\0");
					ImGui::Combo("Picker Mode", &picker_mode, "Auto/Current\0Hue bar + SV rect\0Hue wheel + SV triangle\0");
					ImGui::SameLine(); ShowHelpMarker("User can right-click the picker to change mode.");
					ImGuiColorEditFlags flags = misc_flags;
					if (!alpha) flags |= ImGuiColorEditFlags_NoAlpha; // This is by default if you call ColorPicker3() instead of ColorPicker4()
					if (alpha_bar) flags |= ImGuiColorEditFlags_AlphaBar;
					if (!side_preview) flags |= ImGuiColorEditFlags_NoSidePreview;
					if (picker_mode == 1) flags |= ImGuiColorEditFlags_PickerHueBar;
					if (picker_mode == 2) flags |= ImGuiColorEditFlags_PickerHueWheel;
					if (inputs_mode == 1) flags |= ImGuiColorEditFlags_NoInputs;
					if (inputs_mode == 2) flags |= ImGuiColorEditFlags_RGB;
					if (inputs_mode == 3) flags |= ImGuiColorEditFlags_HSV;
					if (inputs_mode == 4) flags |= ImGuiColorEditFlags_HEX;
					ImGui::ColorPicker4("MyColor##4", (float*)&color, flags, ref_color ? &ref_color_v.x : NULL);

					ImGui::Text("Programmatically set defaults/options:");
					ImGui::SameLine(); ShowHelpMarker("SetColorEditOptions() is designed to allow you to set boot-time default.\nWe don't have Push/Pop functions because you can force options on a per-widget basis if needed, and the user can change non-forced ones with the options menu.\nWe don't have a getter to avoid encouraging you to persistently save values that aren't forward-compatible.");
					if (ImGui::Button("Uint8 + HSV"))
						ImGui::SetColorEditOptions(ImGuiColorEditFlags_Uint8 | ImGuiColorEditFlags_HSV);
					ImGui::SameLine();
					if (ImGui::Button("Float + HDR"))
						ImGui::SetColorEditOptions(ImGuiColorEditFlags_Float | ImGuiColorEditFlags_RGB);

					ImGui::TreePop();
				}

				ImGui::End();
			}

			///////////////////////////////////////////////////////////////////////

			//test window
			if (show_test_window)
			{
				ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiCond_FirstUseEver);
				ImGui::ShowTestWindow(&show_test_window);
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


