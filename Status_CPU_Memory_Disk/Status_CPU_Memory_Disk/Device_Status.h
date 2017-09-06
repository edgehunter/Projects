#pragma once

#include <stdio.h>
#include <Windows.h>
#include <tchar.h>


#define DRIVE_MAX 40 

struct Disk_Struct
{
	char Disk_Name[10];
	UINT DiskType;
	unsigned int available_memory;//MB
	unsigned int total_memory;//MB
	unsigned int free_memory;//MB

	float Usage_Percent;//x%
};



class CDevice_Status
{
public:
	CDevice_Status();
	~CDevice_Status();


	int Get_Cpu_Usage();
	int Get_Memory_Usage();
	Disk_Struct* Get_Disks_Usage(int* Disk_Count);

private:

	//CPU利用率
	////////////////////////////////////////////////////////////
	FILETIME m_preidleTime;
	FILETIME m_prekernelTime;
	FILETIME m_preuserTime;

	FILETIME idleTime;
	FILETIME kernelTime;
	FILETIME userTime;

	int cpu_usage;

	//Memory使用率
	////////////////////////////////////////////////////////////
	MEMORYSTATUSEX statex;

	int memory_usage;

	//Disk使用率
	////////////////////////////////////////////////////////////
	Disk_Struct* m_Disk_Struct;
	int disk_count;
	unsigned long long available, total, free;

private:

	//CPU利用率
	////////////////////////////////////////////////////////////
	__int64 CompareFileTime(FILETIME time1, FILETIME time2);
	void Compute_CpuUseage();


	//Memory使用率
	////////////////////////////////////////////////////////////

	void  Compute_MemoryUseage();

	//Disk使用率
	////////////////////////////////////////////////////////////

	void CDevice_Status::putDisksType(const char* lpRootPathName);
	void CDevice_Status::putDisksFreeSpace(const char* lpRootPathName);
	void CDevice_Status::Compute_Disks_Usage();

};