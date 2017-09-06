// Status_CPU_Memory_Disk.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"


/*
#include <windows.h>
#include <stdio.h>
#pragma comment(lib, "user32.lib")
void main()
{
SYSTEM_INFO siSysInfo;
// Copy the hardware information to the SYSTEM_INFO structure.
GetSystemInfo(&siSysInfo);
// Display the contents of the SYSTEM_INFO structure.
printf("Hardware information: \n");
printf("  OEM ID: %u\n", siSysInfo.dwOemId);
printf("  Number of processors: %u\n",
siSysInfo.dwNumberOfProcessors);
printf("  Page size: %u\n", siSysInfo.dwPageSize);
printf("  Processor type: %u\n", siSysInfo.dwProcessorType);
printf("  Minimum application address: %lx\n",
siSysInfo.lpMinimumApplicationAddress);
printf("  Maximum application address: %lx\n",
siSysInfo.lpMaximumApplicationAddress);
printf("  Active processor mask: %u\n",
siSysInfo.dwActiveProcessorMask);
system("pause");
}
*/

/////////////////////////////////////////////////////////////////////////////

/*
//  Sample output:
//  There is       51 percent of memory in use.
//  There are 2029968 total KB of physical memory.
//  There are  987388 free  KB of physical memory.
//  There are 3884620 total KB of paging file.
//  There are 2799776 free  KB of paging file.
//  There are 2097024 total KB of virtual memory.
//  There are 2084876 free  KB of virtual memory.
//  There are       0 free  KB of extended memory.
#include <windows.h>
#include <stdio.h>
#include <tchar.h>
// Use to convert bytes to KB
#define DIV 1024
// Specify the width of the field in which to print the numbers.
// The asterisk in the format specifier "%*I64d" takes an integer
// argument and uses it to pad and right justify the number.
#define WIDTH 7
void main()
{
MEMORYSTATUSEX statex;
statex.dwLength = sizeof(statex);
GlobalMemoryStatusEx(&statex);
_tprintf(TEXT("There is  %*ld percent of memory in use.\n"),
WIDTH, statex.dwMemoryLoad);
_tprintf(TEXT("There are %*I64d total KB of physical memory.\n"),
WIDTH, statex.ullTotalPhys / DIV);
_tprintf(TEXT("There are %*I64d free  KB of physical memory.\n"),
WIDTH, statex.ullAvailPhys / DIV);
_tprintf(TEXT("There are %*I64d total KB of paging file.\n"),
WIDTH, statex.ullTotalPageFile / DIV);
_tprintf(TEXT("There are %*I64d free  KB of paging file.\n"),
WIDTH, statex.ullAvailPageFile / DIV);
_tprintf(TEXT("There are %*I64d total KB of virtual memory.\n"),
WIDTH, statex.ullTotalVirtual / DIV);
_tprintf(TEXT("There are %*I64d free  KB of virtual memory.\n"),
WIDTH, statex.ullAvailVirtual / DIV);
// Show the amount of extended memory available.
_tprintf(TEXT("There are %*I64d free  KB of extended memory.\n"),
WIDTH, statex.ullAvailExtendedVirtual / DIV);
system("pause");
}
*/

/////////////////////////////////////////////////////////////////////////////


/*
#define GB(x) (((x).HighPart << 2) + ((DWORD)(x).LowPart) / 1024.0 / 1024.0 / 1024.0)
//硬盘使用率 调用windows API
ULARGE_INTEGER FreeBytesAvailableC, TotalNumberOfBytesC, TotalNumberOfFreeBytesC;
ULARGE_INTEGER FreeBytesAvailableD, TotalNumberOfBytesD, TotalNumberOfFreeBytesD;
ULARGE_INTEGER FreeBytesAvailableE, TotalNumberOfBytesE, TotalNumberOfFreeBytesE;
ULARGE_INTEGER FreeBytesAvailableF, TotalNumberOfBytesF, TotalNumberOfFreeBytesF;
GetDiskFreeSpaceEx(_T("C:"), &FreeBytesAvailableC, &TotalNumberOfBytesC, &TotalNumberOfFreeBytesC);
GetDiskFreeSpaceEx(_T("D:"), &FreeBytesAvailableD, &TotalNumberOfBytesD, &TotalNumberOfFreeBytesD);
GetDiskFreeSpaceEx(_T("E:"), &FreeBytesAvailableE, &TotalNumberOfBytesE, &TotalNumberOfFreeBytesE);
GetDiskFreeSpaceEx(_T("F:"), &FreeBytesAvailableF, &TotalNumberOfBytesF, &TotalNumberOfFreeBytesF);
//参数 类型及说明
//lpRootPathName String，不包括卷名的磁盘根路径名
//lpFreeBytesAvailableToCaller LARGE_INTEGER，指定一个变量，用于容纳调用者可用的字节数量
//lpTotalNumberOfBytes LARGE_INTEGER，指定一个变量，用于容纳磁盘上的总字节数
//lpTotalNumberOfFreeBytes LARGE_INTEGER，指定一个变量，用于容纳磁盘上可用的字节数
//适用平台
//Windows 95 OSR2，Windows NT 4.0
float totalHardDisk = GB(TotalNumberOfBytesC) + GB(TotalNumberOfBytesD) + GB(TotalNumberOfBytesE);// +GB(TotalNumberOfBytesF);
float freeHardDisk = GB(TotalNumberOfFreeBytesC) + GB(TotalNumberOfFreeBytesD) + GB(TotalNumberOfFreeBytesE);// +GB(TotalNumberOfFreeBytesF);
float hardDiskUsage = 1 - freeHardDisk / totalHardDisk;
printf("硬盘总量：%f, 硬盘剩余：%f, 硬盘使用率：%f%%\n", totalHardDisk, freeHardDisk, hardDiskUsage);
*/

#include "Device_Status.h"

int main()
{
	Disk_Struct* m_Disk_Struct;
	CDevice_Status* m_CDevice_Status = new CDevice_Status();

	for (size_t i = 0; i < 100; i++)
	{

		printf("CPU使用率：%d%%\n", m_CDevice_Status->Get_Cpu_Usage());
		printf("Memory使用率：%d%%\n", m_CDevice_Status->Get_Memory_Usage());

		int Disk_Count;
		m_Disk_Struct = m_CDevice_Status->Get_Disks_Usage(&Disk_Count);

		for (size_t j = 0; j < Disk_Count; j++)
		{
			printf("硬盘名称：%s, 硬盘总量：%d, 硬盘可用：%d, 硬盘剩余：%d, 硬盘使用率：%f%%\n", m_Disk_Struct[j].Disk_Name, m_Disk_Struct[j].total_memory, m_Disk_Struct[j].available_memory, m_Disk_Struct[j].free_memory, m_Disk_Struct[j].Usage_Percent);
		}

		Sleep(500);
	}

	system("pause");

}

