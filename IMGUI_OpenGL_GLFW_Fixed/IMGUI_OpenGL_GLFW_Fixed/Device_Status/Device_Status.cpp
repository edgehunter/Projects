#include "Device_Status.h"

CDevice_Status::CDevice_Status()
{
	GetSystemTimes(&m_preidleTime, &m_prekernelTime, &m_preuserTime);
	cpu_usage = 0;
	compute_Frequency = 0;

	statex.dwLength = sizeof(statex);
	memory_usage = 0;

	m_Disk_Struct = new Disk_Struct[DRIVE_MAX];
	disk_count = 0;

	GetLocalTime(&m_SystemTime);
	m_CharLength = 256;
	m_Char_SystemTime = new char[m_CharLength];
}


CDevice_Status::~CDevice_Status()
{
	delete m_Disk_Struct;
	delete m_Char_SystemTime;
}

//
///////////////////////////////////////////////////////////////////////

__int64 CDevice_Status::CompareFileTime(FILETIME time1, FILETIME time2)
{
	__int64 a = time1.dwHighDateTime << 32 | time1.dwLowDateTime;
	__int64 b = time2.dwHighDateTime << 32 | time2.dwLowDateTime;
	return   b - a;
}

void CDevice_Status::Compute_CpuUseage()
{
	GetSystemTimes(&idleTime, &kernelTime, &userTime);

	int idle = CompareFileTime(m_preidleTime, idleTime);
	int kernel = CompareFileTime(m_prekernelTime, kernelTime);
	int user = CompareFileTime(m_preuserTime, userTime);

	if (kernel + user == 0)
	{
		cpu_usage = 0;
	}
	else
	{
		//���ܵ�ʱ��-����ʱ�䣩/�ܵ�ʱ��=ռ��cpu��ʱ�����ʹ����
		cpu_usage = abs((kernel + user - idle) * 100 / (kernel + user));
	}

	m_preidleTime = idleTime;
	m_prekernelTime = kernelTime;
	m_preuserTime = userTime;
}

int CDevice_Status::Get_Cpu_Usage()
{
	if (++compute_Frequency % 100 == 0)
	{
		Compute_CpuUseage();
		compute_Frequency = 0;
	}
	
	return cpu_usage;
}

//
///////////////////////////////////////////////////////////////////////

void CDevice_Status::Compute_MemoryUseage()
{
	GlobalMemoryStatusEx(&statex);
	memory_usage = statex.dwMemoryLoad;
}

int CDevice_Status::Get_Memory_Usage()
{
	Compute_MemoryUseage();
	return memory_usage;
}

//
///////////////////////////////////////////////////////////////////////

void CDevice_Status::putDisksType(const char* lpRootPathName)
{
	UINT uDiskType = GetDriveTypeA(lpRootPathName);

	strcpy(m_Disk_Struct[disk_count].Disk_Name, lpRootPathName);
	m_Disk_Struct[disk_count].DiskType = uDiskType;

	/*
	switch (uDiskType)
	{
	case DRIVE_UNKNOWN:
	puts("δ֪�Ĵ�������");
	break;
	case DRIVE_NO_ROOT_DIR:
	puts("·����Ч");
	break;
	case DRIVE_REMOVABLE:
	puts("���ƶ�����");
	break;
	case DRIVE_FIXED:
	puts("�̶�����");
	break;
	case DRIVE_REMOTE:
	puts("�������");
	break;
	case DRIVE_CDROM:
	puts("����");
	break;
	case DRIVE_RAMDISK:
	puts("�ڴ�ӳ����");
	break;
	default:
	break;
	}
	*/

}

void CDevice_Status::putDisksFreeSpace(const char* lpRootPathName)
{

	if (GetDiskFreeSpaceExA(lpRootPathName, (ULARGE_INTEGER*)&available, (ULARGE_INTEGER*)&total, (ULARGE_INTEGER*)&free))
	{
		//printf("Drives %s | total = %lld GB,available = %lld GB,free = %lld GB\n",
		//	lpRootPathName, total >> 30, available >> 30, free >> 30);

		m_Disk_Struct[disk_count].total_memory = total >> 30;
		m_Disk_Struct[disk_count].available_memory = available >> 30;
		m_Disk_Struct[disk_count].free_memory = free >> 30;
		m_Disk_Struct[disk_count].Usage_Percent = (1 - (float)m_Disk_Struct[disk_count].available_memory / (float)m_Disk_Struct[disk_count].total_memory);
	}
	else
	{
		//puts("��ȡ������Ϣʧ��");
		m_Disk_Struct[disk_count].total_memory = 0;
		m_Disk_Struct[disk_count].available_memory = 0;
		m_Disk_Struct[disk_count].free_memory = 0;
		m_Disk_Struct[disk_count].Usage_Percent = 0;
	}
}

void CDevice_Status::Compute_Disks_Usage()
{
	disk_count = 0;

	DWORD dwSize = DRIVE_MAX;
	char szLogicalDrives[DRIVE_MAX] = { 0 };
	//��ȡ�߼����������ַ���
	DWORD dwResult = GetLogicalDriveStringsA(dwSize, szLogicalDrives);
	//�����ȡ���Ľ��
	if (dwResult > 0 && dwResult <= DRIVE_MAX)
	{
		char* szSingleDrive = szLogicalDrives;  //�ӻ�������ʼ��ַ��ʼ
		while (*szSingleDrive)
		{
			putDisksType(szSingleDrive);           //����߼�����������
			putDisksFreeSpace(szSingleDrive);

			// ��ȡ��һ������������ʼ��ַ
			szSingleDrive += strlen(szSingleDrive) + 1;
			disk_count++;
		}
	}
}

Disk_Struct* CDevice_Status::Get_Disks_Usage(int* Disk_Count)
{
	Compute_Disks_Usage();
	memcpy(Disk_Count, &disk_count, sizeof(int));

	return m_Disk_Struct;
}

//ʱ����Ϣ
////////////////////////////////////////////////////////////
void CDevice_Status::Compute_SystemTime()
{
	GetLocalTime(&m_SystemTime);
	sprintf_s(m_Char_SystemTime, m_CharLength, "System Time: %d-%02d-%02d %02d:%02d:%02d\n",
		m_SystemTime.wYear,
		m_SystemTime.wMonth,
		m_SystemTime.wDay,
		m_SystemTime.wHour,
		m_SystemTime.wMinute,
		m_SystemTime.wSecond);
	
}

void CDevice_Status::Get_SystemTime(char* Char_SystemTime)
{
	Compute_SystemTime();
	memcpy(Char_SystemTime, m_Char_SystemTime, m_CharLength);
}

int CDevice_Status::Get_CharLength()
{
	return m_CharLength;
}