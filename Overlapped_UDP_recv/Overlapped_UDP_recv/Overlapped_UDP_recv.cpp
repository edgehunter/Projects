// Overlapped_UDP_recv.cpp : Defines the entry point for the console application.
//

#ifndef UNICODE
#define UNICODE
#endif

#define WIN32_LEAN_AND_MEAN

#include <winsock2.h>
#include <Ws2tcpip.h>
#include <stdio.h>

// Link with ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

int __cdecl main()
{

	WSADATA wsaData;
	WSABUF DataBuf;
	WSAOVERLAPPED Overlapped;

	SOCKET RecvSocket = INVALID_SOCKET;
	struct sockaddr_in RecvAddr;
	struct sockaddr_in SenderAddr;

	int SenderAddrSize = sizeof(SenderAddr);
	u_short Port = 27015;

	char RecvBuf[1024];
	int BufLen = 1024;
	DWORD BytesRecv = 0;
	DWORD Flags = 0;

	int err = 0;
	int rc;
	int retval = 0;

	//-----------------------------------------------
	// Initialize Winsock
	rc = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (rc != 0) {
		/* Could not find a usable Winsock DLL */
		wprintf(L"WSAStartup failed with error: %ld\n", rc);
		return 1;
	}

	// Make sure the Overlapped struct is zeroed out
	SecureZeroMemory((PVOID)&Overlapped, sizeof(WSAOVERLAPPED));

	// Create an event handle and setup the overlapped structure.
	Overlapped.hEvent = WSACreateEvent();
	if (Overlapped.hEvent == NULL) {
		wprintf(L"WSACreateEvent failed with error: %d\n", WSAGetLastError());
		WSACleanup();
		return 1;
	}
	//-----------------------------------------------
	// Create a receiver socket to receive datagrams
	RecvSocket = WSASocket(AF_INET,
		SOCK_DGRAM,
		IPPROTO_UDP, NULL, 0, WSA_FLAG_OVERLAPPED);

	if (RecvSocket == INVALID_SOCKET) {
		/* Could not open a socket */
		wprintf(L"WSASocket failed with error: %ld\n", WSAGetLastError());
		WSACloseEvent(Overlapped.hEvent);
		WSACleanup();
		return 1;
	}
	//-----------------------------------------------
	// Bind the socket to any address and the specified port.
	RecvAddr.sin_family = AF_INET;
	RecvAddr.sin_port = htons(Port);
	RecvAddr.sin_addr.s_addr = htonl(INADDR_ANY);

	rc = bind(RecvSocket, (SOCKADDR *)& RecvAddr, sizeof(RecvAddr));
	if (rc != 0) {
		/* Bind to the socket failed */
		wprintf(L"bind failed with error: %ld\n", WSAGetLastError());
		WSACloseEvent(Overlapped.hEvent);
		closesocket(RecvSocket);
		WSACleanup();
		return 1;
	}

	//-----------------------------------------------
	// Call the recvfrom function to receive datagrams
	// on the bound socket.
	DataBuf.len = BufLen;
	DataBuf.buf = RecvBuf;
	wprintf(L"Listening for incoming datagrams on port=%d\n", Port);
	rc = WSARecvFrom(RecvSocket,
		&DataBuf,
		1,
		&BytesRecv,
		&Flags,
		(SOCKADDR *)& SenderAddr,
		&SenderAddrSize, &Overlapped, NULL);

	if (rc != 0) {
		err = WSAGetLastError();
		if (err != WSA_IO_PENDING) {
			wprintf(L"WSARecvFrom failed with error: %ld\n", err);
			WSACloseEvent(Overlapped.hEvent);
			closesocket(RecvSocket);
			WSACleanup();
			return 1;
		}
		else {
			rc = WSAWaitForMultipleEvents(1, &Overlapped.hEvent, TRUE, INFINITE, TRUE);
			if (rc == WSA_WAIT_FAILED) {
				wprintf(L"WSAWaitForMultipleEvents failed with error: %d\n", WSAGetLastError());
				retval = 1;
			}

			rc = WSAGetOverlappedResult(RecvSocket, &Overlapped, &BytesRecv,
				FALSE, &Flags);
			if (rc == FALSE) {
				wprintf(L"WSArecvFrom failed with error: %d\n", WSAGetLastError());
				retval = 1;
			}
			else
				wprintf(L"Number of received bytes = %d\n", BytesRecv);

			wprintf(L"Finished receiving. Closing socket.\n");
		}

	}
	//---------------------------------------------
	// When the application is finished receiving, close the socket.

	WSACloseEvent(Overlapped.hEvent);
	closesocket(RecvSocket);
	wprintf(L"Exiting.\n");

	//---------------------------------------------
	// Clean up and quit.
	WSACleanup();
	return (retval);
}