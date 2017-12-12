// Overlapped_UDP_send.cpp : Defines the entry point for the console application.
//

#define WIN32_LEAN_AND_MEAN

#include <winsock2.h>
#include <Ws2tcpip.h>
#include <stdio.h>
#include <stdlib.h>

// Link with ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

int __cdecl main(int argc, char **argv)
{

	//---------------------------------------------
	// Declare and initialize variables
	WSADATA wsaData;
	WSABUF DataBuf;

	WSAOVERLAPPED Overlapped;
	SOCKET SendToSocket = INVALID_SOCKET;

	struct sockaddr_in RecvAddr;
	struct sockaddr_in LocalAddr;
	int RecvAddrSize = sizeof(RecvAddr);
	int LocalAddrSize = sizeof(LocalAddr);

	u_short Port = 27777;
	struct hostent *localHost;
	char *ip;

	char *targetip;
	char *targetport;

	char SendBuf[1024] = "Data buffer to send";
	int BufLen = 1024;
	DWORD BytesSent = 0;
	DWORD Flags = 0;

	int rc, err;
	int retval = 0;

	// Validate the parameters
	/*
	if (argc != 3) {
		printf("usage: %s targetip port\n", argv[0]);
		printf("  to sendto the localhost on port 27777\n");
		printf("       %s 127.0.0.1 27777\n", argv[0]);
		return 1;
	}
	*/

	targetip = "192.168.1.211"; argv[1];
	targetport = "27015"; argv[2];

	//---------------------------------------------
	// Initialize Winsock
	// Load Winsock
	rc = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (rc != 0) {
		printf("Unable to load Winsock: %d\n", rc);
		return 1;
	}

	// Make sure the Overlapped struct is zeroed out
	SecureZeroMemory((PVOID)&Overlapped, sizeof(WSAOVERLAPPED));

	// Create an event handle and setup the overlapped structure.
	Overlapped.hEvent = WSACreateEvent();
	if (Overlapped.hEvent == WSA_INVALID_EVENT) {
		printf("WSACreateEvent failed with error: %d\n", WSAGetLastError());
		WSACleanup();
		return 1;
	}
	//---------------------------------------------
	// Create a socket for sending data
	SendToSocket =
		WSASocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP, NULL, 0,
		WSA_FLAG_OVERLAPPED);
	if (SendToSocket == INVALID_SOCKET) {
		printf("socket failed with error: %d\n", WSAGetLastError());
		WSACloseEvent(Overlapped.hEvent);
		WSACleanup();
		return 1;
	}
	//---------------------------------------------
	// Set up the RecvAddr structure with the IP address of
	// the receiver (in this example case "123.123.123.1")
	// and the specified port number.
	RecvAddr.sin_family = AF_INET;

	RecvAddr.sin_addr.s_addr = inet_addr(targetip);
	if (RecvAddr.sin_addr.s_addr == INADDR_NONE)  {
		printf("The target ip address entered must be a legal IPv4 address\n");
		WSACloseEvent(Overlapped.hEvent);
		WSACleanup();
		return 1;
	}
	RecvAddr.sin_port = htons((u_short)atoi(targetport));
	if (RecvAddr.sin_port == 0) {
		printf("The targetport must be a legal UDP port number\n");
		WSACloseEvent(Overlapped.hEvent);
		WSACleanup();
		return 1;
	}

	//---------------------------------------------
	// Set up the LocalAddr structure with the local IP address
	// and the specified port number.
	localHost = gethostbyname("");
	ip = inet_ntoa(*(struct in_addr *) *localHost->h_addr_list);

	LocalAddr.sin_family = AF_INET;
	LocalAddr.sin_addr.s_addr = inet_addr(ip);
	LocalAddr.sin_port = htons(Port);

	//---------------------------------------------
	// Bind the sending socket to the LocalAddr structure
	// that has the internet address family, local IP address
	// and specified port number.  
	rc = bind(SendToSocket, (struct sockaddr *) &LocalAddr, LocalAddrSize);
	if (rc == SOCKET_ERROR) {
		printf("bind failed with error: %d\n", WSAGetLastError());
		WSACloseEvent(Overlapped.hEvent);
		closesocket(SendToSocket);
		WSACleanup();
		return 1;
	}
	//---------------------------------------------
	// Send a datagram to the receiver
	printf("Sending datagram from IPv4 address = %s port=%d\n",
		inet_ntoa(LocalAddr.sin_addr), ntohs(LocalAddr.sin_port));
	printf("   to IPv4 address = %s port=%d\n",
		inet_ntoa(RecvAddr.sin_addr), ntohs(RecvAddr.sin_port));

	//    printf("Sending a datagram...\n");
	DataBuf.len = BufLen;
	DataBuf.buf = SendBuf;
	rc = WSASendTo(SendToSocket, &DataBuf, 1,
		&BytesSent, Flags, (SOCKADDR *)& RecvAddr,
		RecvAddrSize, &Overlapped, NULL);

	if ((rc == SOCKET_ERROR) && (WSA_IO_PENDING != (err = WSAGetLastError()))) {
		printf("WSASendTo failed with error: %d\n", err);
		WSACloseEvent(Overlapped.hEvent);
		closesocket(SendToSocket);
		WSACleanup();
		return 1;
	}

	rc = WSAWaitForMultipleEvents(1, &Overlapped.hEvent, TRUE, INFINITE, TRUE);
	if (rc == WSA_WAIT_FAILED) {
		printf("WSAWaitForMultipleEvents failed with error: %d\n",
			WSAGetLastError());
		retval = 1;
	}

	rc = WSAGetOverlappedResult(SendToSocket, &Overlapped, &BytesSent,
		FALSE, &Flags);
	if (rc == FALSE) {
		printf("WSASendTo failed with error: %d\n", WSAGetLastError());
		retval = 1;
	}
	else
		printf("Number of sent bytes = %d\n", BytesSent);

	//---------------------------------------------
	// When the application is finished sending, close the socket.
	printf("Finished sending. Closing socket.\n");
	WSACloseEvent(Overlapped.hEvent);
	closesocket(SendToSocket);
	printf("Exiting.\n");

	//---------------------------------------------
	// Clean up and quit.
	WSACleanup();
	return (retval);
}