// Overlapped_UDP_recv_send.cpp : Defines the entry point for the console application.
//


#define WIN32_LEAN_AND_MEAN

#include <winsock2.h>
#include <Ws2tcpip.h>
#include <stdio.h>
#include <stdlib.h>

// Link with ws2_32.lib
#pragma comment(lib, "Ws2_32.lib")

int  main()
{

	//---------------------------------------------
	// Declare and initialize variables
	WSADATA wsaData;
	WSABUF DataBuf_Send;
	WSABUF DataBuf_Recv[1];

	WSAOVERLAPPED Overlapped_Send;
	WSAOVERLAPPED Overlapped_Recv;

	SOCKET Socket = INVALID_SOCKET;
	
	struct sockaddr_in RecvAddr;
	struct sockaddr_in LocalAddr;
	int RecvAddrSize = sizeof(RecvAddr);
	int LocalAddrSize = sizeof(LocalAddr);

	u_short Host_Port = 27777;
	struct hostent *localHost;
	char *Host_IP;

	char *Target_IP;
	char *Target_Port;

	Target_IP = "192.168.1.19";
	Target_Port = "8080";

	char SendBuf[8];// = "Data buffer to send";

	//5345543C51530001
	SendBuf[0] = 0x53;
	SendBuf[1] = 0x45;
	SendBuf[2] = 0x54;
	SendBuf[3] = 0x3C;
	SendBuf[4] = 0x51;
	SendBuf[5] = 0x53;
	SendBuf[6] = 0x00;
	SendBuf[7] = 0x01;

	//for (size_t i = 0; i < 8; i++)
	//{
	//	printf("%c", SendBuf[i]);
	//}
	

	int SendBufLen = 8;
	DWORD BytesSend = 0;
	DWORD Flags_Send = 0;

	char RecvBuf[4096];
	int RecvBufLen = 4096;
	DWORD BytesRecv = 0;
	DWORD Flags_Recv = 0;

	int rc, err;
	int retval = 0;

	unsigned int Recv_Count = 0;

	//---------------------------------------------
	// Initialize Winsock
	// Load Winsock
	rc = WSAStartup(MAKEWORD(2, 2), &wsaData);
	if (rc != 0) {
		printf("Unable to load Winsock: %d\n", rc);
		return 1;
	}

	// Make sure the Overlapped_Send struct is zeroed out
	SecureZeroMemory((PVOID)&Overlapped_Send, sizeof(WSAOVERLAPPED));

	// Make sure the Overlapped_Recv struct is zeroed out
	SecureZeroMemory((PVOID)&Overlapped_Recv, sizeof(WSAOVERLAPPED));

	// Create an event handle and setup the Overlapped_Send structure.
	Overlapped_Send.hEvent = WSACreateEvent();
	if (Overlapped_Send.hEvent == WSA_INVALID_EVENT) {
		printf("WSACreateEvent failed with error: %d\n", WSAGetLastError());
		WSACleanup();
		return 1;
	}

	// Create an event handle and setup the Overlapped_Recv structure.
	Overlapped_Recv.hEvent = WSACreateEvent();
	if (Overlapped_Recv.hEvent == WSA_INVALID_EVENT) {
		printf("WSACreateEvent failed with error: %d\n", WSAGetLastError());
		WSACleanup();
		return 1;
	}

	//---------------------------------------------
	// Create a socket for sending data
	Socket =
		WSASocket(AF_INET, SOCK_DGRAM, IPPROTO_UDP, NULL, 0,
		WSA_FLAG_OVERLAPPED);
	if (Socket == INVALID_SOCKET) {
		printf("socket failed with error: %d\n", WSAGetLastError());
		WSACloseEvent(Overlapped_Send.hEvent);
		WSACloseEvent(Overlapped_Recv.hEvent);
		WSACleanup();
		return 1;
	}

	//---------------------------------------------
	// Set up the RecvAddr structure with the IP address of
	// the receiver (in this example case "123.123.123.1")
	// and the specified port number.
	RecvAddr.sin_family = AF_INET;

	RecvAddr.sin_addr.s_addr = inet_addr(Target_IP);
	if (RecvAddr.sin_addr.s_addr == INADDR_NONE)  {
		printf("The target ip address entered must be a legal IPv4 address\n");
		WSACloseEvent(Overlapped_Send.hEvent);
		WSACloseEvent(Overlapped_Recv.hEvent);
		WSACleanup();
		return 1;
	}
	RecvAddr.sin_port = htons((u_short)atoi(Target_Port));
	if (RecvAddr.sin_port == 0) {
		printf("The Target_Port must be a legal UDP port number\n");
		WSACloseEvent(Overlapped_Send.hEvent);
		WSACloseEvent(Overlapped_Recv.hEvent);
		WSACleanup();
		return 1;
	}

	//---------------------------------------------
	// Set up the LocalAddr structure with the local IP address
	// and the specified port number.
	localHost = gethostbyname("");
	Host_IP = inet_ntoa(*(struct in_addr *) *localHost->h_addr_list);

	LocalAddr.sin_family = AF_INET;
	LocalAddr.sin_addr.s_addr = inet_addr(Host_IP);
	LocalAddr.sin_port = htons(Host_Port);

	//---------------------------------------------
	// Bind the sending socket to the LocalAddr structure
	// that has the internet address family, local IP address
	// and specified port number.  
	rc = bind(Socket, (struct sockaddr *) &LocalAddr, LocalAddrSize);
	if (rc == SOCKET_ERROR) {
		printf("bind failed with error: %d\n", WSAGetLastError());
		WSACloseEvent(Overlapped_Send.hEvent);
		WSACloseEvent(Overlapped_Recv.hEvent);
		closesocket(Socket);
		WSACleanup();
		return 1;
	}


	//UDP发送数据
	//////////////////////////////////////////////////////////////////////////

	//---------------------------------------------
	// Send a datagram to the receiver
	printf("Sending datagram from IPv4 address = %s port=%d\n",
		inet_ntoa(LocalAddr.sin_addr), ntohs(LocalAddr.sin_port));
	printf("   to IPv4 address = %s port=%d\n",
		inet_ntoa(RecvAddr.sin_addr), ntohs(RecvAddr.sin_port));

	//    printf("Sending a datagram...\n");
	DataBuf_Send.len = SendBufLen;
	DataBuf_Send.buf = SendBuf;

	rc = WSASendTo(Socket, &DataBuf_Send, 1,
		&BytesSend, Flags_Send, (SOCKADDR *)& RecvAddr,
		RecvAddrSize, &Overlapped_Send, NULL);

	if ((rc == SOCKET_ERROR) && (WSA_IO_PENDING != (err = WSAGetLastError()))) {
		printf("WSASendTo failed with error: %d\n", err);
		WSACloseEvent(Overlapped_Send.hEvent);
		WSACloseEvent(Overlapped_Recv.hEvent);
		closesocket(Socket);
		WSACleanup();
		return 1;
	}

	rc = WSAWaitForMultipleEvents(1, &Overlapped_Send.hEvent, TRUE, INFINITE, TRUE);
	if (rc == WSA_WAIT_FAILED) {
		printf("WSAWaitForMultipleEvents failed with error: %d\n",
			WSAGetLastError());
		retval = 1;
	}

	rc = WSAGetOverlappedResult(Socket, &Overlapped_Send, &BytesSend,
		FALSE, &Flags_Send);
	if (rc == FALSE) {
		printf("WSASendTo failed with error: %d\n", WSAGetLastError());
		retval = 1;
	}
	else
		printf("Number of sent bytes = %d\n", BytesSend);




	//UDP接收数据
	//////////////////////////////////////////////////////////////////////////

	//-----------------------------------------------
	// Call the recvfrom function to receive datagrams
	// on the bound socket.
	DataBuf_Recv[0].len = RecvBufLen;
	DataBuf_Recv[0].buf = RecvBuf;

	wprintf(L"Listening for incoming datagrams on port=%d\n", Host_Port);

	while (true)
	{
		/*
		++Recv_Count;
		//wprintf(L"Listening for incoming datagrams on port=%d, Recv_Count=%d\n", Host_Port, ++Recv_Count);
		rc = WSARecvFrom(Socket,
			&DataBuf_Recv,
			1,
			&BytesRecv,
			&Flags_Recv,
			(SOCKADDR *)& RecvAddr,
			&RecvAddrSize, &Overlapped_Recv, NULL);


		if (rc != 0)
		{
			err = WSAGetLastError();
			if (err != WSA_IO_PENDING)
			{
				wprintf(L"WSARecvFrom failed with error: %ld\n", err);
				WSACloseEvent(Overlapped_Recv.hEvent);
				closesocket(Socket);
				WSACleanup();
				return 1;
			}
			else
			{

				rc = WSAWaitForMultipleEvents(1, &Overlapped_Recv.hEvent, TRUE, INFINITE, TRUE);
				if (rc == WSA_WAIT_FAILED) {
					wprintf(L"WSAWaitForMultipleEvents failed with error: %d\n", WSAGetLastError());
					retval = 1;
				}

				if (!WSAResetEvent(Overlapped_Recv.hEvent))
				{
					wprintf(L"\n WSAResetEvent failed with error: %d\n", WSAGetLastError());
				}

				rc = WSAGetOverlappedResult(Socket, &Overlapped_Recv, &BytesRecv,
					FALSE, &Flags_Recv);
				if (rc == FALSE) {
					wprintf(L"WSArecvFrom failed with error: %d\n", WSAGetLastError());
					retval = 1;
				}
				else
				{
					//wprintf(L"Number of received bytes = %d\n", BytesRecv);

					//printf("SenderAddr.S_addr: %s\n", inet_ntoa(RecvAddr.sin_addr));
					//printf("SenderAddr.sin_port: %d\n", ntohs(RecvAddr.sin_port));
	
					//printf("Recv_Count: %d\n", ++Recv_Count);
				}


			}

		}
		*/

		++Recv_Count;
		//wprintf(L"Listening for incoming datagrams on port=%d, Recv_Count=%d\n", Host_Port, ++Recv_Count);
		rc = WSARecvFrom(Socket,
			DataBuf_Recv,
			1,
			&BytesRecv,
			&Flags_Recv,
			(SOCKADDR *)& RecvAddr,
			&RecvAddrSize, &Overlapped_Recv, NULL);


		if (rc != 0)
		{
			err = WSAGetLastError();
			if (err != WSA_IO_PENDING)
			{
				wprintf(L"WSARecvFrom failed with error: %ld\n", err);
				WSACloseEvent(Overlapped_Recv.hEvent);
				closesocket(Socket);
				WSACleanup();
				return 1;
			}
			else
			{

				WSAWaitForMultipleEvents(1, &Overlapped_Recv.hEvent, TRUE, INFINITE, TRUE);
				WSAResetEvent(Overlapped_Recv.hEvent);
				//WSAGetOverlappedResult(Socket, &Overlapped_Recv, &BytesRecv,FALSE, &Flags_Recv);

				if (Recv_Count > 1500)
				{
					//wprintf(L"Number of received bytes = %d\n", BytesRecv);
					printf("Recv_Count: %d\n", ++Recv_Count);
				}

				//wprintf(L"Number of received bytes = %d\n", BytesRecv);

				//printf("SenderAddr.S_addr: %s\n", inet_ntoa(RecvAddr.sin_addr));
				//printf("SenderAddr.sin_port: %d\n", ntohs(RecvAddr.sin_port));

				//printf("Recv_Count: %d\n", ++Recv_Count);
				}


			}

		}




	//---------------------------------------------
	// When the application is finished sending, close the socket.
	printf("Finished sending. Closing socket.\n");
	WSACloseEvent(Overlapped_Send.hEvent);
	WSACloseEvent(Overlapped_Recv.hEvent);
	closesocket(Socket);
	printf("Exiting.\n");

	//---------------------------------------------
	// Clean up and quit.
	WSACleanup();
	return (retval);
}
