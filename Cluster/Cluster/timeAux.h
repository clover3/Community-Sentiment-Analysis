#pragma once

#ifdef WINVS
#include <Windows.h>
#endif

#ifdef _WINDOWS_
#else
#include <sys/time.h>
#include <unistd.h>
class __GET_TICK_COUNT
{
public:
	__GET_TICK_COUNT()
	{
		if (gettimeofday(&tv_, NULL) != 0)
			throw 0;
	}
	timeval tv_;
};
__GET_TICK_COUNT timeStart;

unsigned long GetTickCount()
{
	static time_t   secStart = timeStart.tv_.tv_sec;
	static time_t   usecStart = timeStart.tv_.tv_usec;
	timeval tv;
	gettimeofday(&tv, NULL);
	return (tv.tv_sec - secStart) * 1000 + (tv.tv_usec - usecStart) / 1000;
}
using DWORD = unsigned long;
#endif


DWORD lt = GetTickCount();
DWORD elapsed()
{
	DWORD dt = GetTickCount() - lt;
	lt = GetTickCount();
	return dt;
}
