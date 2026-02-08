//========================================================================
// hardware_profiling.h - Hardware resource usage profiling for VRAD RTX
//========================================================================

#ifndef HARDWARE_PROFILING_H
#define HARDWARE_PROFILING_H

// Initialize profiling (takes baseline snapshot)
void HardwareProfile_Init();

// Take a snapshot and log current hardware usage with a label
void HardwareProfile_Snapshot(const char *pszLabel);

// Print peak usage summary (called at end of run)
void HardwareProfile_PrintSummary();

#endif // HARDWARE_PROFILING_H
