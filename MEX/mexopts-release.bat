@echo off
rem MEXOPTS.BAT
rem
rem    Compile and link options used for building MEX-files
rem    using NVCC and the Microsoft Visual C++ compiler version 10.0.
rem
rem    Copyright 2012 The MathWorks, Inc.
rem
rem StorageVersion: 1.0
rem C++keyFileName: NVCCOPTS.BAT
rem C++keyName: nvcc
rem C++keyManufacturer: NVIDIA
rem C++keyVersion: 
rem C++keyLanguage: C++
rem C++keyLinkerName: Microsoft Visual C++ 2010
rem C++keyLinkerVersion: 10.0
rem
rem ********************************************************************
rem General parameters
rem ********************************************************************

set MATLAB=%MATLAB%
set VSINSTALLDIR=%VS120COMNTOOLS%\..\..
set VCINSTALLDIR=%VSINSTALLDIR%\VC
rem In this case, LINKERDIR is being used to specify the location of the SDK
set LINKERDIR='.registry_lookup("SOFTWARE\Microsoft\Microsoft SDKs\Windows\v7.1A" , "InstallationFolder").'
rem We assume that the CUDA toolkit is already on your path. If this is not the
rem case, you can set the environment variable MW_NVCC_PATH to the place where
rem nvcc is installed.
set PATH=%MW_NVCC_PATH%;%VCINSTALLDIR%\bin\amd64;%VCINSTALLDIR%\bin;%VCINSTALLDIR%\VCPackages;%VSINSTALLDIR%\Common7\IDE;%VSINSTALLDIR%\Common7\Tools;%LINKERDIR%\bin\x64;%LINKERDIR%\bin;%MATLAB_BIN%;%PATH%
rem Include path needs to point to a directory that includes gpu/mxGPUArray.h
set INCLUDE=%VCINSTALLDIR%\INCLUDE;%VCINSTALLDIR%\ATLMFC\INCLUDE;%LINKERDIR%\include;%INCLUDE%;%MATLAB%\toolbox\distcomp\gpu\extern\include;%CUDA_PATH%\include
rem extern\lib\win64 points to gpu.lib: CUDA_LIB_PATH points to cudart.lib
set LIB=%VCINSTALLDIR%\LIB\amd64;%VCINSTALLDIR%\ATLMFC\LIB\amd64;%LINKERDIR%\lib\x64;%MATLAB%\extern\lib\win64;%LIB%;%CUDA_LIB_PATH%;%CUDA_PATH%\lib\x64
set MW_TARGET_ARCH=win64

rem ********************************************************************
rem Compiler parameters
rem ********************************************************************
set COMPILER=nvcc
set COMPFLAGS=-gencode=arch=compute_35,code=sm_35 -c --compiler-options=/GR,/W0,/EHs,/D_CRT_SECURE_NO_DEPRECATE,/D_SCL_SECURE_NO_DEPRECATE,/D_SECURE_SCL=0,/DMATLAB_MEX_FILE,/nologo,/openmp,/MT,/D_SECURE_SCL=0,/D_ITERATOR_DEBUG_LEVEL=0
set OPTIMFLAGS=--compiler-options=/Ox,/Oy-,/DNDEBUG
set DEBUGFLAGS=--compiler-options=/Z7

rem ********************************************************************
rem Linker parameters
rem ********************************************************************
rem Link with the standard mex libraries and gpu.lib.
set LIBLOC=%MATLAB%\extern\lib\win64\microsoft
set LINKER=link
set LINKFLAGS=/dll /export:%ENTRYPOINT% /LIBPATH:"%LIBLOC%" libmx.lib libmex.lib libmat.lib gpu.lib cudart.lib cufft.lib cublas.lib curand.lib "..\x64\Release\GTOM.lib" /MACHINE:X64 /nologo /manifest /incremental:NO /implib:"%LIB_NAME%.x" /MAP:"%OUTDIR%%MEX_NAME%%MEX_EXT%.map"
set LINKOPTIMFLAGS=
set LINKDEBUGFLAGS=
set LINK_FILE=
set LINK_LIB=
set NAME_OUTPUT=/out:"%OUTDIR%%MEX_NAME%%MEX_EXT%"
set RSP_FILE_INDICATOR=@

rem ********************************************************************
rem Resource compiler parameters
rem ********************************************************************
set RC_COMPILER=rc /fo "%OUTDIR%.res"
set RC_LINKER=

set POSTLINK_CMDS=del "%LIB_NAME%.x" "%LIB_NAME%.exp"
set POSTLINK_CMDS1=mt -outputresource:"%OUTDIR%%MEX_NAME%%MEX_EXT%;2" -manifest "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS2=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.manifest"
set POSTLINK_CMDS3=del "%OUTDIR%%MEX_NAME%%MEX_EXT%.map"
