#ifndef UTILS_H__
#define UTILS_H__

#include <iostream>

// Copied from helper_cuda.h

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line)
{
  if (result)
  {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

static const char *_cudaGetErrorEnum(cudaError_t error)
{
  return cudaGetErrorName(error);
}

// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file,
                               const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program in case error detected.
#define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file,
                                 const int line)
{
  cudaError_t err = cudaGetLastError();

  if (cudaSuccess != err)
  {
    fprintf(stderr,
            "%s(%i) : getLastCudaError() CUDA error :"
            " %s : (%d) %s.\n",
            file, line, errorMessage, static_cast<int>(err),
            cudaGetErrorString(err));
  }
}

// Copied from helper_string.h

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter(char delimiter, const char *string)
{
  int string_start = 0;

  while (string[string_start] == delimiter)
  {
    string_start++;
  }

  if (string_start > static_cast<int>(strlen(string) - 1))
  {
    return 0;
  }

  return string_start;
}

inline bool checkCmdLineFlag(const int argc, const char **argv,
                             const char *string_ref)
{
  bool bFound = false;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(
          equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length &&
          !strncasecmp(string_argv, string_ref, length))
      {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}

// This function wraps the CUDA Driver API into a template function
template <class T>
inline bool getCmdLineArgumentValue(const int argc, const char **argv,
                                    const char *string_ref, T *value)
{
  bool bFound = false;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length))
      {
        if (length + 1 <= static_cast<int>(strlen(string_argv)))
        {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          *value = (T)atoi(&string_argv[length + auto_inc]);
        }

        bFound = true;
        i = argc;
      }
    }
  }

  return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref)
{
  bool bFound = false;
  int value = -1;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length))
      {
        if (length + 1 <= static_cast<int>(strlen(string_argv)))
        {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        }
        else
        {
          value = 0;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound)
  {
    return value;
  }
  else
  {
    return 0;
  }
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv,
                                     const char *string_ref)
{
  bool bFound = false;
  float value = -1;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length))
      {
        if (length + 1 <= static_cast<int>(strlen(string_argv)))
        {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = static_cast<float>(atof(&string_argv[length + auto_inc]));
        }
        else
        {
          value = 0.f;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound)
  {
    return value;
  }
  else
  {
    return 0;
  }
}

inline bool getCmdLineArgumentString(const int argc, const char **argv,
                                     const char *string_ref,
                                     char **string_retval)
{
  bool bFound = false;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      char *string_argv = const_cast<char *>(&argv[i][string_start]);
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length))
      {
        *string_retval = &string_argv[length + 1];
        bFound = true;
        continue;
      }
    }
  }

  if (!bFound)
  {
    *string_retval = NULL;
  }

  return bFound;
}

//

struct GpuTimer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

  GpuTimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  void stop()
  {
    cudaEventRecord(stop_, 0);
  }

  float elapsed()
  {
    float elapsed_;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed_, start_, stop_);
    return elapsed_;
  }
};

#endif
