# Compiler settings
NVCC = /usr/local/cuda-12.1/bin/nvcc
CCFLAGS = -I"../NetLib" -I/usr/local/include/utf8 -I/usr/local/cuda-samples/Common -O3 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 -ccbin g++
LINKERFLAGS = -L"../NetLib" -L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/usr/local/cuda-12.1/lib64 -lNetLib -lboost_system -lboost_regex -lboost_filesystem -ljsoncpp -lcublas

# Find all .cpp files in the current directory and subdirectories
SRCS = $(wildcard *.cpp) $(wildcard */*.cpp) $(wildcard */*/*.cpp)

# Object files corresponding to source files
OBJS = $(SRCS:.cpp=.o)
PROGRAM_VERSION = 1 # Default program version (single)
NETWORK_VERSION = 1 # Default network version (residual-recurrent)

# Name of the executable
EXEC = SentEncDec

# Default target (single with residual-recurrent)
all: rr

# Define build rules for different configurations
rr: NETWORK_VERSION = 1
rr: $(EXEC)

gru: NETWORK_VERSION = 2
gru: $(EXEC)

search-rr: PROGRAM_VERSION = 2
search-rr: NETWORK_VERSION = 1
search-rr: clean rr

search-gru: PROGRAM_VERSION = 2
search-gru: NETWORK_VERSION = 2
search-gru: clean gru

extend-rr: PROGRAM_VERSION = 3
extend-rr: NETWORK_VERSION = 1
extend-rr: clean rr

extend-gru: PROGRAM_VERSION = 3
extend-gru: NETWORK_VERSION = 2
extend-gru: clean gru

encode: PROGRAM_VERSION = 4
encode: $(EXEC)

decode: PROGRAM_VERSION = 5
decode: $(EXEC)

# Compile source files into object files
%.o: %.cpp
	$(NVCC) -DPROGRAM_VERSION=$(PROGRAM_VERSION) -DNETWORK_VERSION=$(NETWORK_VERSION) $(CCFLAGS) -c $< -o $@

# Build the executable
$(EXEC): $(OBJS)
	$(NVCC) --cudart=static -o $@ $^ $(LINKERFLAGS)

# Clean up
clean:
	rm -f $(OBJS) $(EXEC)
