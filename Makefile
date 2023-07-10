CUDA_ROOT_DIR=/usr/local/cuda-12.2

CC=g++
CC_FLAGS=
CC_LIBS=

NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

CUDA_LIB_DIR= -L $(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I $(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

SRC_DIR = src
OBJ_DIR = bin
INC_DIR = include

EXE = sim_run.out
OBJS = $(OBJ_DIR)/cuda_main.o $(OBJ_DIR)/print_util.o $(OBJ_DIR)/single_gate_simulation.o $(OBJ_DIR)/nQubit_gate_simulation.o

$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(OBJ_DIR)/%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp include/%.h
	$(CC) $(CC_FLAGS) -c $< -o $@

$(OBJ_DIR)/cuda_main.o : $(SRC_DIR)/cuda_main.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

run: $(EXE)
	./$(EXE)

clean:
	$(RM) bin/* *.o $(EXE)