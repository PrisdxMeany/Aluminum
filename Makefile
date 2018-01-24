CXXFLAGS += -Wall -Wextra -pedantic -Wshadow -O3 -std=c++11 -fopenmp -g -fPIC -lhwloc -I/opt/cudatoolkit-8.0/include
cur_dir = $(shell pwd)
LIB = -L$(cur_dir) -lallreduce -Wl,-rpath=$(cur_dir) -L/usr/workspace/wsb/brain/nccl2/nccl-2.0.5+cuda8.0/lib -lnccl -L/opt/cudatoolkit-8.0/lib64 -lcudart -lrt

all: liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_overlap benchmark_reductions test_correctness test_multi_nballreduces

liballreduce.so: allreduce.cpp allreduce_mpi_impl.cpp allreduce.hpp allreduce_impl.hpp allreduce_mempool.hpp allreduce_mpi_impl.hpp tuning_params.hpp
	mpicxx $(CXXFLAGS) -shared -o liballreduce.so allreduce.cpp allreduce_mpi_impl.cpp

benchmark_allreduces: liballreduce.so benchmark_allreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_allreduces benchmark_allreduces.cpp

benchmark_nballreduces: liballreduce.so benchmark_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_nballreduces benchmark_nballreduces.cpp

benchmark_overlap: liballreduce.so benchmark_overlap.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o benchmark_overlap benchmark_overlap.cpp

test_correctness: liballreduce.so test_correctness.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_correctness test_correctness.cpp

test_multi_nballreduces: liballreduce.so test_multi_nballreduces.cpp
	mpicxx $(CXXFLAGS) $(LIB) -o test_multi_nballreduces test_multi_nballreduces.cpp

benchmark_reductions: benchmark_reductions.cpp
	mpicxx $(CXXFLAGS) -o benchmark_reductions benchmark_reductions.cpp

clean:
	rm -f liballreduce.so benchmark_allreduces benchmark_nballreduces benchmark_reductions test_correctness test_multi_nballreduces benchmark_overlap
