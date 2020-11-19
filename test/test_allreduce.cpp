////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018, Lawrence Livermore National Security, LLC.  Produced at the
// Lawrence Livermore National Laboratory in collaboration with University of
// Illinois Urbana-Champaign.
//
// Written by the LBANN Research Team (N. Dryden, N. Maruyama, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-756777.
// All rights reserved.
//
// This file is part of Aluminum GPU-aware Communication Library. For details, see
// http://software.llnl.gov/Aluminum or https://github.com/LLNL/Aluminum.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "Al.hpp"
#include "test_utils.hpp"
#ifdef AL_HAS_NCCL
#include "test_utils_nccl_cuda.hpp"
#endif
#ifdef AL_HAS_HOST_TRANSFER
#include "test_utils_ht.hpp"
#endif

#include <stdlib.h>
#include <math.h>
#include <string>

size_t start_size = 1;
size_t max_size = 1<<30;

/**
 * Test allreduce algo on input, check with expected.
 */
template <typename Backend>
void test_allreduce_algo(const typename VectorType<Backend>::type& expected,
                         typename VectorType<Backend>::type input,
                         typename Backend::comm_type& comm,
                         typename Backend::allreduce_algo_type algo) {
  auto recv = get_vector<Backend>(input.size());
  // Test regular allreduce.
  // NOTE 代码主要学习部分 Allreduce算法如何进行的
  Al::Allreduce<Backend>(input.data(), recv.data(), input.size(),
                         Al::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
        std::endl;
    std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // Test in-place allreduce.
  // DOUBT 啥叫in-place，是指接收的缓冲区就是发送的缓冲区吗
  Al::Allreduce<Backend>(input.data(), input.size(),
                         Al::ReductionOperator::sum, comm, algo);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
    std::abort();
  }
}

/**
 * Test non-blocking allreduce algo on input, check with expected.
 */
template <typename Backend>
void test_nb_allreduce_algo(const typename VectorType<Backend>::type& expected,
                            typename VectorType<Backend>::type input,
                            typename Backend::comm_type& comm,
                            typename Backend::allreduce_algo_type algo) {
  typename Backend::req_type req = get_request<Backend>();
  auto recv = get_vector<Backend>(input.size());
  // Test regular allreduce.
  Al::NonblockingAllreduce<Backend>(input.data(), recv.data(), input.size(),
                                    Al::ReductionOperator::sum, comm,
                                    req, algo);
  Al::Wait<Backend>(req);
  if (!check_vector(expected, recv)) {
    std::cout << comm.rank() << ": regular allreduce does not match" <<
      std::endl;
    std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  // Test in-place allreduce.
  Al::NonblockingAllreduce<Backend>(input.data(), input.size(),
                                    Al::ReductionOperator::sum, comm,
                                    req, algo);
  Al::Wait<Backend>(req);
  if (!check_vector(expected, input)) {
    std::cout << comm.rank() << ": in-place allreduce does not match" <<
      std::endl;
    std::abort();
  }
}

template <typename Backend>
void test_correctness() {
  //DOUBT 这三个函数作用
  //ANSWER 第一个是获取阻塞的allreduce算法的，可以有多个，第二个是获取非阻塞的算法的，第三个猜测是文中提到将通信和计算绑定的
  auto algos = get_allreduce_algorithms<Backend>();
  auto nb_algos = get_nb_allreduce_algorithms<Backend>();
  // DOUBT 不同文件里的重载函数有啥区别
  typename Backend::comm_type comm = get_comm_with_stream<Backend>(MPI_COMM_WORLD);
  // Compute sizes to test.
  std::vector<size_t> sizes = get_sizes(start_size, max_size, true);
  for (const auto& size : sizes) {
    if (comm.rank() == 0) {
      //NOTE 这命名挺好
      std::cout << "Testing size " << human_readable_size(size) << std::endl;
    }
    // Compute true value.
    typename VectorType<Backend>::type &&data = gen_data<Backend>(size);
    //NOTE 获取期望的结果
    auto expected(data);
    get_expected_allreduce_result(expected);
    // Test algorithms.
    //DOUBT 不止一种Allreduce算法？
    for (auto&& algo : algos) {
      //DOUBT 这里还是用的MPI 所以Backend到底是指的啥 通信时使用的库？
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: " << Al::algorithm_name(algo) << std::endl;
      }
      //NOTE 测试allreduce阻塞算法
      test_allreduce_algo<Backend>(expected, data, comm, algo);
    }
    for (auto&& algo : nb_algos) {
      MPI_Barrier(MPI_COMM_WORLD);
      if (comm.rank() == 0) {
        std::cout << " Algo: NB " << Al::algorithm_name(algo) << std::endl;
      }
      //NOTE 测试allreduce非阻塞算法
      test_nb_allreduce_algo<Backend>(expected, data, comm, algo);
    }
  }
  //NOTE 与前面get_comm_with_stream()对应
  free_comm_with_stream<Backend>(comm);
}


//allreduce测试代码
int main(int argc, char** argv) {
  // Need to set the CUDA device before initializing Aluminum.
//需要预先设置相关的设备

#ifdef AL_HAS_CUDA
//NOTE 主要是CUDA设备的初始化
  set_device();
#endif
  Al::Initialize(argc, argv);

  //DOUBT backend是指通信基于的环境？
  // ANSWER backend指的是基于的后端通信库
  std::string backend = "MPI";
  parse_args(argc, argv, backend, start_size, max_size);
  if (backend == "MPI") {
    test_correctness<Al::MPIBackend>();
#ifdef AL_HAS_NCCL
  } else if (backend == "NCCL") {
    test_correctness<Al::NCCLBackend>();
#endif
#ifdef AL_HAS_MPI_CUDA
  } else if (backend == "MPI-CUDA") {
    std::cerr << "Allreduce not supported on MPI-CUDA backend." << std::endl;
    std::abort();
#endif
#ifdef AL_HAS_HOST_TRANSFER
  } else if (backend == "HT") {
    test_correctness<Al::HostTransferBackend>();
#endif
  }

  Al::Finalize();
  return 0;
}
