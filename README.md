# python-ggml-matmul
ggml-matmul with pytorch

# quick start

1. copy your cuda path to setup.py line24
2. install pytorch >= 2.1.2
3. install ggml (cd ggml-master, mkdir build, cd build, cmake .., export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/cuda/lib64, cmake --build . --config Release -j 8)
4. pip install .
5. test.py is the example

# changes in ggml

1. ggml-master\src\ggml-cuda.cu   line 2270-2275
2. ggml-master\include\ggml-cuda.h line 45

# to do
supoort to pytorch cuda graph (I have no idea now)
