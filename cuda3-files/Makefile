all: vectorDotProduct

vectorDotProduct: vectorDotProduct.o
	g++ -fPIC -o $@ $< -L /usr/local/cuda/lib64 -lcudart

vectorDotProduct.o: vectorDotProduct.cu
	/usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
          -I/usr/local/cuda/include \
          -DUNIX -O2 -o $@ -c $<
clean:
	rm -rf *.o
	rm vectorDotProduct
