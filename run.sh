rm conv-example
g++ -g -o conv-example conv-example.cc  -L /usr/local/lib -lmkldnn

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/
 ./conv-example