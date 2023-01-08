# CFLAGS=-Wall -Werror -O2
CC=/mnt/cache/share/gcc/gcc-7.5.0/bin/gcc
INCLUDES=-I./include -I/mnt/cache/share/cuda-11.3/include 
LDFLAGS=-L/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib -L/mnt/cache/share/cuda-11.3/lib64 -L/mnt/cache/wangguoteng.p/nccl/build_master/lib -L/mnt/cache/wangguoteng.p/SimpleRDMA/lib
# CC=gcc
# INCLUDES=-I/usr/local/cuda-11.3/include 
# LDFLAGS=-L/mnt/lustre/wangguoteng2/SimpleRDMA/lib -L/usr/local/cuda-11.3/lib64 -L/mnt/lustre/wangguoteng2/nccl/build_master/lib -L/usr/lib64/
# -rpath=/mnt/cache/wangguoteng.p/SimpleRDMA/lib
# ,-rpath=./lib
$(info $(LDFLAGS) )
# about -wl and rpath
# https://stackoverflow.com/questions/6562403/i-dont-understand-wl-rpath-wl
LIBS=-libverbs -pthread -lrdmacm -ldl -lnuma -lcudart -lrt -lnccl -lstdc++ -lpci -Wl,-rpath ./lib
SRCS=main.c setup_ib.c sock.c polling.c ib.c bruck.c bruck_ngpus.c
OBJS=$(SRCS:.c=.o)
PROG=rdma-tutorial
PROG2=test_mlock
PROG3=h2d2h

all: $(PROG) $(PROG2)
# -g -DDEBUG_IB
# export LD_LIBRARY_PATH=/mnt/cache/wangguoteng.p/nccl/build_master/lib:/mnt/cache/share/cuda-11.3/lib64:/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib
# # export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/mnt/lustre/wangguoteng2/nccl/build_master/lib:/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib
# make clean && make debug
debug: CFLAGS=-Wall -Wno-unused-variable -Werror -g -DDEBUG 
debug: $(PROG) $(PROG2) $(PROG3) 

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS)  $(LIBS) $(LDFLAGS)

$(PROG2): 
	$(CC) $(CFLAGS) -o $@ test_mlock.cc

$(PROG3):
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ h2d2h.c  $(LIBS)  $(LDFLAGS)

clean:
	$(RM) *.o *~ $(PROG) $(PROG2) $(PROG3)
