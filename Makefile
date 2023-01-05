# CFLAGS=-Wall -Werror -O2
# CC=/mnt/cache/share/gcc/gcc-7.5.0/bin/gcc
# INCLUDES=-I/mnt/cache/share/cuda-11.3/include 
# LDFLAGS=-L/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib -L/mnt/cache/share/cuda-11.3/lib64 -L/mnt/cache/wangguoteng.p/nccl/build_master/lib
CC=gcc
INCLUDES=-I/usr/local/cuda-11.3/include 
LDFLAGS=-L/mnt/lustre/wangguoteng2/SimpleRDMA/lib -L/usr/local/cuda-11.3/lib64 -L/mnt/lustre/wangguoteng2/nccl/build_master/lib -L/usr/lib64/

LIBS=-libverbs -pthread -lrdmacm -ldl -lnuma -lcudart -lrt -lnccl -lstdc++
SRCS=main.c setup_ib.c sock.c polling.c ib.c bruck.c bruck_ngpus.c
OBJS=$(SRCS:.c=.o)
PROG=rdma-tutorial
PROG2=test_mlock
PROG3=h2d2h

all: $(PROG) $(PROG2)
# -g -DDEBUG_IB
# export LD_LIBRARY_PATH=/mnt/cache/wangguoteng.p/nccl/build_master/lib:/mnt/cache/share/cuda-11.3/lib64:/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib
# # export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:/mnt/lustre/wangguoteng2/nccl/build_master/lib:/usr/lib64/
# make clean && make debug
debug: CFLAGS=-Wall -Werror -g -DDEBUG 
debug: $(PROG) $(PROG2) $(PROG3) 

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

$(PROG2): 
	$(CC) $(CFLAGS) -o $@ test_mlock.cc

$(PROG3):
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ h2d2h.c  $(LDFLAGS) $(LIBS)

clean:
	$(RM) *.o *~ $(PROG) $(PROG2) $(PROG3)
