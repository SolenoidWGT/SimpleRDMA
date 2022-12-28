CC=/mnt/cache/share/gcc/gcc-7.5.0/bin/gcc
CFLAGS=-Wall -Werror -O2
INCLUDES=
LDFLAGS=-libverbs
LIBS=-pthread -lrdmacm -ldl -lnuma

SRCS=main.c setup_ib.c sock.c polling.c ib.c
OBJS=$(SRCS:.c=.o)
PROG=rdma-tutorial
PROG2=test_mlock

all: $(PROG) $(PROG2)
#  -g 
debug: CFLAGS=-Wall -Werror -O3 -DDEBUG -DDEBUG_IB -L/mnt/cache/wangguoteng.p/RDMA-Tutorial/lib
debug: $(PROG) $(PROG2)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

$(PROG2): 
	$(CC) $(CFLAGS) -o $@ test_mlock.cc

clean:
	$(RM) *.o *~ $(PROG) $(PROG2)
