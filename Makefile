CC=gcc
CFLAGS=-Wall -Werror -O2
INCLUDES=
LDFLAGS=-libverbs
LIBS=-pthread -lrdmacm -ldl

SRCS=main.c client.c config.c ib.c server.c setup_ib.c sock.c
OBJS=$(SRCS:.c=.o)
PROG=rdma-tutorial
PROG2=test_mlock

all: $(PROG) $(PROG2)

debug: CFLAGS=-Wall -Werror -g -DDEBUG
debug: $(PROG) $(PROG2)

.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

$(PROG): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $(OBJS) $(LDFLAGS) $(LIBS)

$(PROG2): 
	$(CC) $(CFLAGS) -o $@ test_mlock.cc

clean:
	$(RM) *.o *~ $(PROG) $(PROG2)
