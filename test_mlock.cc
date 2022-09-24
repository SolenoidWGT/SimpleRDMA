#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>
const int alloc_size = 1 * 1024 * 1024 * 1024; //分配32M内存
int main()
{
        char *memory = (char *)malloc(alloc_size);
        if (mlock(memory, alloc_size) == -1)
        {
                perror("mlock");
                return (-1);
        }
        size_t i;
        size_t page_size = getpagesize();
        for (i = 0; i < alloc_size; i += page_size)
        {
                memory[i] = 0;
        }
        printf("addr %p, size %ld\n", memory, page_size);

        if (munlock(memory, alloc_size) == -1)
        {
                perror("munlock");
                return (-1);
        }
        printf("test 32 locked mem ok!\n");

        return 0;
}
