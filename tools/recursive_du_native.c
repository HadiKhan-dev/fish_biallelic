#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <sys/stat.h>
#include <fcntl.h>
#include <stdint.h>
#include <stddef.h>
#include <errno.h>

/* Only directories, multiply linked entries, and failures need Python work. */
struct du_entry {
    uint64_t index, blocks, device, inode;
    uint32_t mode;
    int32_t error;
};

size_t du_batch(int directory_fd, const char *const *names, size_t count,
                struct du_entry *special, uint64_t *totals)
{
    size_t used = 0;
    totals[0] = totals[1] = 0;
    for (size_t i = 0; i < count; ++i) {
        struct stat st;
        int status = fstatat(directory_fd, names[i], &st, AT_SYMLINK_NOFOLLOW);
        int saved_errno = status == 0 ? 0 : errno;
        if (status == 0 && !S_ISDIR(st.st_mode) && st.st_nlink <= 1) {
            totals[0] += (uint64_t)st.st_blocks * 512;
            ++totals[1];
        } else {
            struct du_entry *entry = &special[used++];
            entry->index = i;
            entry->error = saved_errno;
            entry->blocks = status == 0 ? (uint64_t)st.st_blocks : 0;
            entry->device = status == 0 ? (uint64_t)st.st_dev : 0;
            entry->inode = status == 0 ? (uint64_t)st.st_ino : 0;
            entry->mode = status == 0 ? (uint32_t)st.st_mode : 0;
        }
    }
    return used;
}
