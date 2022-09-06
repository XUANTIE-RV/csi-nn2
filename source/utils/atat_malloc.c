/****************************************************************
Copyright 1990, 1994, 2000 by AT&T, Lucent Technologies and Bellcore.

Permission to use, copy, modify, and distribute this software
and its documentation for any purpose and without fee is hereby
granted, provided that the above copyright notice appear in all
copies and that both that the copyright notice and this
permission notice and warranty disclaimer appear in supporting
documentation, and that the names of AT&T, Bell Laboratories,
Lucent or Bellcore or any of their entities not be used in
advertising or publicity pertaining to distribution of the
software without specific, written prior permission.

AT&T, Lucent and Bellcore disclaim all warranties with regard to
this software, including all implied warranties of
merchantability and fitness.  In no event shall AT&T, Lucent or
Bellcore be liable for any special, indirect or consequential
damages or any damages whatsoever resulting from loss of use,
data or profits, whether in an action of contract, negligence or
other tortious action, arising out of or in connection with the
use or performance of this software.
****************************************************************/

// #define F _malloc_free_

#include <string.h>
#include <unistd.h>

#ifdef SHL_BUILD_RTOS
#define SBGULP 0x800000
#else
#define SBGULP 0x8000000
#endif

typedef struct shl_atat_mem {
    struct shl_atat_mem *next;
    size_t len;
} shl_atat_mem;

#define MINBLK (2 * sizeof(struct shl_atat_mem) + 16)

shl_atat_mem *F;

static char *sbrk_wrapper(int size)
{
#ifdef SHL_BUILD_RTOS
    return (char *)0x60000000;
#else
    return sbrk(size);
#endif
}

void *shl_atat_malloc(register size_t size)
{
    register shl_atat_mem *p, *q, *r, *s;
    unsigned register k, m;
    //  extern void *sbrk(Int);
    char *top, *top1;

    size = (size + 7) & ~7;
    r = (shl_atat_mem *)&F;
    for (p = F, q = 0; p; r = p, p = p->next) {
        if ((k = p->len) >= size && (!q || m > k)) {
            m = k;
            q = p;
            s = r;
        }
    }
    if (q) {
        if (q->len - size >= MINBLK) { /* split block */
            p = (shl_atat_mem *)(((char *)(q + 1)) + size);
            p->next = q->next;
            p->len = q->len - size - sizeof(shl_atat_mem);
            s->next = p;
            q->len = size;
        } else {
            s->next = q->next;
        }
    } else {
        top = (void *)(((long)sbrk_wrapper(0) + 7) & ~7);
        if (F && (char *)(F + 1) + F->len == top) {
            q = F;
            F = F->next;
        } else {
            q = (shl_atat_mem *)top;
        }
        top1 = (char *)(q + 1) + size;
        if (sbrk_wrapper((int)(top1 - top + SBGULP)) == (void *)-1) {
            return 0;
        }
        r = (shl_atat_mem *)top1;
        r->len = SBGULP - sizeof(shl_atat_mem);
        r->next = F;
        F = r;
        q->len = size;
    }
    return (void *)(q + 1);
}

void shl_atat_free(void *f)
{
    shl_atat_mem *p, *q, *r;
    char *pn, *qn;

    if (!f) return;
    q = (shl_atat_mem *)((char *)f - sizeof(shl_atat_mem));
    qn = (char *)f + q->len;
    for (p = F, r = (shl_atat_mem *)&F;; r = p, p = p->next) {
        if (qn == (void *)p) {
            q->len += p->len + sizeof(shl_atat_mem);
            p = p->next;
        }
        pn = p ? ((char *)(p + 1)) + p->len : 0;
        if (pn == (void *)q) {
            p->len += sizeof(shl_atat_mem) + q->len;
            q->len = 0;
            q->next = p;
            r->next = p;
            break;
        }
        if (pn < (char *)q) {
            r->next = q;
            q->next = p;
            break;
        }
    }
}

void *shl_atat_calloc(size_t n, size_t m)
{
    void *rv;
    rv = shl_atat_malloc(n *= m);
    if (n && rv) {
        memset(rv, 0, n);
    }
    return rv;
}
