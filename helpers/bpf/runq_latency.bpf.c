// Run-queue and block I/O latency, aggregated in kernel space.
//
// Nothing here records a pid, a comm, or a path.  The maps hold only log2
// bucket counts, so the identifying data never leaves the kernel — which is
// what lets an unprivileged reader consume this at all.
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

char LICENSE[] SEC("license") = "GPL";

#define MAX_SLOTS 27

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_SLOTS);
    __type(key, __u32);
    __type(value, __u64);
} runq_latency_us SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, MAX_SLOTS);
    __type(key, __u32);
    __type(value, __u64);
} block_latency_us SEC(".maps");

// Wakeup timestamps are keyed by pid only so the delta can be computed; the
// key never leaves the kernel and the entry is deleted as soon as it is used.
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);
    __type(value, __u64);
} wakeup_at SEC(".maps");

static __always_inline __u32 log2_slot(__u64 value)
{
    __u32 slot = 0;
    while (value > 1 && slot < MAX_SLOTS - 1) {
        value >>= 1;
        slot++;
    }
    return slot;
}

static __always_inline void record(void *map, __u64 microseconds)
{
    __u32 slot = log2_slot(microseconds);
    __u64 *count = bpf_map_lookup_elem(map, &slot);
    if (count)
        __sync_fetch_and_add(count, 1);
}

SEC("tp_btf/sched_wakeup")
int BPF_PROG(on_wakeup, struct task_struct *task)
{
    __u32 pid = BPF_CORE_READ(task, pid);
    __u64 now = bpf_ktime_get_ns();
    bpf_map_update_elem(&wakeup_at, &pid, &now, BPF_ANY);
    return 0;
}

SEC("tp_btf/sched_switch")
int BPF_PROG(on_switch, bool preempt, struct task_struct *prev, struct task_struct *next)
{
    __u32 pid = BPF_CORE_READ(next, pid);
    __u64 *queued = bpf_map_lookup_elem(&wakeup_at, &pid);
    if (!queued)
        return 0;
    __u64 waited = bpf_ktime_get_ns() - *queued;
    bpf_map_delete_elem(&wakeup_at, &pid);
    record(&runq_latency_us, waited / 1000);
    return 0;
}

SEC("tp_btf/block_rq_complete")
int BPF_PROG(on_block_complete, struct request *rq, int error, unsigned int nr_bytes)
{
    __u64 started = BPF_CORE_READ(rq, start_time_ns);
    if (!started)
        return 0;
    record(&block_latency_us, (bpf_ktime_get_ns() - started) / 1000);
    return 0;
}
