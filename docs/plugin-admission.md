# Plugin admission: how weight decides anything for plugin workloads

An inference job queues in a scheduler backend lane, where the profile's
`weight` decides how often it is served while lanes contend. A job that runs in
an **installed plugin** — `ask-selected-files`, `selected-text-tools`,
`file-organizer`, `event-extraction`, `media-transcription` and the other
external providers — never entered those lanes. It was handed to its worker the
moment it was accepted, so:

- every plugin ran as many jobs at once as callers submitted, and
- `weight` was a stored number that ordered nothing,

which is why a settings surface could only draw Enabled, Weight and Run-on
disabled for exactly the workloads a person starts by hand.

`omnitensor.plugin_admission.PluginAdmissionQueue` is the missing half.

## How it works

1. **A slot, not a queue ticket.** Before its worker call begins, a plugin job
   takes one of a bounded number of slots. The slot is held for the whole
   worker call and released when it settles. A pool released at dispatch would
   bound nothing.
2. **Stride selection.** When the pool is full, waiting profiles are ordered by
   the rule the scheduler already uses: the pending profile with the smallest
   pass value goes next, and its pass advances by `1 / weight`. A profile at
   weight 5 is therefore served about five times per weight-1 serve *while both
   have work waiting*. With no contention, weight changes nothing — there is
   nothing to order.
3. **Newcomers join at virtual time.** A profile that has just started waiting
   adopts the smallest pass currently tracked, so it cannot monopolise the pool
   "catching up" on service it never waited for.
4. **Disabled means held, not dropped.** A profile the policy rejects (its
   `enabled` is false, or the whole runtime is paused) keeps its place and its
   queued jobs. Its pass is clamped up to the pool's virtual time, so being
   held neither banks credit nor costs it any. Re-enabling it, or unpausing,
   kicks the pool immediately rather than waiting for a running job to finish.
5. **Cancelling while queued is free.** The worker call has not started, so
   there is nothing to stop and no slot to give back.
6. **A full waiting list refuses at admission.** Beyond the waiting bound, a
   submission is answered `plugin-queue-full` before a job id is minted, rather
   than being accepted into a queue it would never leave.

Policy for a plugin profile is adopted the first time the runtime discovers it
(`OmniTensorService._adopt_plugin_profiles`) and is persisted like any other
policy, so a plugin someone disabled or weighted stays that way across
restarts. Adoption increments the policy revision — a client holding the
previous revision refreshes, exactly as after any other policy change.

## Tuning

| Knob | Default | Ceiling | What it controls |
| --- | --- | --- | --- |
| `OMNITENSOR_PLUGIN_SLOTS` | 2 | 32 | Plugin jobs running at once, across every plugin |
| `plugin_slots=` (service argument) | 2 | 32 | The same, for an embedded service |
| `max_waiting=` (queue argument) | 64 | 1024 | Jobs allowed to wait before submissions are refused |

An unreadable `OMNITENSOR_PLUGIN_SLOTS` is the default rather than a startup
failure: a typo in a tuning knob should not take the runtime down.

Choosing a value:

- **2 (default).** A plugin worker is a whole process, usually holding a model.
  Two keeps one running while another loads, and keeps the pool contended
  enough that weight is a real setting.
- **1.** Strict serialisation. Useful on a machine with one GPU and a large
  model, where two concurrent workers means both are slow or one is killed.
  Weight still orders who goes next.
- **4 or more.** Only with the memory to hold that many workers at once. Past
  the point where jobs stop waiting, weight stops meaning anything, because
  nothing contends.

Set it where the service reads its environment:

```sh
OMNITENSOR_PLUGIN_SLOTS=1 omnitensor-service
```

## Reading it back

The published snapshot now carries a profile entry for every installed plugin,
not only for catalog workloads: `status` (`idle`, `running`, or `paused` when
policy holds it), `queued`, and a detail naming the weight it is waiting at.
That entry is also what tells a client the profile is governed at all — the
reason the three policy controls can be drawn live for plugins.
