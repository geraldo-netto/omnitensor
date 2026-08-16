# What this service owns, and what a client owns

This service and the desktop client (`../xpuwlm`) are two programs over one
set of workloads. This document says which of them owns each decision, so a
new feature lands on the right side by rule rather than by taste.

The split is **capability against intent**.

- **This service owns what *may* run.** The sandbox and the permission grants,
  the qualification receipts binding a task to a model and to this hardware,
  admission and host pressure, job execution, and the snapshot it publishes.
- **A client owns what *should* run, and how it reads.** Which workloads a
  person wants enabled and at what weight, which card they prefer, what things
  are called on screen, and every preference that changes a rendering rather
  than an execution.

## The service is correct with no client attached

This is the rule that decides most borderline cases, and it is not a
preference: workloads run without a window. `document-intelligence` is
`watching` on an idle desk with nothing open, and the panel applet draws its
icon from the published snapshot alone. So anything whose absence would change
what runs must live here.

That is why **pause is service state**. A pause that ends when a window closes
is not a pause, and a client that owns it would resume every workload the
moment the person quits the application they used to pause them.

It is also why **admission and the host-pressure gate stay here**. They keep a
machine usable while jobs are queued, and there is no client in that loop.

## Policy has one writer and one store

Enabled, weight, device choice, model choice and paused are **intent** — a person's, so a
client sets them — but they are stored *here*, in the policy store that answers
`apply-command`, because this service enforces them with or without that client.

The client sends the change and renders what the snapshot publishes back. It
does not keep its own copy. Two stores for one fact is a bug with a delay on
it: the desktop client kept a private mirror for exactly as long as it took to
drift, and then showed one weight in its window while the scheduler ran another.

## What is refused, and by whom

Everything that refuses lives here, because a client cannot refuse on this
service's behalf: it is unprivileged, replaceable, and frequently absent.

- Permission grants and the sandbox that enforces them.
- Digest-pinned artifacts, and the qualification receipt a worker is checked
  against before it loads a task. **Which models a workload may run** is part
  of that: a client offers only the models the receipt records as passing for
  that workload, and this service refuses any other — a model qualified *as a
  model* is not qualified *for a task*, and an unqualified pair is not a
  slower answer, it is a worker that refuses to start.
- Input roots: a source outside them is refused here, whatever a chooser
  offered.

A client may *pre-empt* a refusal to save a person a wasted job — the desktop
client checks file types and bounds before submitting — but it may never be the
only thing checking.

## Presentation

A manifest's `ui` block is a fact the provider declares about itself, like its
version, and this service carries it. It does not decide display. A client
decides what a person sees, and may override any of it — the desktop client
names five workloads its own way and takes the rest from the manifests.

## The surface

The control surface is the one that already exists, and it is deliberately
small:

| Intent | Method |
| --- | --- |
| Status | the published snapshot, read from disk |
| Start or stop one workload | `apply-command` `set-profile-enabled` |
| Pause everything | `apply-command` `set-paused` |
| Prefer a card, set a weight | `apply-command` `set-profile-device`, `set-profile-weight` |
| Choose the model a workload runs | `apply-command` `set-profile-model` |
| Run something | `submit-job`, `get-job-result`, `cancel-job` |
| Configure a workload | `set-plugin-configuration` (planned, OMNI-0355) |

Two things were considered and rejected, recorded here so they are not
re-derived:

- **No socket verb that stops the service.** Lifecycle belongs to systemd,
  which already owns it. A client able to stop this service is a client able to
  strand every watching workload, and the person who wanted the window closed
  did not ask for that.
- **No rename of the published snapshot path.** It still reads
  `xpu-workload-manager/state.json` from the era when the applet was the whole
  product. Renaming it costs three consumers a coordinated migration and buys
  tidiness.
