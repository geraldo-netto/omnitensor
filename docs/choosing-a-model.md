# Choosing the model a workload runs

Every text workload — `ask-selected-files`, `selected-text-tools`,
`file-organizer`, `event-extraction` — runs whichever model you point it at.
The default is what you get if you never say; it is not a limit.

## The default, and why

**`qwen3-5-9b-iq4-xs`** since 2026-08-17. Measured on this machine's RX 6600 XT
against forty labelled cases, ten per workload, alongside the models that were
default before it:

| workload | Qwen3.5-9B IQ4_XS | Qwen3-8B Q4_K_M | Qwen3-4B Q4_K_M |
| --- | --- | --- | --- |
| ask-selected-files | 9/10 · 18.5s | 9/10 · 68.1s | 9/10 · 37.1s |
| selected-text-tools | **10/10** · 19.4s | 9/10 · 81.4s | 9/10 · 46.3s |
| file-organizer | 8/10 · 39.5s | 8/10 · 168.6s | **10/10** · 80.7s |
| event-extraction | 3/10 | 3/10 | 3/10 |

It matches or beats the 8B on every workload and answers in roughly a quarter
of the time. It is the only model so far to score 10/10 on `selected-text-tools`,
which is where the Hebrew work lives. `event-extraction` is 3/10 on every model
ever measured, on both cards, across two model generations — that is a defect in
the task rather than a property of any model, and choosing differently will not
help it.

The one place the default is not the best answer is `file-organizer`, where the
4B still scores 10/10 against the 9B's 8/10. If that workload matters most to
you, choose the 4B for it — which is exactly what choosing is for.

Seconds are indicative rather than exact: the machine was busy during much of
the measuring, and the same model on the same card has varied by a factor of six
between runs. The accuracy columns are stable.

## Choosing another one

Per workload, through the control socket:

```sh
omnitensor-apply set-profile-model ask-selected-files qwen3-4b-q4-k-m
```

The choice is published in the runtime snapshot as the profile's `modelId`, so
the client shows what is actually running rather than what was asked for. The
worker restarts on the change; jobs already running finish on the model they
started with.

To go back to the default, choose the default explicitly. There is no "unset".

## What is installed here

```sh
ls ~/.local/share/omnitensor/artifacts
```

Anything in that store with a manifest declaration can be selected. Adding one
means installing the weights with their pinned digest and declaring the artifact
in the workload's manifest — see `docs/qwen-workload-installation.md`.

Candidates that were measured but not installed live in
`~/.local/share/omnitensor/benchmark-candidates`, and
`benchmarks/results/` holds what each of them scored.

## Your machine, your choice

A model you select is your decision, and the runtime will run it. Two things it
will still refuse, because they are not preferences:

- **A model that does not fit.** Weights plus the KV cache at 32,768 tokens must
  hold on the card. Check before choosing:
  `python -m omnitensor.probe_cli --artifacts ~/.local/share/omnitensor/artifacts`
  A model that overflows does not refuse — it spills into host memory and
  crawls, and the load log claims full offload throughout. On the integrated
  610M the practical ceiling is about 12.5 GiB per process, well under the 45
  GiB of mapped memory it advertises.
- **A model whose task digest does not match its receipt.** The prompt, schema
  and limits a workload runs are digest-bound; if they change, the receipt is
  reissued or the worker refuses to start. This protects you from a workload
  silently becoming a different one, not from choosing.

Speed follows the card as much as the model. The same 9B answers in about 19
seconds on the discrete RX 6600 XT and roughly twice that on the integrated
610M, at identical accuracy — so the device choice and the model choice are two
independent knobs on the same trade.
