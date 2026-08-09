# Installing and verifying OmniTensor plus the Cinnamon applet

A successful install is not evidence of a working one. The unit can be active
while another process owns the bus name, the canonical schemas can be missing
from the installed package, the published snapshot can be stale from a previous
run, and the applet can be a half-copied tree. Each of those looks healthy in
`systemctl` and produces a desktop that silently shows nothing, so the install
sequence below ends with an automated check rather than an assumption.

## Service

Install into an environment of its own rather than the development checkout, so
the running service is a fixed snapshot and not whatever the working tree
happens to contain:

```sh
python3 -m venv ~/.local/share/omnitensor/venv
~/.local/share/omnitensor/venv/bin/pip install /path/to/omnitensor
ln -sf ~/.local/share/omnitensor/venv/bin/omnitensor ~/.local/bin/omnitensor
ln -sf ~/.local/share/omnitensor/venv/bin/omnitensor-verify-install \
       ~/.local/bin/omnitensor-verify-install

install -Dm0644 systemd/omnitensor.service ~/.config/systemd/user/omnitensor.service
systemctl --user daemon-reload
systemctl --user enable --now omnitensor.service
```

Inference stays fail-closed until an artifact store is configured, because
nothing else can prove a model file is the model a manifest declares. Point
`OMNITENSOR_ARTIFACT_ROOT` at the verified store (default
`~/.local/share/omnitensor/artifacts`); with no store the service still runs,
publishes, and answers the bus, but refuses every inference job.

The unit uses `StateDirectory=`, so systemd creates the state directories on
first start; it needs no pre-existing paths. It also declares `Delegate=yes` so
each plugin worker can be accounted in its own cgroup.

The service refuses to start when `org.cinnamon.OmniTensor1` is already owned,
because two instances would publish to the same snapshot path. If start fails
with `already owned`, find the other instance and stop it first — an instance
started by hand outside systemd is the usual cause:

```sh
dbus-send --session --dest=org.freedesktop.DBus --print-reply \
  /org/freedesktop/DBus org.freedesktop.DBus.GetConnectionUnixProcessID \
  string:org.cinnamon.OmniTensor1
```

## Applet

Build the payload, install exactly it, and verify the installed tree against
the checksums it was built from:

```sh
cd /path/to/cinnamon-tpuwlm
npm run package
rm -rf ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto
cp -a dist/cinnamon-tpuwm@geraldo-netto ~/.local/share/cinnamon/applets/
node scripts/package-applet.js verify \
  ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto
```

Reload the applet in the running session without restarting Cinnamon:

```sh
dbus-send --session --dest=org.Cinnamon --type=method_call /org/Cinnamon \
  org.Cinnamon.ReloadXlet string:'cinnamon-tpuwm@geraldo-netto' string:'APPLET'
```

## Verify

```sh
omnitensor-verify-install \
  --applet-root ~/.local/share/cinnamon/applets/cinnamon-tpuwm@geraldo-netto \
  --applet-checksums /path/to/cinnamon-tpuwlm/dist/cinnamon-tpuwm@geraldo-netto.SHA256SUMS
```

It exits non-zero if any check fails and prints one line per check:

| Check | What a failure means |
| --- | --- |
| `executable` | the unit's `ExecStart` does not resolve |
| `service` | the unit is not active |
| `schemas` | a canonical schema is missing from the install, or is present but is not a valid schema |
| `workloads` | the bundled catalog does not load — a packaging defect, not a user error |
| `discovery` | plugin discovery raised instead of returning a catalog |
| `isolation` | an external worker would launch without a sandbox |
| `dbus` | the bus did not answer, or its acknowledgement violates the applet contract |
| `snapshot` | no snapshot, an invalid one, or a stale one the applet would still render |
| `applet` | the installed tree does not match the payload checksums |

The D-Bus check sends a deliberately invalid command and requires a
contract-valid rejection, so it exercises the whole transport without changing
any policy.

Run this before provisioning use cases. Which accelerator lanes the host can
then qualify against is a separate question — see
[use-cases.md](use-cases.md).
