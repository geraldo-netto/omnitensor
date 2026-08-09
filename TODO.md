# TODO

## Findings

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
| OMNI-0029 | open | medium | xs | — | Cinnamon integration: OmniTensor publishes by default to `~/.local/state/omnitensor/runtime-snapshot.json`, while the applet reads `~/.local/state/tpu-workload-manager/state.json`; zero-configuration installs never connect, and the systemd sandbox cannot write the applet path. |
| OMNI-0030 | open | medium | m | — | workload catalog: OmniTensor ships no built-in manifests for the applet's nine bundled profiles, so profile commands are rejected as unknown and snapshots omit every advertised use case unless operators duplicate the catalog manually. |
| OMNI-0031 | open | low | s | OMNI-0030 | documentation: OmniTensor does not map the Cinnamon workload profiles to model, host-pipeline, routing, and acceptance responsibilities, leaving operators unable to tell catalog readiness from implemented inference. |

## Rejected / Won't fix

| id | status | severity | effort | related ids | description |
| --- | --- | --- | --- | --- | --- |
