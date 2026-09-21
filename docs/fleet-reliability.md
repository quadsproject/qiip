# Fleet reliability measurements

Open **Admin > Fleet reliability** (`/dashboard/reliability`). The page and
`GET /admin/provisioning/reliability` require administrator access. Reporting
reads the gateway's retained SQLite attempt records; it never contacts a node
or runs setup, launch, or diagnostic commands.

Filter by UTC start time and representative hostnames, inspect recurring
failures, and download JSON. `since` is inclusive and `until` exclusive; both
require timezone offsets. Repeat `hostname` to select multiple nodes. The
`group_by` parameter accepts `signature`, `stage`, `engine`, `runtime_version`,
`gpu_family`, `os`, `kernel`, `bundle_version`, or `all`. `download=true` returns
an attachment. Each export records its generation time, window, counts,
definitions, evidence gaps, and attempt IDs. Preserve exports outside the log
retention window when comparing releases.

## Definitions

An attempt enters the report when managed provisioning creates its durable
record. Registration/adoption, relaunch, and teardown are excluded. Rejected
API requests, power-on failures, and reconcile failures that occur before an
attempt record exists are outside the denominator.

A provisioning series starts with the first recorded setup for a host and
requested engine/model. Subsequent failed, running, or interrupted attempts
for that same target belong to the series. A completed operation, explicit
cancellation, intervening teardown/relaunch, or requested target change starts
a new series. Runtime and setup-bundle changes do not reset the series. An
automatically selected model retains the original unspecified requested target.

Series identity, ordinal, and start time are committed with the attempt and
survive gateway restarts. This defines first attempts within recorded series,
not the first setup in a machine's lifetime. Legacy attempts without this
identity have unknown origins. If retention has removed history and there is
no retained predecessor, a new origin is conservatively unknown. Legacy
records remain available for outcomes, failure grouping, and drill-down.

| Measure | Numerator / sample | Denominator |
| --- | --- | --- |
| First-attempt success | Known first attempts that finished at the complete/registered stage | Known first attempts starting in the window with a terminal outcome, including cancellation and unsupported hardware |
| Retry recovery | Initially failed series with a successful retry starting in the window | Initially failed series with at least one terminal retry starting in the window; the original failure must still be retained |
| Cancellation | Explicitly cancelled attempts | All terminal provisioning attempts starting in the window |
| Unsupported nodes | Failed attempts carrying the explicit `unsupported_hardware` rejection marker | All terminal provisioning attempts starting in the window |
| Time to usable inference | Median seconds from series start to recorded registered readiness, including retry delays | Successful attempts starting in the window with a known series origin and valid readiness timestamp; sample count and total successes are shown |

Terminal outcomes are success, failure, cancellation, and unsupported hardware.
Running and unknown outcomes are shown separately and excluded from terminal
denominators. A zero denominator displays **Not measured**, not 0%.
Interrupted collection and a completed remote command without completed
registration do not prove success. Readiness means health checks passed and
the node was registered; it does not measure the first generated token.
Older successes without a readiness timestamp are excluded from duration
samples rather than assigned an estimated completion time.

Retry recovery excludes series with no retry and series whose selected retries
are all running or unknown. Its first failure may precede the selected window.
Consequently a successful retry can appear in one window while its first
failure appears in an earlier window. Compare matching windows and cohorts;
do not add percentages from different windows.

## Failure evidence

Signatures use a versioned `v1` symptom classification. Recognized symptoms
include CUDA memory exhaustion, full disks, kernel-header problems,
CUDA probe compilation failures, permissions, connection refusal, timeouts,
and explicit unsupported hardware.
Other errors receive a stable hash after normalizing hostnames, IDs, and
numbers. Missing error text is `v1:unknown`. These are grouping aids, not
confirmed root causes. Each attempt retains its original error text and links
to its exact logs and downloadable diagnostic bundle.

Engine and setup-bundle identity come from the attempt. Measured runtime
version, OS release, and kernel release come only from the latest complete,
retained diagnostic source. Diagnostic collection can occur after the failure;
the bundle contains source timestamps. A retry or configuration change cannot
silently replace a missing historical version with today's configuration.

GPU family uses the setup script's selected runtime-profile family when
available. Otherwise it shows the GPU model names from the diagnostic inventory,
without inferring an architecture from a marketing name. Reports distinguish
unknown dimension values, unknown series origins, incomplete diagnostics,
invalid timestamps, and evicted manifests. Successful attempts often have
unknown environment dimensions because diagnostics are collected on failure.

## Baseline and canary record

**Status: historical snapshot and matched L4 pilot complete; target agreement
pending.** On 2026-09-21 the operator supplied a
development gateway, the existing workload configuration, and no maintenance
constraints. The initially selected T4 hostname returned authoritative
`NXDOMAIN`; the operator replaced it with a reachable, managed L4 node. The
pilot preserves that node's existing model artifact and catalog runtime profile.

A consistent SQLite backup captured 14 retained attempts across five hosts:
10 provisioning attempts and four excluded teardowns. The closed reporting
window is `2026-09-19T15:18:19.137199+00:00` through (exclusive)
`2026-09-21T20:53:01.414633+00:00`. Six provisioning attempts succeeded and four
failed. Nine used llama.cpp and one used vLLM, across six setup-bundle hashes.
These historical records alone do not establish a matched baseline or
demonstrate improvement. All ten provisioning records lack known series origins
and complete diagnostic environment evidence; no readiness samples qualify.
No manifests were evicted.

The four failure bundles were downloaded and their export footers verified:
all retained records were exported, but the underlying manifests still report
incomplete collection. Two failures contain the same fatal CUDA probe compiler
message; one also preserves the compiler's undefined-`printf` diagnostic. The
report now groups these under `v1:cuda_probe_compile` despite different preceding
output. The other failures record a Hugging Face cache-directory conflict and a
launch command deadline. Their original errors, attempt IDs, and bundle hashes
remain in the exports; these observations do not prove a common root cause.

The baseline gateway ran commit `0bdfbcb` with eight locally modified files.
Those edits remain intact. Six files match the candidate exactly; the remaining
placement and CSS edits are incorporated alongside the probe timeout and
diagnostic UI changes. Candidate `76d61ef` was activated at
`2026-09-21T21:31:58.755168+00:00` from a separate release directory, using the
same configuration, dependencies, and data. Startup checks verified all six
previously healthy nodes, the authenticated report and dashboard, and rejection
of unauthenticated report requests. The original checkout and service unit are
preserved for rollback.

Evidence is retained in the operator workspace under `qiip-124-fleet/`:

| Artifact | SHA-256 |
| --- | --- |
| `baseline.sqlite3` | `13cac5070ca3eabafb21c28b5d477f3e7d049073b737be59906ef005d6f66fc1` |
| `baseline-report.json` (original all-dimension export) | `ed78869b12842b12bfd585d5d1d7c7e78c27a149e23f735a9d4f2e4b3fe5f6d3` |
| `baseline-report-grouped.json` (after grouping fix) | `23486f05e17b5a0dddc80ded26b105f994eedb2c5dd1c39ab7bfc22898e73050` |
| `failure-bundles.zip` (four retained bundles) | `b3f834779f415643b69adb0fa4775c9d2de3887c8be19b8f76ac33e1d4d437b0` |

### Matched L4 pilot

The selected node has one NVIDIA L4 (23,034 MiB, SM 8.9), RHEL 9.8, kernel
`5.14.0-687.39.1.el9_8.x86_64`, driver `580.126.09`, CUDA toolkit `13.0.88`,
glibc `2.34`, and llama.cpp `0.4.1 (build 0, commit v0.4.1)`. These are separately
timestamped operator observations; they do not fill missing historical report
dimensions. The workload is `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, catalog profile
`qwen3.6-35b-a3b-24g` version 1, with 262,144 context tokens, one slot, q8_0 KV,
and MTP speculation. Both cycles use graceful teardown followed by the existing
automatic placement claim restoring that same profile.

The fresh baseline completed its only provisioning attempt and returned `READY`
to a bounded inference request. Its observed setup-to-registration duration was
74.554863 seconds, calculated from the attempt start and persisted task's
completion timestamp. The legacy report still correctly shows no measured
readiness sample or known first-attempt denominator. The complete recovery
cycle took 626.648662 seconds from the teardown request, including placement's
configured scheduling and recovery delays. That time is outside the report's
setup-to-readiness definition and is recorded separately.

The proposed pilot targets are one successful first attempt, zero cancellations
and unsupported outcomes, and readiness within 120% of the observed baseline
(89.465836 seconds). They await operator agreement. A successful pilot on this
one node cannot establish fleet-wide failure probabilities; retry recovery
remains unmeasured unless a real failed series is retried.

The candidate completed one first attempt successfully, with a recorded readiness
duration of 72.521237 seconds and no cancellation or unsupported outcome. This
is 97.27% of the observed baseline duration and meets the proposed threshold.
The complete recovery cycle took 636.394442 seconds, including placement delays.
The same inference request returned HTTP 200 and `READY`; all six original
serving nodes were healthy afterward. One sample per version establishes this
pilot's outcome, not a statistically supported performance improvement.

The report retained the selected `consumer-ada` GPU family and explicitly left
runtime, OS, and kernel dimensions unknown: successful attempts do not receive
automatic failure diagnostics. The original unavailable-journal warning also
remains visible as incomplete evidence. Both attempt-bundle exports contained
all retained records. The report's first-attempt denominator, series identity,
readiness timestamp, counts, and evidence gaps were unchanged after a further
gateway restart at `2026-09-21T22:16:31.452682+00:00`; all six nodes remained
healthy. The candidate remains active on the development gateway.

Pilot evidence is retained in `qiip-124-fleet/l4-pilot/`, with the complete set
also preserved on the gateway under `/root/qiip-124-validation/`:

| Artifact | SHA-256 |
| --- | --- |
| `matched-baseline-report.json` | `28e4f06386a9848df54d80047f5a9dbc925efe0fc5c7203bfb10898f87409808` |
| `canary-report.json` | `034df156dcc027e6cdb724d3638ac6d4fe0cdc2ed5ca4b61b3c69d0d1442c2da` |
| `canary-report-after-restart.json` | `5d694ee0a2e0e05cb6e5d1e59a54eba1a58aa6fd4112f9794d0f2a51db01b7fa` |
| `baseline-attempt.ndjson.gz` | `87969b4ea95c3d777a4d63e3e8a7bbdf0920102bac57cc255765be06690f3e64` |
| `canary-attempt.ndjson.gz` | `9d3d600315458dd0ace2bf58c32ce31ce580066e715c0d4ab692e8f119b5f228` |
| `l4-pilot-evidence.tar.gz` (in the parent directory) | `096d90147fe461bd942e2c8f278ac229950cae52febb386a2ab2b7789878e539` |

For additional cohorts, follow this procedure:

1. Choose representative supported and unsupported hosts across the deployed
   GPU families, OS/kernel versions, engines, and recurring failure groups.
   Record the host list, model targets, bundle revisions, and why each host is
   representative. Obtain operator approval before changing fleet nodes.
2. Observe the baseline through the actual managed setup/launch path. Export a
   closed UTC window and the diagnostic bundles for its failures. Record
   incomplete/evicted evidence and keep unknown outcomes visible.
3. Review that baseline with the operator. Agree numeric improvement targets
   for first-attempt success, retry recovery, and readiness time, plus allowed
   cancellation/unsupported outcomes and rollback conditions. Record counts
   and observation duration along with percentages.
4. Run the approved candidate on the same representative cohort and model
   targets. Close and export the canary window before retention expires.
   Compare bundle/version groups and raw failure evidence. Separate reproduced
   software failures from causes confirmed on real nodes.
5. Fill in the record below and link both exports and diagnostic bundles before
   closing the fleet-validation criteria of issue #124.

| Observation | Matched baseline | Canary | Proposed pilot target (agreement pending) |
| --- | --- | --- | --- |
| Host cohort and workload | One L4; existing Qwen3.6 profile above | Same node, artifact, and requested profile | Preserve workload |
| Provisioning attempt ID | `dc078743e2f94134b97fc69097b68726` | `b72add89a89649d7a8e6babfed88c6d5` | Exact attempt drill-down |
| Setup-bundle SHA-256 prefix | `5dba3c1b1c04` | `b0c8793d487d` | Preserve version provenance |
| UTC window and exported report | September 21, 21:19:52.770914–21:31:27.649348; `matched-baseline-report.json` | September 21, 21:32:50.174073–22:15:40.555746; `canary-report.json` | Closed, retained exports |
| Terminal attempt outcomes | 1 succeeded, 0 failed | 1 succeeded, 0 failed | 1 successful first attempt |
| First-attempt success (count/denominator) | Legacy report unmeasured; supervised cycle observed 1 success in 1 attempt | 1/1 (100%) | 1/1 |
| Retry recovery (count/denominator) | Unmeasured (0 eligible series) | Unmeasured | Observe real retries only |
| Readiness median and sample count | Legacy report unmeasured; operator observed 74.554863 s | 72.521237 s, 1 sample | At most 89.465836 s |
| Cancellation / unsupported counts | 0/1 each | 0/1 each | 0 each |
| Unknown outcomes and evidence gaps | 0 unknown outcomes; 1 unknown origin/environment and incomplete manifest | 0 unknown outcomes/origins; 1 incomplete manifest with 3 unknown environment dimensions | Keep evidence gaps visible |
| Full recovery cycle, including placement delays | 626.648662 s | 636.394442 s | Record separately |
| Inference smoke check | HTTP 200; `READY` | HTTP 200; `READY` | Complete the same request |
| Rollout / rollback decision | Original checkout preserved | Candidate active; six nodes healthy; report survived restart | Preserve all six healthy nodes |

## Local verification

Automated validation of commit `c37e088` is complete: 2,811 local tests passed,
with 93.72% branch-enabled coverage. One test was skipped because the local
CUDA compiler was unavailable. Lint, formatting, and strict type checks passed.
GitHub CI also passed both [Quality](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505598981)
and [Python 3.13](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505599225).
These results validate the implementation. The follow-up CUDA grouping
regression failed before the fix and passed afterward. All 20 reliability,
API, setup/launch, and JavaScript
checks passed after that fix, as did lint, formatting, and type checks for the
changed Python files. Reprocessing the real snapshot preserved all outcome
counts, metric denominators, evidence gaps, and original errors while grouping
both compiler failures together.
GitHub [Quality](https://github.com/quadsproject/qiip/actions/runs/35654535559/job/106514755506)
and [Python 3.13](https://github.com/quadsproject/qiip/actions/runs/35654535559/job/106514755311)
checks also passed for the deployed candidate `76d61ef`.

`tests/provisioning/fixtures/reliability.json` covers retries, a bundle change,
first-attempt success, cancellation, unsupported hardware, a running attempt,
an interrupted attempt, failed recovery, legacy history, and excluded teardown.
Its expected first-attempt success is 1/5, retry recovery is 1/2, cancellation
and unsupported rates are each 1/8, and readiness median is 510 seconds across
two measured successes. These are synthetic fixture expectations.

The regression suite also executes the shipped setup, launch, and remote
recorder boundary with controlled probes: a setup fails, the same model is
retried, and registration succeeds. The report must show one recovered series
and a recorded readiness duration. API tests check admin access, export,
filters, timestamps, and exact bundle drill-down; the JavaScript harness
checks literal error rendering, denominators, filters, downloads, and failure
states. Native disclosure controls and labelled filters support keyboard
navigation; the layout reflows and uses the dashboard's light/dark tokens.
