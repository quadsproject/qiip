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

**Status: historical fleet snapshot collected; matched baseline, targets, and
canary pending.** On 2026-09-21 the operator supplied a development gateway,
one Tesla T4 canary, the existing workload configuration, and no maintenance
constraints. The gateway is reachable, but both of its configured DNS servers
return authoritative `NXDOMAIN` for the supplied canary hostname. QUADS lists
that host as available with one T4 and no assigned workload or attempt history;
its inventory provides no host IP address. A reachable address is needed before
the managed setup/launch comparison can run. No candidate deployment, gateway
restart, or node mutation has been performed during this observation.

A consistent SQLite backup captured 14 retained attempts across five hosts:
10 provisioning attempts and four excluded teardowns. The closed reporting
window is `2026-09-19T15:18:19.137199+00:00` through (exclusive)
`2026-09-21T20:53:01.414633+00:00`. Six provisioning attempts succeeded and four
failed. Nine used llama.cpp and one used vLLM, across six setup-bundle hashes.
This retained historical cohort does not establish a matched T4 baseline or
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

The gateway runs base commit `0bdfbcb` with eight locally modified files. Those
edits were inspected without changing them. Six files match the candidate
exactly; the remaining placement and CSS edits are incorporated in the
candidate alongside the probe timeout and diagnostic UI changes. Any later
deployment must preserve the current configuration, data, and rollback copy.

Evidence is retained in the operator workspace under `qiip-124-fleet/`:

| Artifact | SHA-256 |
| --- | --- |
| `baseline.sqlite3` | `13cac5070ca3eabafb21c28b5d477f3e7d049073b737be59906ef005d6f66fc1` |
| `baseline-report.json` (original all-dimension export) | `ed78869b12842b12bfd585d5d1d7c7e78c27a149e23f735a9d4f2e4b3fe5f6d3` |
| `baseline-report-grouped.json` (after grouping fix) | `23486f05e17b5a0dddc80ded26b105f994eedb2c5dd1c39ab7bfc22898e73050` |
| `failure-bundles.zip` (four retained bundles) | `b3f834779f415643b69adb0fa4775c9d2de3887c8be19b8f76ac33e1d4d437b0` |

The remaining validation procedure is:

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

| Observation | Retained historical observation | Canary | Operator-approved target |
| --- | --- | --- | --- |
| Host cohort and workload | Five historical hosts; unmatched cohort | One T4 selected; existing setup defaults; DNS blocked | Await matched baseline |
| UTC window and exported report | September 19-21; exports above | Not started | Await matched baseline |
| Engine/runtime and bundle revisions | 9 llama.cpp, 1 vLLM; 6 bundles; measured runtime unknown | Not started | Await matched baseline |
| Terminal attempt outcomes | 6 succeeded, 4 failed | Unmeasured | Await matched baseline |
| First-attempt success (count/denominator) | Unmeasured (0 known terminal first attempts) | Unmeasured | Await matched baseline |
| Retry recovery (count/denominator) | Unmeasured (0 eligible series) | Unmeasured | Await matched baseline |
| Readiness median and sample count | Unmeasured (0 samples, 6 successes) | Unmeasured | Await matched baseline |
| Cancellation / unsupported counts | 0/10 each | Unmeasured | Await matched baseline |
| Unknown outcomes and evidence gaps | 0 unknown outcomes; 10 unknown origins/environments and incomplete manifests | Unmeasured | Await matched baseline |
| Recurring failures and retained bundles | 2 CUDA probe compilation failures; 4 bundles saved | Not started | Await matched baseline |
| Rollout / rollback decision | Candidate not deployed | Pending reachable canary | Await matched baseline |

## Local verification

Automated validation of commit `c37e088` is complete: 2,811 local tests passed,
with 93.72% branch-enabled coverage. One test was skipped because the local
CUDA compiler was unavailable. Lint, formatting, and strict type checks passed.
GitHub CI also passed both [Quality](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505598981)
and [Python 3.13](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505599225).
These results validate the implementation; the matched baseline/canary metrics
remain unmeasured. The follow-up CUDA grouping regression failed before the fix
and passed afterward. All 20 reliability, API, setup/launch, and JavaScript
checks passed after that fix, as did lint, formatting, and type checks for the
changed Python files. Reprocessing the real snapshot preserved all outcome
counts, metric denominators, evidence gaps, and original errors while grouping
both compiler failures together.

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
