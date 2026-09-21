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
permissions, connection refusal, timeouts, and explicit unsupported hardware.
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

**Status: pending fleet observations and operator agreement.** No real-node
baseline or canary improvement is claimed by this implementation. Local tests
use controlled hardware responses and subprocesses; they do not reproduce a
physical GPU, driver, storage server, or real inference workload.

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

| Observation | Baseline | Canary | Operator-approved target |
| --- | --- | --- | --- |
| Host cohort, model targets, operator | Pending | Pending | Pending |
| UTC window and exported report | Pending | Pending | Pending |
| Engine/runtime and bundle revisions | Pending | Pending | Pending |
| First-attempt success (count/denominator) | Unmeasured | Unmeasured | Await baseline |
| Retry recovery (count/denominator) | Unmeasured | Unmeasured | Await baseline |
| Readiness median and sample count | Unmeasured | Unmeasured | Await baseline |
| Cancellation / unsupported counts | Unmeasured | Unmeasured | Await baseline |
| Unknown outcomes and evidence gaps | Unmeasured | Unmeasured | Await baseline |
| Recurring failures and diagnostic bundles | Pending | Pending | Await baseline |
| Rollout / rollback decision | Pending | Pending | Await baseline |

## Local verification

Automated validation of commit `c37e088` is complete: 2,811 local tests passed,
with 93.72% branch-enabled coverage. One test was skipped because the local
CUDA compiler was unavailable. Lint, formatting, and strict type checks passed.
GitHub CI also passed both [Quality](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505598981)
and [Python 3.13](https://github.com/quadsproject/qiip/actions/runs/35651756542/job/106505599225).
These results validate the implementation; the real-node observations in the
baseline/canary record remain unmeasured.

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
