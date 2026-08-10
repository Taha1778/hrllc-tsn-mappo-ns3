# Private Linux cluster workflow

This directory adds a Linux-native and distributed execution path without
changing the existing Windows/WSL paper-reference workflow.

## What is parallelized

`submit_figure8_ray.py` schedules the 42 independent Figure 8 points: seven
methods over six K values. A point gets an isolated output directory and never
writes the normal project `runs/` or `weights/` paths. The generated results
are explicitly labelled `source_python_reference`; they are not C++ ns-3 data.

The individual MAPPO training rounds remain sequential within one point because
each round depends on the preceding model state.

## Linux worker setup

On a clean, dedicated Linux clone, install the approved build prerequisites,
CPU PyTorch from `https://download.pytorch.org/whl/cpu`, and
`requirements-linux.txt`, then run:

```bash
./scripts/setup_ns3_linux.sh
./scripts/run_training_linux.sh --preflight
```

Do not run this in the live OneDrive checkout from more than one machine. Each
worker receives an exact Git commit and its own clone or container image.

## Container image

Build the image only from a committed revision and push it to the college's
approved private registry:

```bash
docker build -t registry.college.example/hrllc-tsn:<commit> .
```

The three nodes must run the same image tag. Configure Ray/KubeRay to mount one
college-approved persistent results volume at the same path on every node, for
example `/cluster-results`.

`ray_runtime_env.yaml` sends only committed source files with a submitted job;
it excludes local ns-3 builds, generated output, and the live OneDrive state.

## Run the Figure 8 job matrix

From the Ray head node inside the approved private network:

```bash
python3 cluster/submit_figure8_ray.py \
  --address auto \
  --results-root /cluster-results/figure8-<commit>
python3 cluster/aggregate_figure8.py \
  --results-root /cluster-results/figure8-<commit>
```

For a quick smoke test, use `--methods CPPO --scenarios 2`. The normal paper
reference point uses 1,000 MAPPO episodes and 100 evaluation scenarios.

The submitter keeps only as many jobs in flight as the Ray cluster reports CPU
capacity, prints completion progress, and safely reuses a matching existing
point after interruption. Use `--methods BCD --k-values 8,12` for a targeted
recovery. Build the image with `--build-arg GIT_COMMIT=$(git rev-parse HEAD)` so
every result records its exact source revision even when `.git` is not copied
into the container.

## Tests and full runs after a code change

Run these on every pull request or commit:

```bash
python3 -m py_compile python/hrllc_tsn_trainer.py cluster/*.py
python3 -m unittest tests/test_paper_reference_baselines.py -v
python3 cluster/run_figure8_job.py --method CPPO --k 2 --scenarios 2 --output-dir /tmp/hrllc-smoke
```

After those pass, a full Figure 8 cluster run can be submitted manually from
the controller. Keep it manual rather than every push: it schedules 42 jobs,
uses substantial college compute time, and produces research results that must
be reviewed. Save the container image tag and Git commit with the aggregated
results.

The repository also includes two GitHub Actions workflows:

- `Verify source changes` runs syntax checks, the baseline regression, and a
  short CPPO smoke point on every push and pull request.
- `Full Figure 8 cluster run` is manual-only and runs on the controller's
  `cluster-controller` self-hosted runner. Set the repository variable
  `RAY_DASHBOARD_ADDRESS` to the internal Ray Jobs address. Do not register
  this runner with a public repository or allow untrusted pull-request code to
  use its label.

## Native ns-3 QoS extension

`configs/extension_qos.json` is a separately labelled extension configuration.
It keeps the network and QoS constraints unchanged while screening six MAPPO-H
policy choices: reliability-reward weight 1/2/4 with the conservative retry
mask disabled/enabled. `cluster/submit_ns3_campaign_ray.py` gives every Ray
task a private `runs/` and `weights/` workspace, trains Python MAPPO, then
executes the local C++ ns-3 binary and ranks candidates by the equal-weight
mean of failure, latency-violation, and reliability-violation rates.

The `Native ns-3 extension campaign` GitHub workflow runs only for trusted
direct pushes to `cluster`; it never runs for pull requests or forks. It first
performs the native preflight, synchronizes the exact Python/C++ source revision
to each Ray pod, rebuilds ns-3 locally, and persists `selection.json` under
`/cluster-results/native-ns3/<commit>-<run-id>`.
