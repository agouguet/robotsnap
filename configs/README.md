# Configurations

Two levels of defaults, one mechanism:

- `algorithms/` - one file per learning rule (`ppo`, `dqn`, `a2c`, `sac`,
  `reinforce`): the discount, the learning rate, the network width, the
  algorithm's own `policy_kwargs`/`algo_kwargs`, the pacing.
- `methods/` - one file per method of the literature (`cadrl`, `sarl`,
  `ga3c_cadrl`, `rgl`, `template`): the options that reach that method, which
  brings its own observation, action set, reward and policy.
- `curriculum/` - one file per ladder of scenarios: each stage names the
  scenario it runs and the rule that promotes the next one (`--curriculum`).
- `benchmarks/` - one file per suite of scenarios a campaign runs
  (`--suite`), with the number of episodes each scenario is played for.

Both are used the same way:

```console
python -m robotsnap train --config ppo
python -m robotsnap train --config configs/algorithms/ppo.yaml
python -m robotsnap train --config /abs/path/to/my.yaml
```

**Precedence.** Defaults defined in the code < values read from the
configuration file < options typed explicitly on the command line. A file only
supplies what the line left out, so `--config ppo --gamma 0.5` trains with
`gamma = 0.5`.

**Name resolution.** A name that is not a path is looked up as
`configs/algorithms/<name>.yaml`, then `configs/methods/<name>.yaml`, then
`configs/curriculum/<name>.yaml`, then `configs/benchmarks/<name>.yaml`, and
finally anywhere under `configs/`. `.yml` and `.json` are accepted too. An
unknown name raises an error listing the configurations that do exist. A path
is used as given: relative to the current directory, or absolute.

**The keys.** Every key is the name of an option of the sub-command it is
applied to (`train`, `play`, `bench`, ...) - the same name the help prints with
dashes replaced by underscores, e.g. `observation_params` for
`--observation-params`. A key the command does not define is refused with both
the key and the command named, so a typo never half-configures a run.
`robotsnap.config.describe` renders a file as one `key = value` line per entry,
which is what a `--print-config` flag shows.

The two latter folders are read by their own flags rather than by `--config`:
`train --curriculum social_navigation` reads `curriculum/social_navigation.yaml`,
and `benchmark --suite basic` reads `benchmarks/basic.yaml`. They sit here so
that everything a study tunes lives in one place; `configs/benchmarks/basic.yaml`
is also a plain list of the six basic navigation scenarios, which is a
convenient way to name them from a script.

**Writing your own.** Copy the closest file, rename it, and edit the values.
Add it under `algorithms/` to name it beside the built-in rules, or under
`methods/` to keep it with the methods.

**Requirement.** YAML is read with PyYAML, which is already installed with
`rosbags`. A JSON configuration needs nothing beyond the standard library. If
PyYAML is missing, reading a `.yaml` file reports it; `.json` still works.
