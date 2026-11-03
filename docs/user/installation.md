# Installation

The supported platform is **Linux x86_64 on CPUs**. A GPU is not required or used.
Use an isolated Python environment and the wheel or source archive supplied with
the release. Package metadata requires Python 3.14 or later. The tested reference uses Python
3.14.8; the [generated environment reference](../reference/environment.md) lists
the exact dependency set. Later combinations require validation.

```sh
python3.14 -m venv .venv
. .venv/bin/activate
python -m pip install ./dist/haplotype_reconstruction-4.2.0-py3-none-any.whl
haplotypes --help
```

To install the source distribution instead, select its release archive:

```sh
python -m pip install ./dist/haplotype_reconstruction-4.2.0.tar.gz
```

These commands install an existing release artifact; replace the filename with the
artifact you downloaded. The source distribution includes the documentation and
small validation fixtures. The wheel installs the Python package and the
`haplotypes` command. `python -m haplotype_reconstruction` exposes the same CLI.

The release's `requirements-tested.txt` and installed package metadata are the
machine-readable dependency records. The dependency envelope is tested as a set,
rather than treating arbitrary combinations as interchangeable. Numerical-library
versions form part of a run's numerical-policy identity.

The first computation may compile kernels. The cache is outside the installed
package; it does not require write access to `site-packages`. An unavailable cache
falls back to compilation. Use the documented execution settings for CPU and
scratch placement; scientific behaviour is configured through TOML, not environment
variables. See [runtime controls](../reference/environment.md).

Continue with the [quick start](quickstart.md). For an existing 3.5 installation,
read the [migration guide](migration.md) before opening a checkpoint directory.
