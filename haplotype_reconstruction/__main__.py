"""Expose the same lazy command parser through python -m haplotype_reconstruction."""

if __name__ == "__main__":
    from .interface.cli import main

    raise SystemExit(main())
