"""Dataset adapters and shared end-to-end reconstruction orchestration."""

# Default reference order for the two cichlid adapters and empirical simulation
# templates. General reconstruction uses the input header, not this species list.
CICHLID_AUTOSOMES = tuple(f"chr{index}" for index in (*range(1, 21), 22, 23))
