Copy and compile deeply nested built-in SDF trees without recursive preparation
walks. Preserve private geometry snapshots, shared subtrees, distance analysis,
and explicit free-parameter gradients. Reject cyclic containers with a clear
error. Cover mixed geometry through 600 nested operations with float32 JAX
evaluation and gradients, without raising Python's recursion limit.
Keep inference's early exits for unanalysed lofts and smooth operations so large
polygon profiles do not incur unnecessary distance-domain proofs.
