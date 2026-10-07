# LatticeBoltzmann3DML
 machine learning off 3d lattice boltzmann runs

## Resuming `latticeboltzmanngraph.py` training

The script writes `cfd_flow_predictor_gnn.pt` next to the script, so its location
does not depend on the current working directory. It saves a checkpoint after
each completed epoch, including model weights, output normalization, optimizer
state, and the completed epoch number.

The script defaults to `mode = "resume"`. Set `mode` in
`latticeboltzmanngraph.py` to:

- `"train"` to start from scratch and replace the existing checkpoint.
- `"resume"` to continue from the most recently completed epoch. Any interrupted
  epoch is run again from its beginning. Legacy weights-only checkpoints are
  loaded as starting weights; because they lack optimizer and epoch data, resume
  starts with a fresh optimizer at epoch 1 and upgrades the file to the new
  checkpoint format.
- `"load"` to load model weights for prediction without training.

Training prints the current geometry and radius while each sample is being
processed. Set `"train"` explicitly if you want to recalculate output
normalization and start with a fresh model.
