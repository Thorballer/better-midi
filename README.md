# Better MIDI

Piano audio-to-MIDI transcription in PyTorch, with convolution/attention layers, onset/frame/velocity prediction, and full-song inference in 30-second chunks.

[Open the original notebook](notebooks/better_midi_%281%29.ipynb).

## Recorded result

The saved output reports **63.74% note pitch-and-onset F1** on one full song:

| Precision | Recall | F1 |
|---:|---:|---:|
| 59.13% | 69.12% | 63.74% |

The evaluator uses `mir_eval`, a 50 ms onset tolerance, and ignores note endings (`offset_ratio=None`). The recording contains 9,525 reference notes; the model predicts 11,134 notes across 1,196.2 seconds.

This is a historical single-song result, not a test-set average or a newly reproduced score. The notebook does not establish whether the recording was excluded from training.

## Running

1. Open the notebook in a CUDA GPU environment with bfloat16 support.
2. Install the packages in `requirements.txt` and supply the referenced MAESTRO HDF5 data. Package versions are not pinned; data and trained weights are not included.
3. Review setup cells, adjust `/kaggle/input`, `/home`, and cache paths, then run the relevant cells in order.

Inference writes predicted and reference MIDI files; the final cell saves the model. These generated files are not included.

## Source

Daniel Goldberg's notebook, preserved byte-for-byte with its outputs from [Alain-Cycling](https://github.com/Thorballer/Alain-Cycling/blob/f7eb204a12b68283b093fbdb089f64708bd656c6/better_midi_(1).ipynb). See [PROVENANCE.json](PROVENANCE.json) for the pinned source and hash. Documentation was added for this archive; training was not rerun.

The source repository's Apache-2.0 license is retained. MAESTRO and dependencies have separate licenses.
