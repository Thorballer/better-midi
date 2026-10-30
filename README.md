# Better MIDI — piano audio transcription

An original PyTorch piano audio-to-MIDI notebook with convolution/attention-based modeling, onset/frame/velocity prediction, note decoding, and full-song chunk-and-stitch inference.

## Original notebook

- [Open the notebook](notebooks/better_midi_%281%29.ipynb)
- Recovered from [Alain-Cycling at its pinned source commit](https://github.com/Thorballer/Alain-Cycling/blob/f7eb204a12b68283b093fbdb089f64708bd656c6/better_midi_(1).ipynb).
- Notebook source, outputs, and metadata are preserved byte-for-byte. This is **not** the later assistant-modified `piano-transcription` project.
- [PROVENANCE.json](PROVENANCE.json) records the source commit and SHA-256 hash. The original repository remains unchanged.

## Recorded result

The saved notebook output reports the following for **one randomly selected full song**, processed in 40 chunks:

| Metric | Recorded value |
|---|---:|
| Note pitch-and-onset precision | 59.13% |
| Note pitch-and-onset recall | 69.12% |
| Note pitch-and-onset F1 | **63.74%** |
| Reference notes | 9,525 |
| Predicted notes | 11,134 |
| Recording duration | 1,196.2 seconds |

Recording: `MIDI-Unprocessed_Recital9-11_MID--AUDIO_11_R1_2018_wav--2`, read from `2018.h5`.

The evaluator calls `mir_eval.transcription.precision_recall_f1_overlap` with MIDI pitches converted to Hz, `onset_tolerance=0.05`, and `offset_ratio=None`. Therefore **note endings are not scored**. This is a retained historical result, not a newly reproduced measurement, whole-test-set average, or verified held-out score. The notebook does not establish independence of this recording from training. Do not describe it as “63.74% overall accuracy” or as onset-and-offset F1.

## Running the original experiment

1. Open the notebook in an appropriate GPU-enabled notebook environment. It uses CUDA and bfloat16 autocast.
2. Inspect setup/download and filesystem-writing cells before running them. Supply the referenced MAESTRO HDF5 files and any cache directories; the data and trained weights are not bundled.
3. Install the imported libraries listed in `requirements.txt`, adapting versions for the original notebook environment. This file is a dependency inventory, not a tested lockfile.
4. Adjust hardcoded `/kaggle/input`, `/home`, and cache paths, then run the intended cells in order. Some cells are experimental training stages, not independent command-line entry points.
5. Full-song inference writes a predicted `.mid` and a reference `_actual.mid`; the final cell saves the model. These generated artifacts are not recovered in this repository.

The original model code has not been rewritten or retrained during portfolio organization. Notebook structure, transformed-cell syntax, credential-pattern checks, and source hashes were checked; that does not prove runtime compatibility or independently reproduce the score.

## Attribution and data

Daniel Goldberg’s original notebook is archived from his GitHub account; documentation was added during portfolio organization. The precise human/AI/tool contribution history is not reconstructed here. The source repository’s Apache-2.0 LICENSE is retained. Dependencies and MAESTRO data have their own licenses; no audio, dataset archive, or model checkpoint is redistributed in this repository.
