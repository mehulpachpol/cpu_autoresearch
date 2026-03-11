## Optimization Summary

The training pipeline in [train.py] was iteratively optimized under the guardrails defined in [program.md] only `train.py` could be changed, `prepare.py` remained untouched, and every experiment had to fit within the fixed 60-second CPU training budget.

The process started from a very small MLP baseline, which reached `86.17%` Final Validation Accuracy. The largest improvement came from replacing that model with a compact CNN that preserved compatibility with the existing evaluation path. That change raised validation accuracy to `96.64%`. Adding batch normalization improved convergence further and increased the score to `97.66%`. Switching the optimizer to `AdamW` with light weight decay produced another gain, reaching `98.12%`.

A second optimization pass focused on training dynamics rather than raw model size. Adding a `OneCycleLR` learning-rate schedule on top of the batch-normalized CNN + `AdamW` configuration improved the best score again to `98.59%`, which became the final retained result.

Several additional ideas were tested and explicitly rejected because they did not beat the current best:

- Label smoothing reduced accuracy to `97.89%`
- A deeper CNN with global average pooling dropped to `83.05%`
- Evaluation-time test-time augmentation scored `98.12%`
- Using `optimizer.zero_grad(set_to_none=True)` scored `98.44%`

## Final Best Configuration

The final retained model is:

- A small CNN instead of the original MLP
- Batch normalization after each convolution
- `AdamW` optimizer with light weight decay
- `OneCycleLR` scheduling during the 60-second training window

## Best Result

- Baseline MLP: `86.17%`
- CNN: `96.64%`
- CNN + BatchNorm: `97.66%`
- CNN + BatchNorm + AdamW: `98.12%`
- CNN + BatchNorm + AdamW + OneCycleLR: `98.59%`

## Git History

The best-performing checkpoints were ratcheted into git:

- `fbb36f9` - `feat: batchnorm cnn with AdamW - acc: 98.12`
- `7a72268` - `feat: add one-cycle AdamW schedule - acc: 98.59`
