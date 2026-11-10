# Focused confirmatory protocol for fractional limb-gap completion

## Development finding and scope change

The original aggregate development gate across limb48, limb72, and mixed masks
was not met. Across training seeds 20260829--20260831, the joint-adaptive
fractional bridge improved over the parameter-matched finite-window bridge by
only 0.0756% on the aggregate score. The method must therefore not be described
as a general missing-motion repair improvement.

The same frozen development runs showed a narrower replicated finding for
72-frame localized limb outages: relative MPJPE gains were 0.2815%, 0.2808%,
and 0.2983%. The focused confirmatory question is whether this longest
partial-limb result transfers to the untouched choreography-held-out test set.

## Frozen test protocol

- Split seed: 20260829 (1,002 train, 185 validation, 221 test sequences).
- Training seeds: 20260829, 20260830, and 20260831.
- Test window length: 128 native 60 Hz frames.
- Test stride: 128; all available fixed windows are used.
- Test dataset seed: 20260829.
- Test mask seed: 20270829, fixed across methods and training seeds.
- Primary endpoint: MPJPE under the limb72 mask.
- Secondary endpoints: velocity error and bone-relative error under limb72;
  MPJPE under limb48 and mixed masks.
- Scope stress tests: temporal32 and temporal48. These cannot support a
  positive general-completion claim.
- Models: original DanceRouter, parameter-matched finite-window bridge, and the
  frozen joint-adaptive fractional bridge with the fixed fractional trajectory
  loss used during training.

No architecture, training schedule, checkpoint selection, routing threshold,
or mask definition may change after this protocol is written.

## Confirmation rule

The focused limb72 hypothesis is supported only if all of the following hold:

1. Fractional-versus-finite MPJPE gain is positive for all three training
   seeds.
2. The gain after averaging seed predictions at the sequence level is at least
   0.2%.
3. The 10,000-resample paired sequence-bootstrap 95% interval for relative
   gain excludes zero.
4. No secondary partial-body condition regresses by more than 1.5%.

Failure leaves the Fractal and Fractional submission route unsupported. Test
results will be retained and reported without further model tuning.
