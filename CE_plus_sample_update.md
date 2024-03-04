Basic structure:

just mine-CE_plus_sample_update [dataset] [version] [workload] [epochs] 

```bash
# [dataset] could be set as census13/forest10/power7/dmv11
# [version] could be set as original+original_cor_0.2 where the rate could be 0.2/0.4/0.6/0.8/1.0
# an example:
just mine-CE_plus_sample_update census13 original+original_cor_0.2 base 40 200
```
