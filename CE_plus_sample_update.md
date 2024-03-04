Basic structure:

just mine-CE_plus_sample_update [dataset] [version] [workload] [epochs] 

```bash
# [dataset] could be set as census13/forest10/power7/dmv11
# [version] could be set as original+original_cor_0.2 where the rate could be 0.2/0.4/0.6/0.8/1.0
# an example:
just mine-CE_plus_sample_update census13 original+original_cor_0.2 base 40 200
```
Commands:
```bash
# census13
just mine-CE_plus_sample_update census13 original+original_cor_0.2 base 40 200

just mine-CE_plus_sample_update census13 original+original_cor_0.4 base 40 200

just mine-CE_plus_sample_update census13 original+original_cor_0.6 base 40 200

just mine-CE_plus_sample_update census13 original+original_cor_0.8 base 40 200

just mine-CE_plus_sample_update census13 original+original_cor_1.0 base 40 200
```

```bash
#forest10
just mine-CE_plus_sample_update forest10 original+original_cor_0.2 base 40 200

just mine-CE_plus_sample_update forest10 original+original_cor_0.4 base 40 200

just mine-CE_plus_sample_update forest10 original+original_cor_0.6 base 40 200

just mine-CE_plus_sample_update forest10 original+original_cor_0.8 base 40 200

just mine-CE_plus_sample_update forest10 original+original_cor_1.0 base 40 200
```

```bash
# power7
just mine-CE_plus_sample_update power7 original+original_cor_0.2 base 40 200

just mine-CE_plus_sample_update power7 original+original_cor_0.4 base 40 200

just mine-CE_plus_sample_update power7 original+original_cor_0.6 base 40 200

just mine-CE_plus_sample_update power7 original+original_cor_0.8 base 40 200

just mine-CE_plus_sample_update power7 original+original_cor_1.0 base 40 200
```

```bash
# dmv11
just mine-CE_plus_sample_update dmv11 original+original_cor_0.2 base 40 200

just mine-CE_plus_sample_update dmv11 original+original_cor_0.4 base 40 200

just mine-CE_plus_sample_update dmv11 original+original_cor_0.6 base 40 200

just mine-CE_plus_sample_update dmv11 original+original_cor_0.8 base 40 200

just mine-CE_plus_sample_update dmv11 original+original_cor_1.0 base 40 200
```
