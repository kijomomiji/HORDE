# The following codes are used for HORDE's performance in static environments.

The basic construction of these codes is:

just mine-CE_plus_sample [dataset] [version] [workload] [sample_size] [hidden_size] [training_epochs] [batch_size]

-----------------------------------------------------

`just mine-CE_plus_sample census13 original base 500 32 100 200`

`just mine-CE_plus_sample forest10 original base 3000 256 100 200`

`just mine-CE_plus_sample power7 original base 5000 256 100 200`

`just mine-CE_plus_sample dmv11 original base 10000 512 100 200`