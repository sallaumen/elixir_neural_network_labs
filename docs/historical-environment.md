# Historical benchmark environment

The original repository recorded these versions before the modernization work:

| Component | Historical version | Source in the original repository |
| --- | --- | --- |
| Erlang/OTP | 24.3.4.1 | `.tool-versions` |
| Elixir | 1.13.4-otp-24 | `.tool-versions` |
| Python | 3.8.0 | `.tool-versions` |
| Bazel | 3.7.2 | `.tool-versions` |
| Nx | 0.2.1 | [`historical/mix.lock`](historical/mix.lock) |
| EXLA | 0.2.2 | [`historical/mix.lock`](historical/mix.lock) |
| Scidata | 0.1.8 | [`historical/mix.lock`](historical/mix.lock) |
| Axon | Git commit `b086784875eb6231672a47cca6e95d8c3b81cdb8` | [`historical/mix.lock`](historical/mix.lock) |

This is a record of the original project configuration, not proof that every checked-in figure used every listed version. The figures lack a complete run log with commit, hardware, dataset checksum, and timing protocol. The modern `.tool-versions` and root `mix.lock` describe the maintained code and should be used for new runs.

Model definitions and data handling also changed while migrating from Nx/EXLA 0.2 to 1.0: the CIFAR-10 model now explicitly uses channels-first operations, the default batch size is 16, and evaluation uses Scidata's held-out test split. Do not compare new metrics directly with historical screenshots.
