# nondeterminismwtf

```console
mamba env create -f environment.yml --force
```

```console
mamba activate nondeterminismwtf
```

```console
python -m pytest tests/
```

With Nightly Pytorch
----------

```console
conda install -y pytorch-nightly::pytorch -c pytorch-nightly
```

```console
python -m pytest tests/
```
