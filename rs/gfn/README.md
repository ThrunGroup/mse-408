# Setup

1. Download the [libtorch](https://pytorch.org/get-started/locally/) C++ library.
2. Extract `unzip libtorch.zip -d /opt/`
3. Update environment by adding the following lines to `~/.bashrc` or equivalent:

```bash
export LIBTORCH=/opt/libtorch
export LD_LIBRARY_PATH=${LIBTORCH}/lib:$LD_LIBRARY_PATH
```

4. Install rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
5. Run: `cargo run`
