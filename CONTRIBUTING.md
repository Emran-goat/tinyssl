# Contributing to TinySSL

We welcome contributions that make TinySSL better for everyone.

## How to Contribute

1. **Fork** the repository and create a branch from `main`.
2. **Make your changes** — code, documentation, or bug fixes.
3. **Add tests** for any new functionality.
4. **Update documentation** if you change public APIs or add features.
5. **Run the test suite** to confirm nothing breaks:

   ```bash
   python -m pytest tinyssl/tests/
   ```

6. **Open a pull request** with a clear description of what you changed and why.

## Code Style

- Type hints for all public APIs with `mypy-strict` where practical
- Follow PEP 8 with 88-character line width
- Use black and ruff for formatting

## Reporting Issues

Open a GitHub issue with:
- A minimal reproduction (code snippet + traceback)
- Your environment (OS, Python version, PyTorch version)
- Expected vs actual behavior

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
