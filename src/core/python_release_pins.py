"""The standalone interpreter this build fetches, and the digests for it.

Three pinned facts, kept together because they only ever move together: the
release, the patch version taken from it for each Python minor version, and
the SHA256 of every archive a download can ask for.

They live apart from python_manager so the code that fetches, verifies and
extracts an interpreter reads as code, and so a release bump is a diff in one
small file rather than a hundred lines buried in a thousand.

The server may correct all three (see install_config). What it cannot do is
move the release or a version without the digests to go with them:
python_manager holds every override against this table before a byte is
fetched, and refuses one nothing here describes.
"""
from __future__ import annotations

# Release tag from python-build-standalone
# Update this periodically to get newer Python builds
RELEASE_TAG = "20251014"

# Mapping of Python minor versions to their latest patch versions in the release
PYTHON_VERSIONS = {
    (3, 9): "3.9.24",
    (3, 10): "3.10.19",
    (3, 11): "3.11.14",
    (3, 12): "3.12.12",
    (3, 13): "3.13.9",
    (3, 14): "3.14.0",
}

# SHA256 of every standalone-Python archive get_download_urls() can request,
# copied from the release's official SHA256SUMS. Public integrity checks, not
# secrets. Covers all PYTHON_VERSIONS x every platform string _get_platform_info()
# emits x both variants (install_only_stripped preferred, install_only fallback).
# MUST be regenerated in the same commit as any RELEASE_TAG or PYTHON_VERSIONS
# bump: the download fails closed on a mismatch or a missing entry, so a stale
# or partial table would brick installs on the affected platform.
PYTHON_STANDALONE_SHA256 = {
    "cpython-3.9.24+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "6292c6484ab2c96c80116f4bdb3da638d816206fe11a102e83787a2f75591b94",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-apple-darwin-install_only.tar.gz": "6b65213e639e91eb8072db80ed9c140d769af1d5e0386efd8f153449c3694714",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "eafceb263d9507ff0052ae9d6f1c415bb99299dcb202a931865b8ca044a5e40e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-apple-darwin-install_only.tar.gz": "14beda9465feb6991f73d6f6cb9e69afc576c5cac8c185bd729f491aa4305bfb",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "fc9b3af198bdc85ff532eade79825d18b4a4d4036caf8f895922e97e3378c642",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "a2fdaf290361386396bbfaa08e13fc2b88e1149f870adf18836e262c609406db",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "4a5faa90b3f76b235f2706be501605fc8f57e4f1e2c6c596e6fb328639e0d65a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "d840efd9d81ad557019ebd0d435828fc32101cd01be82046087b4aee463dca0c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "0339804b69bc00d5dde58c6694174c8e97e6f16c8ace90fe9a1b1a15456ac510",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.9.24+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "866745efbee219a3f9b9d54ee1477ebf92542bb9ff9f6591a7e5a3643a0d4214",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "37931bdcd24496bf57415e34f93dcf360f80b6a2b5bf91d32ceecde14fe9f29f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-apple-darwin-install_only.tar.gz": "06cfdfa8966dfd86204d45c6a241dd37cb0b3ede90986591fc0b0dbe576848de",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "82703c1d9de3b6b686269361dd61c29aa65f52d04dbf0c4f53fc6fd8faf38dfd",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-apple-darwin-install_only.tar.gz": "b4e0c82f350f18a8fb1b1982f03c1c90aaba5d9ab74fe6ede9896306f64a287c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "47ffefa9240d7354c086e9eb84e917d2460c6ae2a719281337218a2a3c83e4cf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "e2d9193b2d2fd99fac3fb90eda216100b64cd7cf14f291d9425436ea9b1eaa04",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "6ea9ff46ae3e0eb551558754c78a41cb90968b1950e1a2c716e339e6264bcc96",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "c4c760f49dbba10a0f91b2fd52c847dd50cbe7cb8cb19bb7598c4dc38a358e9c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c0b09b744293f2aad85b1ef84544f8a7ba383675b29d1f7efd1e96bb9984399f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.10.19+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "85c96114de83d783db18137f3858bcd3b5a9c4cbe9053f0072d7b5f52154a8c9",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "cd54cc868a9b1056fbd4654509431f402f0365329618e2583b60c82f73da4e56",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-apple-darwin-install_only.tar.gz": "99d98bf73d9906d18a9184054a328288ede2cb4a2d245a05411a28e8d023aab6",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "a1eb602d2bbbfdbba005c54334b33779de8e0f2225f1d5e03c7a1e3e95cb822e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-apple-darwin-install_only.tar.gz": "d234fa6518634daf3aa812895ec757d0e0b1fea3335fd0c5038d4e2bcc5d7ee5",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "c2a7b0bb86eb9f1cc094a01bdabef7ddc77f89f8e45161fa7819f2b4a7ba7bc1",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "80022423ca581c88d5bb7beb889f10c12d3d8d2e5cc6422fd2b060b52e45aa05",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "b59afc432a64df8fdcfeab5bca98e66c7272cd3e6bc3611b9b48996f714ae15a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "8b033614f3a6969d86c20f9b823277ee8e1f72788307c082a44d2ad4cc856e2b",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "91ab434738647ab45d630aaa02e3808bb516239beeb52f7799c12a12d1557a38",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.11.14+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "d0623c777fb89b904b56cd5aba51af29cbb34b1f9d45f0672f90f6dce30fa93e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "84cb7acbf75264982c8bdd818bfa1ff0f1eb76007b48a5f3e01d28633b46afdf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-apple-darwin-install_only.tar.gz": "6ceba34fe78802853a30bde6f303a0a54f71f6ab07a673da34e90c0aa06c786e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "f76a921e71e9c8954cccd00f176b7083041527b3b4223670d05bbb2f51209d3f",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-apple-darwin-install_only.tar.gz": "9b8589eefb153cbe7cb652993d0ecc94aeb2fa13c1a2e8bc240f5f74f23bb21b",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "3c8b9b10a933909c98b9916297e2093b24a9c2abaa23df1c2622c2bfe052cb94",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "2d670beb3b930d30e3a13cc909923a001dbdfcb5537692d5da40b6b41643ce1c",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "d2a6c0d4ceea088f635b309a59d5d700a256656423225f96ddfb71d532adb1aa",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "d32487b853d6f5709019a471770be5e5d3e6bd2ac507e5629e2d6825565d3e71",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c74addcd1b033a6e4d60ead3ab47fcc995569027e01d3061c4a934f363c4a0cf",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.12.12+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "1ab2b6594d1c3d76cbebea09d6bc3e6ba68d8eb3b6322080375c4cc3dd188f34",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "52721745b0fa3196e4d0381fa5c06dda1d54343b90d49d90c3bba52d1171bd98",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-apple-darwin-install_only.tar.gz": "931db8f735e18700d4eab9ee39dbbd0b4c114d7d039dd2707b2d932ded039698",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "7c33b153a69c6255e6f2659cf39738f316b03969d6230d7bc47c73b7fde9a0d4",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-apple-darwin-install_only.tar.gz": "9f6bc3c15e2f9e2c9c90db2c8b3ee94598e777789f8aea6e36b69ae55d007d01",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "76e0a9749c4deeb975a4b6b36d54be4e43f0c2a4c654bedab5d2e4d62dbc3006",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "8b0efc2674bb293ce2d423d59765b1ca3a2d80dc0ca6168f6279cb569e72b55e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "baa3d107d17e4328448e30c3c9c83cca0eb41ca7a37c10982e14d46a5c3db07d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "c86606a45fb6540b1b66d9c52c6f5466fba8affb29acb9ab6a0b7f5ad54e588a",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "c0ccda275948c79996e46993c2c5476ff5cca606dee530f1dea712179131b348",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.13.9+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "b4b0204658930337c85c321b49ed2585fe544097a72bc76dcf0b77e49fff8473",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-apple-darwin-install_only_stripped.tar.gz": "057476264b07222a2baeff68a733647f91a9d61c94f79beba46a44eb42101749",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-apple-darwin-install_only.tar.gz": "1333ce2807fbea673eb242edbf4997ea1e2f6cbc01cd80dec1f9d19de2cd63ed",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-apple-darwin-install_only_stripped.tar.gz": "56dcb0cdafabac9d6d976690fb05d9ee92d20ce798c3aabe9049259ebe7d3e0d",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-apple-darwin-install_only.tar.gz": "0a4cc33ca56830b92545950aacdde8925c9d4259e4f00ceda04fedf853f70679",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-pc-windows-msvc-install_only_stripped.tar.gz": "b064fca740da03dbae1bad7f73fcaabbc76681ad635b9897ed3808c3eecff122",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-pc-windows-msvc-install_only.tar.gz": "d90e97fe69b819f0a776cd665d06fef6526a4259211d11f00e501688659f1c0e",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-unknown-linux-gnu-install_only_stripped.tar.gz": "7dbb43b742c040835a277318355fb359b41e509dbf4fbb614da38005a9290e16",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-aarch64-unknown-linux-gnu-install_only.tar.gz": "e613f44e60227b3423a994698426698569e055c24447c10dd9c1c022cf511f05",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz": "493c477b4a88bb1ea2f6c6f57fa0e88ffbe55d9e7b1405c4699f2d41c04eb154",  # noqa: E501  # pragma: allowlist secret
    "cpython-3.14.0+20251014-x86_64-unknown-linux-gnu-install_only.tar.gz": "74d4516a64abc63ae4bcbffb35482879a85b7faa187fcfa47c1ca8f00faebf5f",  # noqa: E501  # pragma: allowlist secret
}
