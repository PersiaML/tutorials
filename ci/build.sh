#!/bin/bash

set -euo pipefail

wget 'https://github.com/rust-lang/mdBook/releases/download/v0.4.12/mdbook-v0.4.12-x86_64-unknown-linux-gnu.tar.gz'
tar xzf mdbook-v0.4.12-x86_64-unknown-linux-gnu.tar.gz

wget 'https://github.com/Michael-F-Bryan/mdbook-linkcheck/releases/download/v0.7.4/mdbook-linkcheck.v0.7.4.x86_64-unknown-linux-gnu.zip'
unzip -n mdbook-linkcheck.v0.7.4.x86_64-unknown-linux-gnu.zip

chmod +x mdbook-linkcheck

env PATH=.:$PATH mdbook build
