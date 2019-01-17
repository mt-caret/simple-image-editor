#!/usr/bin/env bash
ls js/* crate/*.toml crate/src/*.rs *.html |
entr -cs "rustfmt crate/src/*.rs; npm run build"
