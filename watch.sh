#!/usr/bin/env bash
ls js/* crate/*.toml crate/src/*.rs |
entr -cs "rustfmt crate/src/*.rs; npm run build"
