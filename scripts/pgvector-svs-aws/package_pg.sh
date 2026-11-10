#!/usr/bin/env bash
# Package a PostgreSQL + pgvector + svs install built with the SVS extension's
# docs/build_guide on Ubuntu 22.04, for a server running a newer Ubuntu.
# Bundles libsvs_c_api and ICU 70 (which newer Ubuntu releases don't ship);
# the server needs `apt install libgomp1`. PostgreSQL installs are
# relocatable, so the tarball can be unpacked anywhere.
#
# Usage: package_pg.sh <build_guide dir> <output.tgz>
set -euo pipefail

guide=$(cd "$1" && pwd)
out=$(realpath "$2")
stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT

mkdir "$stage/pgsvs"
cp -a "$guide/pgsql_install" "$stage/pgsvs/"
cp -a "$guide"/svs_install/lib/libsvs_c_api.so* "$stage/pgsvs/pgsql_install/lib/"
for lib in libicuuc.so.70 libicui18n.so.70 libicudata.so.70; do
    cp -L "/lib/x86_64-linux-gnu/$lib" "$stage/pgsvs/pgsql_install/lib/"
done
git -C "$guide/../.." rev-parse --short HEAD >"$stage/pgsvs/SVS_EXTENSION_COMMIT"
tar -C "$stage" -czf "$out" pgsvs
echo "Wrote $out"
