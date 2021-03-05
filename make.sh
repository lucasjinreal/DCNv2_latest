#!/usr/bin/env bash
rm *.so
rm -r build/
python3 setup.py build develop
