#!/bin/bash
python gec_to_m2.py
python m2_to_spellcorrection.py -corr ./extract/test.bea60k -incorr ./extract/test.bea60k.noise -id 0 ./extract/test.bea.noise.m2
