#!/bin/bash

cd /workspaces/ML_Project Credit_ScoreCard/src
pip install --upgrade pip setuptools wheel\
	    && pip install -e ".[dev]"
