# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a time-series foundation model fine-tuning project using PatchTCT on the Smart Buildings Dataset. The project focuses on fine-tuning a pre-trained time-series foundation model for building energy management and temperature prediction tasks.

## Dataset and Architecture

**Dataset**: Smart Buildings Dataset from Google Research
- Building data from 2022-2024 with temperature sensors and control actions
- Training data: Jan-June 2022 (2022_a partition)
- Validation data: July-December 2022 (2022_b partition)
- Data includes observation matrices, action matrices, floorplans, and device layouts
- Target prediction: zone air temperature sensors
- Exogenous variables: all non-temperature sensor data

**Model**: PatchTCT (Patch-based Time-series ConvTransformer)
- Pre-trained time-series foundation model for fine-tuning
- Designed for multivariate time-series forecasting tasks

## Key Files

- `dataset_definition.py`: Contains SmartBuildingsDataset class for data loading and preprocessing
  - Downloads data from Google Cloud Storage (sb1.zip)
  - Splits data into training/validation partitions
  - Separates temperature targets from exogenous observations
  - Provides floorplan and device layout information

## Data Structure

**Training Data**:
- `data['observation_value_matrix']`: Sensor observations
- `data['action_value_matrix']`: Control actions
- `metadata["observation_ids"]`, `metadata["action_ids"]`: Variable mappings
- `floorplan`, `device_layout_map`: Spatial information

**Validation/Prediction**:
- `temp_data`: Target temperature sensor values to predict
- `exogenous_observation_data`: Non-temperature sensor inputs
- `initial_condition`: Starting temperature values

## Development Notes

- Python-based project using numpy, requests, pickle for data handling
- No build system or testing framework currently configured
- Focus on time-series model fine-tuning and evaluation
- Data is downloaded automatically when SmartBuildingsDataset is instantiated