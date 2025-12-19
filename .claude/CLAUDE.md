# Zumi Pipeline FastAPI Project

See @CODE_PRACTICES.md for comprehensive coding standards and principles.

## Project Overview

FastAPI-based distributed hardware data capture system with orchestration, motor control, and GoPro integration.

## Key Components

- `orchestrator.py` - Central orchestration and user interaction
- `node_motor.py` - Motor node (DM3510) implementation
- `node_gopro.py` - GoPro camera integration
- `validator.py` - Data validation logic
- `zumi_config.py` - Centralized configuration
- `zumi_core.py` - Base node HTTP service

## Development Commands

- Run tests: `pytest`
- Run specific test: `pytest tests/test_orchestrator.py -v`
- Start orchestrator: `python orchestrator.py`
- Start motor node: `python node_motor.py`
- Start GoPro node: `python node_gopro.py`

## Data Directory Structure

```
data/
└── {run_id}/
    ├── {run_id}_ep001_*.MP4
    ├── {run_id}_ep001_*_motor.npz
    └── {run_id}_ep001_*_imu.json
```
