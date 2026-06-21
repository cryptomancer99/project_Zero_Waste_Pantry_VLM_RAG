# AI Pantry Inspector

A cross-platform desktop application designed to track pantry components, count fruits and vegetables, and parse out stale or damaged items automatically using local edge AI models.

## Features
- **Automatic Multi-Image Parsing**: Scans whole directories for standard media extensions.
- **Initial Inventory Isolation**: Extracts granular product definitions via a VLM.
- **Freshness Control Filtering**: Removes damaged stock dynamically from the final usable view tables.

## Quick Start

### 1. Installation
Ensure you have Python 3.10+ installed. Clone this repository and install dependencies:

```bash
pip install -r requirements.txt