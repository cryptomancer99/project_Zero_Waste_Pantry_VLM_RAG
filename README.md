# Zero Waste Pantry Agent (VLM + RAG)

An intelligent, edge-AI driven pantry assistant that utilizes Vision-Language Models (VLMs) to visually scan and audit inventory from images, filters out stale or damaged stock, and leverages Retrieval-Augmented Generation (RAG) to recommend the best-fit recipes based on what's available.

## Features
- **Visual Inventory Detection**: Scans images in `pantry_images/` using a local VLM to accurately extract and isolate product definitions and item counts.
- **Freshness & Quality Control**: Automatically filters out damaged, expired, or stale stock dynamically from final usable views.
- **Context-Aware Recipe RAG**: Cross-references your fresh, available ingredients with a local vector database to suggest recipes, minimizing kitchen food waste.

## Project Structure
```text
Zero_Waste_Pantry_VLM_RAG/
├── data/             # Vector store indices and recipe knowledge bases
├── models/           # Local VLM configurations and weights
├── pantry_images/    # Input directory for folder/shelf snapshots
├── src/              # Main application source code logic
├── README.md         # Project documentation
└── requirements.txt  # Python package dependencies
