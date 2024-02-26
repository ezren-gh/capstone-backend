# GLIP/DesCo Server Backend

This project serves the following models using FastAPI:

## Models

1. **GLIPv2**: GLIPv2 is a versatile model that excels in both localization tasks, such as object detection and instance segmentation, and Vision-Language understanding tasks like VQA and image captioning. Through innovative pre-training tasks, GLIPv2 streamlines the VLP process, achieving near state-of-the-art performance across a range of tasks while demonstrating strong adaptability and superior grounding capability.
2. **DesCo GLIP/FIBER**: DesCo is a novel approach in vision-language learning that addresses the limitations of current models by leveraging large language models for generating rich descriptions and designing context-sensitive queries. By focusing on contextual information rather than just object names, DesCo achieves significant improvements in object detection performance, surpassing prior state-of-the-art models on benchmarks like LVIS and OmniLabel.

## Usage

The server can be built and served using docker (gpu required), follow these steps:

1. Clone the repository to your local machine.
2. Build the application with `docker build -t <repository_name> .`
3. Run the FastAPI server and forward ports using `docker run --rm --gpu all -p 8080:8080`.
4. Access the models through the provided API endpoints.

## API Endpoints

- **glipv2**: `/detection/base`
- **DesCo GLIP**: `/detection/glip`
- **DesCo FIBER**: `/detection/fiber`
