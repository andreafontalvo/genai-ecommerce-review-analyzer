# GenAI E-Commerce Reviews Analyzer

The following use case presents a tool for analysis of product reviews from the Women’s E-Commerce Clothing Reviews dataset from Kaggle. The objective of this tool is to transform unstructured customer feedback into concrete insights for product stakeholders decision making processes.

The dataset contains real, anonymized customer reviews. The tool analyzes a sample of reviews for a specific product. First, a sentiment analysis is performed to determine overall positive or negative feedback using a pre-trained binary classification model from Hugging Face. Next, pros and cons are extracted from the reviews associated with a selected product using a chunk-based analysis approach. 

Finally, all unique insights are aggregated to produce an overall evaluation, along with recommendations and potential product improvements. Through the Amazon Bedrock API, an LLM is used as the inference engine in a text-only setup, leveraging the Amazon Nova Micro model. The following diagram shows the main stages of the tool's logic.

## Requirements

- Docker
- Docker Compose
- Amazon Bedrock API Key

## Setup

1. Clone repository
2. Add dataset to `data/`
3. Create `.env` file
4. Build and run

```bash
docker-compose build
docker-compose up