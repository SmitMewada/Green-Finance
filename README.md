# Green Finance AI
## Personal Carbon Footprint Advisor for Financial Transactions

### Overview

Green Finance AI is an end-to-end fintech application that automatically tracks the carbon footprint of every financial transaction and provides actionable insights for carbon offsetting and greener spending choices. Built as a prototype combining fintech and climate tech, this project demonstrates real-time transaction classification, emissions calculation, and offset recommendations using open-source AI models.

### Key Features

- **Real-time Transaction Classification**: Uses a fine-tuned Qwen-3B model with QLoRA to categorize banking transactions into meaningful buckets (Groceries, Fuel, Restaurants, etc.)
- **Carbon Footprint Calculation**: Converts spending amounts to kg CO₂e using merchant-level emission factors from DEFRA, USEEIO, and other authoritative sources
- **Hybrid AI Pipeline**: Three-layer classification system (regex/MCC → fast model → fine-tuned model) optimizing for both speed and accuracy
- **Offset Recommendations**: Suggests cost-effective carbon offsets and preventive alternatives for high-emission purchases
- **Privacy-First**: Runs entirely on local/free infrastructure with browser-fallback models

### Architecture

```
[Plaid/Open Banking] → [Transaction Ingestion] → [Classification Pipeline] → [Emission Factors] → [Dashboard/API]
                           ↓
[Postgres Database] ← [QLoRA Fine-tuned Qwen-3B] ← [Baseline Regex/MCC] ← [Transaction Normalization]
                           ↓
[FastAPI Backend] → [Carbon Calculation] → [Offset Recommendations] → [User Interface]
```

### Tech Stack

- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML/AI**: Hugging Face Transformers, PEFT (QLoRA), bitsandbytes
- **Data Sources**: Plaid API (sandbox), DEFRA emission factors, USEEIO
- **Deployment**: AWS Lambda (serverless), Ollama (local inference)
- **Frontend**: Next.js, Tailwind CSS (optional Streamlit for prototyping)

### Performance Metrics

- **Classification Accuracy**: F1 score of 0.82 (baseline: 0.25, LLM-only: 0.78)
- **Processing Speed**: <200ms per transaction classification
- **Coverage**: 95%+ of transactions successfully categorized
- **Model Size**: 3B parameters, ~4GB RAM usage with 4-bit quantization

### Quick Start

#### Prerequisites
- Python 3.9+
- PostgreSQL 14+
- Git LFS (for model weights)
- Plaid developer account (free)

#### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/green-finance-ai.git
cd green-finance-ai

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
createdb greenfinance
psql greenfinance -f schema.sql
```

#### Configuration

Create a `.env` file:
```env
POSTGRES_URL=postgresql://localhost/greenfinance
PLAID_CLIENT_ID=your_client_id
PLAID_SANDBOX_SECRET=your_sandbox_secret
OLLAMA_URL=http://127.0.0.1:11434/api/generate
```

#### Run the Application

```bash
# Start the classification service
ollama run qwen3:1.7b  # In separate terminal

# Load sample data
python load_sample_data.py

# Start the API server
uvicorn app:app --reload

# Access the dashboard
streamlit run dashboard.py
```

### Project Structure

```
green-finance-ai/
├── data/
│   ├── mcc_buckets.csv          # MCC to bucket mapping
│   ├── emission_factors.csv     # CO₂ factors by category
│   └── sample_transactions.json # Demo data
├── models/
│   ├── baseline_classifier.py   # Regex + MCC rules
│   ├── llm_classifier.py        # LLM inference wrapper
│   └── finetune_qwen.py         # QLoRA training script
├── src/
│   ├── ingestion/
│   │   ├── plaid_client.py      # Banking API integration
│   │   └── normalizer.py        # Transaction preprocessing
│   ├── classification/
│   │   ├── pipeline.py          # Hybrid classification
│   │   └── evaluator.py         # Model benchmarking
│   ├── emissions/
│   │   ├── factors.py           # Emission factor lookup
│   │   └── calculator.py        # CO₂ computation
│   └── api/
│       ├── app.py               # FastAPI application
│       └── routes.py            # API endpoints
├── frontend/
│   ├── dashboard.py             # Streamlit interface
│   └── components/              # React components (optional)
├── tests/
├── requirements.txt
├── schema.sql
└── README.md
```

### API Endpoints

- `POST /transactions/ingest` - Import new transactions
- `GET /transactions/{id}/classify` - Classify single transaction
- `GET /emissions/monthly` - Monthly emission summary
- `POST /offsets/recommend` - Get offset suggestions
- `GET /health` - Health check

### Model Training

The project uses QLoRA (Quantized LoRA) to fine-tune Qwen-3B on transaction classification:

```bash
# Prepare training data
python export_training_set.py

# Fine-tune model (Google Colab recommended)
python finetune_qwen.py

# Evaluate performance
python evaluate_model.py --pred_column bucket_llm
```

### Evaluation Results

| Model | Precision | Recall | F1 Score | Avg Latency |
|-------|-----------|--------|----------|-------------|
| Baseline (Regex + MCC) | 0.79 | 0.66 | 0.68 | 0.1ms |
| Gemma-2B (Fast) | 0.72 | 0.92 | 0.80 | 65ms |
| Qwen-3B (Reasoning) | 0.83 | 0.94 | 0.88 | 210ms |
| Hybrid Cascade | 0.82 | 0.93 | 0.87 | 103ms |

### Data Sources

- **Transaction Data**: Plaid sandbox API (privacy-safe test data)
- **Emission Factors**: 
  - DEFRA 2024 (UK government carbon factors)
  - USEEIO (US Environmental Input-Output model)
  - EXIOBASE (Global multi-regional database)
- **Offset Providers**: Patch API, Gold Standard registry

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Future Enhancements

- [ ] Real-time Plaid webhooks via AWS Lambda
- [ ] Merchant-level emission factors using NAICS codes
- [ ] User preference personalization (EV ownership, diet)
- [ ] Preventive recommendations (flight → train alternatives)
- [ ] Mobile app with React Native
- [ ] Integration with major offset providers
- [ ] Multi-currency support with exchange rates

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Qwen team at Alibaba for the base language model
- Hugging Face for the PEFT library and model hosting
- Plaid for providing sandbox banking data
- DEFRA, EPA, and other agencies for open emission factor data

### Contact

**Smit Mewada**  
Toronto, ON  
Email: smitmewada009@gmail.com  
LinkedIn: [https://www.linkedin.com/in/smitmewada/](https://www.linkedin.com/in/smitmewada/)

---

*Built as a demonstration project for fintech + climate tech integration. This prototype showcases end-to-end ML engineering, real-time data processing, and sustainable technology applications.*
