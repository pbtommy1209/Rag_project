# 🧪 RAG System Testing Guide

This guide provides comprehensive instructions for testing your RAG system and measuring its accuracy.

## 🚀 Quick Start Testing

### 1. **System Validation Test**
```bash
# Test if everything is set up correctly
python test_rag_system.py
```
This will check:
- ✅ All required modules are installed
- ✅ Ollama is running and accessible
- ✅ PDF reading works
- ✅ Embedding generation works
- ✅ Chat model works

### 2. **Comprehensive Testing Suite**
```bash
# Run all tests automatically
python run_tests.py

# Or run quick tests only
python run_tests.py --quick
```

## 📊 Accuracy Testing

### **Focused Accuracy Test**
```bash
# Test accuracy with a specific index
python test_accuracy.py --index "./faiss_indexes/adaptive_size500_overlap100" --output "accuracy_results.json"
```

This will test:
- 10 different question types (factual, numerical, reasoning, comparative)
- Different difficulty levels (easy, medium, hard)
- Overall accuracy percentage
- Performance breakdown by question type and difficulty

### **Expected Accuracy Benchmarks**
- **Good Performance**: 70%+ overall accuracy
- **Excellent Performance**: 85%+ overall accuracy
- **Factual Questions**: Should achieve 80%+ accuracy
- **Reasoning Questions**: May be lower (50-70%) depending on complexity

## 🔧 Step-by-Step Testing Workflow

### **Step 1: Environment Setup**
```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies
pip install -r "requirements (1).txt"

# 3. Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# 4. Check Ollama is running
curl http://localhost:11434/api/tags
```

### **Step 2: Create Test Indexes**
```bash
# Create small test indexes for quick testing
python enhanced_ingest.py \
    --pdf "F1_33.pdf" \
    --outdir "./test_indexes" \
    --strategy adaptive \
    --sizes "300,500" \
    --overlaps "50,100" \
    --embed-model "mxbai-embed-large" \
    --batch-size 3
```

### **Step 3: Test Retrieval**
```bash
# Test different retrieval methods
python enhanced_retrieve.py \
    --index "./test_indexes/adaptive_size500_overlap100" \
    --query "Who is Max Verstappen?" \
    --method semantic \
    --k 3

python enhanced_retrieve.py \
    --index "./test_indexes/adaptive_size500_overlap100" \
    --query "Who is Max Verstappen?" \
    --method hybrid \
    --alpha 0.7 \
    --k 3
```

### **Step 4: Test Answer Generation**
```bash
# Test end-to-end RAG
python run_demo.py \
    --index "./test_indexes/adaptive_size500_overlap100" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --k 3
```

### **Step 5: Comprehensive Evaluation**
```bash
# Run full evaluation
python enhanced_eval.py \
    --pdf "F1_33.pdf" \
    --indexes_root "./test_indexes" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --k 3 \
    --output "evaluation_results.json"
```

## 📈 Performance Metrics Explained

### **Retrieval Metrics**
- **Hit@k**: Percentage of questions where correct answer is in top-k retrieved passages
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of first relevant passage
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)

### **Answer Quality Metrics**
- **Exact Match**: Percentage of questions answered correctly
- **BLEU/ROUGE**: Text similarity between generated and expected answers
- **Faithfulness**: How well answers stick to retrieved information

### **System Performance Metrics**
- **Latency**: Time to generate answers
- **Throughput**: Questions processed per minute
- **Memory Usage**: RAM consumption during operation

## 🎯 Testing Different Scenarios

### **1. Question Type Testing**
```bash
# Test factual questions
python test_accuracy.py --index "./faiss_indexes/best_index" --questions 5

# Test reasoning questions  
python test_accuracy.py --index "./faiss_indexes/best_index" --questions 10
```

### **2. Chunking Strategy Comparison**
```bash
# Test different chunking strategies
python enhanced_eval.py \
    --pdf "F1_33.pdf" \
    --indexes_root "./faiss_indexes" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b"
```

### **3. Parameter Optimization**
```bash
# Find optimal parameters
python optimize_params.py \
    --indexes_root "./faiss_indexes" \
    --embed-model "mxbai-embed-large" \
    --chat-model "llama3.2:3b" \
    --max-combinations 20
```

## 🐛 Troubleshooting Common Issues

### **Issue: "Module not found" errors**
```bash
# Solution: Install missing dependencies
pip install pypdf faiss-cpu numpy tqdm requests scikit-learn nltk
```

### **Issue: "Ollama connection failed"**
```bash
# Solution: Start Ollama service
ollama serve

# Check if models are available
ollama list
```

### **Issue: "PDF reading failed"**
```bash
# Solution: Check PDF file exists and is readable
ls -la F1_33.pdf
file F1_33.pdf
```

### **Issue: "Low accuracy scores"**
- Try different chunking strategies (semantic vs adaptive)
- Adjust chunk size and overlap parameters
- Use larger chat models (llama3.1:8b instead of llama3.2:3b)
- Increase number of retrieved passages (k=5 or k=10)

### **Issue: "Slow performance"**
- Reduce batch size in ingestion
- Use smaller embedding models
- Reduce number of retrieved passages
- Use faster chunking strategies

## 📊 Interpreting Results

### **Good Results Indicators**
- ✅ Overall accuracy > 70%
- ✅ Factual questions > 80% accuracy
- ✅ Hit@5 > 0.8 (80% of questions have correct answer in top 5)
- ✅ MRR > 0.6
- ✅ Low latency (< 5 seconds per question)

### **Areas for Improvement**
- ❌ Overall accuracy < 60%
- ❌ Reasoning questions < 50% accuracy
- ❌ High latency (> 10 seconds per question)
- ❌ Many "I couldn't find that" responses

## 🎯 Best Practices

1. **Start Small**: Test with a few questions first
2. **Use Multiple Metrics**: Don't rely on just one accuracy measure
3. **Test Different Configurations**: Try various chunk sizes and strategies
4. **Validate Manually**: Check some answers manually for quality
5. **Monitor Performance**: Track both accuracy and speed
6. **Iterate**: Use results to improve your system

## 📝 Sample Test Questions

The system includes these test questions:

**Factual (Easy)**:
- "Who is the main subject of this article?"
- "Which company sponsors the Red Bull team?"

**Numerical (Medium)**:
- "What is Verstappen's approximate annual contract value?"
- "How many world championships has Verstappen won?"

**Reasoning (Hard)**:
- "Why did Verstappen complain during the Austin race?"
- "What factors contributed to F1's growth in the US?"

**Comparative (Hard)**:
- "Which is more valuable: Verstappen's contract or Hamilton's legacy?"

## 🚀 Next Steps After Testing

1. **If accuracy is good (>70%)**: Deploy for production use
2. **If accuracy is moderate (50-70%)**: Optimize parameters and retest
3. **If accuracy is low (<50%)**: Check data quality and model choices

Remember: Testing is iterative. Use the results to continuously improve your RAG system!
