import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from typing import List, Dict
import pickle
import os

class LightweightToxicityModel:
    """
    Lightweight model chạy local, không cần API
    Based on DistilBERT - chỉ 66MB vs 440MB của BERT
    """
    
    def __init__(self, model_path: str = "models/toxicity_distilbert.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = ToxicityClassifier()
        
        # Load pre-trained weights nếu có
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()
        
        # Cache cho performance
        self.prediction_cache = {}
        self.cache_size = 1000
        
    def predict(self, text: str) -> Dict:
        """
        Fast prediction với caching
        Returns: {
            'toxic_score': 0.0-1.0,
            'categories': {...},
            'confidence': 0.0-1.0
        }
        """
        # Check cache first
        text_hash = hash(text[:100])  # Hash first 100 chars
        if text_hash in self.prediction_cache:
            return self.prediction_cache[text_hash]
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=128,  # Giới hạn để faster
            return_tensors="pt"
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(
                inputs['input_ids'],
                inputs['attention_mask']
            )
            
        # Process results
        probs = torch.softmax(outputs, dim=-1)
        toxic_score = probs[0][1].item()  # Probability of toxic class
        
        result = {
            'toxic_score': toxic_score,
            'categories': self._get_categories(text, toxic_score),
            'confidence': max(probs[0]).item()
        }
        
        # Cache result
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entries
            self.prediction_cache.clear()
        self.prediction_cache[text_hash] = result
        
        return result
        
    def _get_categories(self, text: str, score: float) -> Dict:
        """Simple rule-based category detection"""
        categories = {}
        text_lower = text.lower()
        
        # Quick patterns
        if any(word in text_lower for word in ['hate', 'kill', 'die']):
            categories['severe_toxicity'] = min(score + 0.2, 1.0)
        if any(word in text_lower for word in ['stupid', 'idiot', 'dumb']):
            categories['insult'] = score
        if any(word in text_lower for word in ['threat', 'hurt', 'harm']):
            categories['threat'] = min(score + 0.1, 1.0)
            
        return categories

class ToxicityClassifier(nn.Module):
    """Simple DistilBERT-based classifier"""
    
    def __init__(self, num_labels=2):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_labels)
        
        # Freeze some layers để faster inference
        for param in self.distilbert.embeddings.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

# ===== TRAINING SCRIPT (Optional) =====

def train_model_on_discord_data():
    """
    Train model trên actual Discord data
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset
    
    class ToxicityDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': torch.tensor(label, dtype=torch.long)
            }
    
    # Load your Discord data
    df = pd.read_csv('discord_messages_labeled.csv')
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        df['message'].values,
        df['is_toxic'].values,
        test_size=0.2,
        random_state=42
    )
    
    # Create datasets
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_dataset = ToxicityDataset(X_train, y_train, tokenizer)
    val_dataset = ToxicityDataset(X_val, y_val, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = ToxicityClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Training settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
    # Save model
    torch.save(model.state_dict(), 'models/toxicity_distilbert.pt')

# ===== OPTIMIZED BOT INTEGRATION =====

class OptimizedMindfulBot(commands.Bot):
    """Bot với local ML model - no API costs!"""
    
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        # Initialize local model
        self.ml_model = LightweightToxicityModel()
        self.psych_detector = PsychologyBasedDetector()
        
        # Batch processing queue
        self.message_batch = []
        self.batch_size = 5
        self.batch_timeout = 2.0  # seconds
        
    async def on_message(self, message):
        if message.author.bot:
            return
            
        # Add to batch
        self.message_batch.append(message)
        
        # Process batch if full
        if len(self.message_batch) >= self.batch_size:
            await self.process_batch()
        else:
            # Set timeout for partial batch
            asyncio.create_task(self.batch_timeout_handler())
            
    async def process_batch(self):
        """Process messages in batch để efficient hơn"""
        if not self.message_batch:
            return
            
        batch = self.message_batch.copy()
        self.message_batch.clear()
        
        # Parallel processing
        tasks = []
        for msg in batch:
            task = asyncio.create_task(self.analyze_message(msg))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Handle results
        for msg, result in zip(batch, results):
            if result['needs_intervention']:
                await self.show_reflection_prompt(msg, result)
                
    async def analyze_message(self, message):
        """Analyze với both ML và psychology models"""
        # ML prediction (fast)
        ml_result = self.ml_model.predict(message.content)
        
        # Nếu ML model confident là safe, skip psychology check
        if ml_result['toxic_score'] < 0.3 and ml_result['confidence'] > 0.9:
            return {'needs_intervention': False}
            
        # Psychology-based analysis cho edge cases
        psych_result = self.psych_detector.analyze(
            message.content,
            {'history': []}  # Simplified for speed
        )
        
        # Combine results
        final_score = (ml_result['toxic_score'] * 0.7 + 
                      psych_result['harm_score'] * 0.3)
                      
        return {
            'needs_intervention': final_score > 0.6,
            'score': final_score,
            'prompts': psych_result['suggested_prompts'],
            'ml_confidence': ml_result['confidence']
        }

# ===== COST BREAKDOWN =====
"""
💰 COST ANALYSIS:

1. SELF-HOSTED (Recommended):
   - VPS: $5-10/month (Hetzner/DigitalOcean)
   - Redis: Included in VPS
   - ML Model: Free (runs on CPU)
   - Total: $5-10/month cho unlimited messages

2. CLOUD HOSTED:
   - Railway.app: $5/month (bot + Redis)
   - Fly.io: Free tier (3 shared VMs)
   - Render: $7/month
   
3. API COSTS (nếu dùng external):
   - OpenAI: ~$0.002 per message
   - Google Cloud NLP: ~$0.001 per message
   - For 100k messages/month = $100-200!
   
4. LOCAL MODEL BENEFITS:
   - No API costs
   - <100ms latency
   - Privacy (data stays local)
   - Customizable cho Discord context
   
RECOMMENDATION: Use local DistilBERT model!
"""

# ===== MONITORING & METRICS =====

class MetricsCollector:
    """Prometheus-compatible metrics"""
    
    def __init__(self):
        from prometheus_client import Counter, Histogram, Gauge
        
        # Metrics
        self.messages_processed = Counter(
            'mindful_messages_processed_total',
            'Total messages processed'
        )
        
        self.toxic_detections = Counter(
            'mindful_toxic_detections_total',
            'Total toxic messages detected',
            ['severity']
        )
        
        self.processing_time = Histogram(
            'mindful_processing_duration_seconds',
            'Message processing duration'
        )
        
        self.active_users = Gauge(
            'mindful_active_users',
            'Currently active users'
        )
        
    def record_message(self, processing_time: float, is_toxic: bool, severity: str = None):
        self.messages_processed.inc()
        self.processing_time.observe(processing_time)
        
        if is_toxic and severity:
            self.toxic_detections.labels(severity=severity).inc()

# ===== DEPLOYMENT CHECKLIST =====
"""
✅ PRODUCTION CHECKLIST:

1. Environment Setup:
   [ ] Redis installed và running
   [ ] Python 3.9+ với dependencies
   [ ] Model files downloaded
   [ ] Environment variables set

2. Performance Optimization:
   [ ] Enable Redis persistence
   [ ] Set up log rotation
   [ ] Configure memory limits
   [ ] Enable Prometheus metrics

3. Security:
   [ ] Token trong environment variable
   [ ] Rate limiting enabled
   [ ] Input sanitization
   [ ] Audit logging

4. Monitoring:
   [ ] Health check endpoint
   [ ] Error alerting (Sentry/Rollbar)
   [ ] Uptime monitoring
   [ ] Performance dashboards

5. Backup & Recovery:
   [ ] Database backups
   [ ] Model versioning
   [ ] Rollback procedure
   [ ] Disaster recovery plan
"""
