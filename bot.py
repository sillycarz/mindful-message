class PsychologyBasedDetector:
    """
    Dựa trên nghiên cứu tâm lý học về online harm:
    - Cyberbullying patterns (Kowalski et al., 2014)
    - Online disinhibition effect (Suler, 2004)
    - Social cognitive theory (Bandura, 1986)
    """
    
    def __init__(self):
        # Harm categories based on psychological research
        self.harm_patterns = {
            'dehumanization': {
                'patterns': [
                    r'\b(subhuman|animal|it|thing|creature)\b.*\b(you|they|them)\b',
                    r'\b(you|they|them)\b.*\b(deserve|should)\b.*\b(die|hurt|pain)\b'
                ],
                'weight': 0.9,
                'psychological_impact': 'severe'
            },
            'identity_attack': {
                'patterns': [
                    r'\b(stupid|idiot|retard|moron|dumb)\b',
                    r'\b(worthless|useless|pathetic|loser)\b',
                    r'nobody\s+(likes|wants|cares)\s+you'
                ],
                'weight': 0.7,
                'psychological_impact': 'moderate'
            },
            'social_exclusion': {
                'patterns': [
                    r'(no one|nobody|everyone)\s+(likes|wants|agrees)',
                    r'(kill\s+yourself|kys|end\s+it)',
                    r'(better\s+off|world\s+without)\s+you'
                ],
                'weight': 0.95,
                'psychological_impact': 'severe'
            },
            'gaslighting': {
                'patterns': [
                    r'(never\s+said|making\s+up|imagining)',
                    r'(crazy|insane|mental|psycho).*\b(you|your)\b',
                    r'(overreacting|sensitive|dramatic)'
                ],
                'weight': 0.6,
                'psychological_impact': 'moderate'
            },
            'threat': {
                'patterns': [
                    r'(i\s+will|going\s+to|gonna).*\b(hurt|harm|kill|find)\b',
                    r'(watch\s+your|be\s+careful|consequences)',
                    r'(know\s+where|find\s+you|your\s+address)'
                ],
                'weight': 1.0,
                'psychological_impact': 'severe'
            }
        }
        
        # Context amplifiers (làm tăng mức độ harm)
        self.context_amplifiers = {
            'repetition': lambda msg, history: self._check_repetition(msg, history),
            'targeting': lambda msg, history: self._check_targeting(msg, history),
            'mob_behavior': lambda msg, history: self._check_mob_behavior(msg, history),
            'time_pressure': lambda msg, history: self._check_time_pressure(msg, history)
        }
        
    def analyze(self, message: str, context: dict) -> dict:
        """
        Phân tích message với context đầy đủ
        Returns: {
            'harm_score': 0.0-1.0,
            'categories': [...],
            'psychological_impact': 'none/mild/moderate/severe',
            'intervention_needed': bool,
            'suggested_prompts': [...]
        }
        """
        message_lower = message.lower()
        harm_score = 0.0
        detected_categories = []
        
        # Check harm patterns
        for category, config in self.harm_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, message_lower):
                    harm_score = max(harm_score, config['weight'])
                    detected_categories.append({
                        'category': category,
                        'impact': config['psychological_impact']
                    })
                    break
        
        # Apply context amplifiers
        amplifier_score = 0.0
        for amplifier_name, amplifier_func in self.context_amplifiers.items():
            if amplifier_func(message, context.get('history', [])):
                amplifier_score += 0.1
                
        final_score = min(1.0, harm_score + amplifier_score)
        
        # Determine psychological impact
        if final_score >= 0.8:
            impact = 'severe'
        elif final_score >= 0.6:
            impact = 'moderate'
        elif final_score >= 0.3:
            impact = 'mild'
        else:
            impact = 'none'
            
        return {
            'harm_score': final_score,
            'categories': detected_categories,
            'psychological_impact': impact,
            'intervention_needed': final_score >= 0.6,
            'suggested_prompts': self._get_prompts_for_impact(impact, detected_categories)
        }
    
    def _check_repetition(self, msg, history):
        """Harassment qua repetition"""
        if len(history) < 3:
            return False
        # Check if similar messages repeated
        recent = history[-5:]
        similar_count = sum(1 for h in recent if self._similarity(msg, h['content']) > 0.7)
        return similar_count >= 3
        
    def _check_targeting(self, msg, history):
        """Single person being targeted"""
        mentions = re.findall(r'<@!?(\d+)>', msg)
        if not mentions:
            return False
        # Check if same person mentioned multiple times recently
        recent_mentions = []
        for h in history[-10:]:
            recent_mentions.extend(re.findall(r'<@!?(\d+)>', h.get('content', '')))
        return recent_mentions.count(mentions[0]) >= 4
        
    def _check_mob_behavior(self, msg, history):
        """Multiple people attacking one"""
        return len(set(h.get('author_id') for h in history[-5:])) >= 3
        
    def _check_time_pressure(self, msg, history):
        """Rapid-fire messages (emotional flooding)"""
        if len(history) < 3:
            return False
        timestamps = [h.get('timestamp') for h in history[-3:]]
        if timestamps[0] and timestamps[-1]:
            time_diff = timestamps[-1] - timestamps[0]
            return time_diff.total_seconds() < 30  # 3 messages in 30 seconds
            
    def _similarity(self, str1, str2):
        """Simple similarity check"""
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())
        return len(set1 & set2) / max(len(set1), len(set2))
        
    def _get_prompts_for_impact(self, impact, categories):
        """Prompts dựa trên psychological research"""
        prompts = {
            'severe': [
                "This message could cause serious emotional harm. Please reconsider.",
                "Words like these can have lasting psychological impact. Take a moment.",
                "Consider: Would you say this face-to-face? Digital words hurt too."
            ],
            'moderate': [
                "This might hurt someone's feelings. Is that your intention?",
                "Strong emotions are valid. Express them without attacking others?",
                "Pause and reread. Does this reflect your best self?"
            ],
            'mild': [
                "Consider rephrasing to be more constructive?",
                "Your point is valid. Can you express it more kindly?",
                "How would you feel receiving this message?"
            ],
            'none': []
        }
        
        # Add category-specific prompts
        category_prompts = {
            'dehumanization': "Remember: There's a real person reading this.",
            'identity_attack': "Attack ideas, not people.",
            'social_exclusion': "Exclusion can cause real psychological harm.",
            'gaslighting': "Validate feelings, even in disagreement.",
            'threat': "Threats are never acceptable and may be illegal."
        }
        
        base_prompts = prompts.get(impact, [])
        for cat in categories:
            if cat['category'] in category_prompts:
                base_prompts.append(category_prompts[cat['category']])
                
        return base_prompts[:3]  # Max 3 prompts

# ===== SCALABLE INFRASTRUCTURE =====

class MessageQueue:
    """Redis-based message queue for scalability"""
    
    def __init__(self, redis_url="redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis = None
        
    async def connect(self):
        self.redis = await aioredis.create_redis_pool(self.redis_url)
        
    async def push(self, channel_id: str, message_data: dict):
        """Push message to queue"""
        key = f"queue:{channel_id}"
        await self.redis.lpush(key, json.dumps(message_data))
        # Keep only last 100 messages per channel
        await self.redis.ltrim(key, 0, 99)
        
    async def get_history(self, channel_id: str, limit: int = 10):
        """Get recent history for context"""
        key = f"queue:{channel_id}"
        messages = await self.redis.lrange(key, 0, limit-1)
        return [json.loads(msg) for msg in messages]
        
    async def clear_old_data(self):
        """Clear data older than 1 hour"""
        # Run this periodically to prevent memory bloat
        pattern = "queue:*"
        cursor = b'0'
        while cursor:
            cursor, keys = await self.redis.scan(cursor, match=pattern)
            for key in keys:
                await self.redis.expire(key, 3600)  # 1 hour TTL

@dataclass
class ProcessingConfig:
    """Configuration for distributed processing"""
    max_workers: int = 4
    batch_size: int = 10
    redis_url: str = "redis://localhost:6379"
    api_endpoint: str = "http://localhost:8000/analyze"
    use_local_model: bool = True  # Use local model vs API

class DistributedMindfulBot(commands.Bot):
    """Scalable bot architecture"""
    
    def __init__(self, config: ProcessingConfig):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.config = config
        self.detector = PsychologyBasedDetector()
        self.queue = MessageQueue(config.redis_url)
        self.processing_semaphore = asyncio.Semaphore(config.max_workers)
        
        # Cache for user settings (với TTL)
        self.user_cache = {}  # In production: Use Redis
        
        # Metrics
        self.metrics = {
            'messages_processed': 0,
            'api_calls': 0,
            'cache_hits': 0
        }
        
    async def on_ready(self):
        print(f'{self.user} connected!')
        await self.queue.connect()
        # Start background tasks
        self.loop.create_task(self.cleanup_task())
        self.loop.create_task(self.metrics_reporter())
        
    async def on_message(self, message):
        """Optimized message handler"""
        if message.author.bot:
            return
            
        # Quick command check
        if message.content.startswith(self.command_prefix):
            await self.process_commands(message)
            return
            
        # Get user settings from cache
        user_settings = await self.get_user_settings(message.author.id)
        
        if not user_settings.get('reflection_enabled', True):
            return
            
        # Process with semaphore để limit concurrent processing
        async with self.processing_semaphore:
            await self.process_message(message)
            
    async def process_message(self, message):
        """Process individual message"""
        # Get context từ queue
        history = await self.queue.get_history(
            str(message.channel.id), 
            limit=10
        )
        
        # Quick pre-filter để giảm API calls
        if not self.needs_analysis(message.content):
            self.metrics['cache_hits'] += 1
            return
            
        # Analyze với local model
        if self.config.use_local_model:
            result = self.detector.analyze(
                message.content,
                {'history': history}
            )
        else:
            # Call external API cho heavy processing
            result = await self.call_analysis_api(message.content, history)
            self.metrics['api_calls'] += 1
            
        self.metrics['messages_processed'] += 1
        
        # Store message data
        await self.queue.push(str(message.channel.id), {
            'content': message.content,
            'author_id': str(message.author.id),
            'timestamp': datetime.utcnow().isoformat(),
            'harm_score': result['harm_score']
        })
        
        # Take action if needed
        if result['intervention_needed']:
            await self.show_reflection(message, result)
            
    def needs_analysis(self, content: str) -> bool:
        """Quick pre-filter để reduce processing load"""
        # Skip short messages
        if len(content) < 10:
            return False
            
        # Skip nếu không có trigger words
        quick_triggers = ['hate', 'stupid', 'kill', 'die', 'nobody', 'worthless']
        content_lower = content.lower()
        
        return any(trigger in content_lower for trigger in quick_triggers)
        
    async def get_user_settings(self, user_id: int) -> dict:
        """Get user settings với caching"""
        # Check cache first
        if user_id in self.user_cache:
            cached = self.user_cache[user_id]
            if cached['expires'] > datetime.utcnow():
                return cached['settings']
                
        # Load from database (hoặc default)
        settings = await self.load_user_settings(user_id)
        
        # Cache với TTL
        self.user_cache[user_id] = {
            'settings': settings,
            'expires': datetime.utcnow() + timedelta(minutes=15)
        }
        
        return settings
        
    async def cleanup_task(self):
        """Background task để cleanup memory"""
        while True:
            await asyncio.sleep(300)  # Every 5 minutes
            
            # Clear old queue data
            await self.queue.clear_old_data()
            
            # Clear expired cache entries
            now = datetime.utcnow()
            expired = [
                uid for uid, data in self.user_cache.items() 
                if data['expires'] < now
            ]
            for uid in expired:
                del self.user_cache[uid]
                
            print(f"Cleaned up {len(expired)} cache entries")
            
    async def metrics_reporter(self):
        """Report metrics periodically"""
        while True:
            await asyncio.sleep(60)  # Every minute
            print(f"Metrics: {self.metrics}")
            # In production: Send to monitoring service

# ===== DEPLOYMENT CONFIGURATION =====

"""
Deployment options cho scale:

1. KUBERNETES DEPLOYMENT:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mindful-bot
spec:
  replicas: 3  # Multiple instances
  selector:
    matchLabels:
      app: mindful-bot
  template:
    metadata:
      labels:
        app: mindful-bot
    spec:
      containers:
      - name: bot
        image: mindful-bot:latest
        env:
        - name: DISCORD_TOKEN
          valueFrom:
            secretKeyRef:
              name: discord-secret
              key: token
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: redis-service
spec:
  selector:
    app: redis
  ports:
  - port: 6379
```

2. DOCKER COMPOSE cho development:
```yaml
version: '3.8'
services:
  bot:
    build: .
    environment:
      - DISCORD_TOKEN=${DISCORD_TOKEN}
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      replicas: 2
      
  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data
      
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      
volumes:
  redis-data:
```

3. COST-EFFECTIVE HOSTING OPTIONS:

a) Railway.app (Recommended cho start):
   - $5/month cho bot + Redis
   - Auto-scaling
   - Easy deployment
   
b) Fly.io:
   - Free tier available
   - Global distribution
   - Good cho Discord bots
   
c) Self-hosted VPS:
   - DigitalOcean: $6/month
   - Linode: $5/month
   - Hetzner: €4/month

4. API ALTERNATIVES:

Thay vì dùng expensive AI APIs:
- Use local models: DistilBERT cho toxicity detection
- Caching layer: Redis cho frequent patterns
- Rule-based pre-filter: Reduce API calls by 80%
"""

if __name__ == "__main__":
    config = ProcessingConfig(
        max_workers=4,
        use_local_model=True,  # No API costs!
        redis_url="redis://localhost:6379"
    )
    
    bot = DistributedMindfulBot(config)
    bot.run('YOUR_TOKEN')
