#!/usr/bin/env python3
"""
MindfulMessage Auto Setup Script
Automatically sets up the bot with minimal configuration
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path

class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_banner():
    banner = f"""
{Colors.BLUE}{Colors.BOLD}
╔═══════════════════════════════════════╗
║      MindfulMessage Bot Setup         ║
║   Making Discord Communities Kinder   ║
╚═══════════════════════════════════════╝
{Colors.END}
    """
    print(banner)

def check_python_version():
    """Check if Python version is 3.8+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"{Colors.RED}❌ Python 3.8+ required. You have {sys.version}{Colors.END}")
        sys.exit(1)
    print(f"{Colors.GREEN}✅ Python {version.major}.{version.minor} detected{Colors.END}")

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        'data',
        'logs',
        'models',
        'configs'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"{Colors.GREEN}✅ Created directory structure{Colors.END}")

def download_model():
    """Download pre-trained model"""
    print(f"{Colors.YELLOW}📥 Downloading ML model (66MB)...{Colors.END}")
    
    model_url = "https://github.com/yourusername/mindful-models/releases/download/v1.0/distilbert_discord.pt"
    model_path = "models/toxicity_model.pt"
    
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_path)
        print(f"{Colors.GREEN}✅ Model downloaded successfully{Colors.END}")
    except:
        print(f"{Colors.YELLOW}⚠️  Model download failed. Using rule-based detection only.{Colors.END}")

def create_env_file():
    """Create .env file with user input"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Discord Bot Configuration{Colors.END}")
    
    if os.path.exists('.env'):
        overwrite = input(f"{Colors.YELLOW}.env file exists. Overwrite? (y/n): {Colors.END}").lower()
        if overwrite != 'y':
            return
    
    print(f"\n{Colors.YELLOW}Get your bot token from: https://discord.com/developers/applications{Colors.END}")
    token = input(f"{Colors.BLUE}Enter Discord Bot Token: {Colors.END}").strip()
    
    hosting = input(f"""
{Colors.BLUE}Select hosting option:
1. Railway (Recommended - Easy)
2. Render (Free tier available)  
3. Fly.io (Advanced)
4. Local (Not recommended)

Choice (1-4): {Colors.END}""").strip()
    
    env_content = f"""# Discord Configuration
DISCORD_TOKEN={token}

# Hosting Configuration
HOSTING_PLATFORM={'railway' if hosting == '1' else 'render' if hosting == '2' else 'fly' if hosting == '3' else 'local'}

# Feature Flags
USE_ML_MODEL=false  # Start with rule-based, enable later
USE_REDIS=false     # Start simple, enable for scale
ENABLE_ANALYTICS=true

# Settings
REFLECTION_DELAY_SECONDS=5
MAX_MESSAGE_LENGTH=2000
DEFAULT_LANGUAGE=en
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"{Colors.GREEN}✅ Configuration saved to .env{Colors.END}")

def create_requirements():
    """Create requirements.txt"""
    requirements = """# Core
discord.py==2.3.2
python-dotenv==1.0.0
aiohttp==3.9.1

# Analytics
prometheus-client==0.19.0

# ML (Optional - comment out for lightweight)
# torch==2.1.0
# transformers==4.36.0

# Database (Optional - for scaling)
# aioredis==2.0.1
# asyncpg==0.29.0

# Deployment
gunicorn==21.2.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Lightweight version for free hosting
    requirements_light = """discord.py==2.3.2
python-dotenv==1.0.0
aiohttp==3.9.1
"""
    
    with open('requirements-light.txt', 'w') as f:
        f.write(requirements_light)
    
    print(f"{Colors.GREEN}✅ Created requirements.txt{Colors.END}")

def create_bot_file():
    """Create simplified bot.py"""
    bot_code = '''import discord
from discord.ext import commands
import os
from dotenv import load_dotenv
import json
import asyncio
from datetime import datetime
import re

load_dotenv()

# Simple psychology-based patterns (no ML needed)
HARMFUL_PATTERNS = {
    'severe': {
        'patterns': [r'kill\s+yourself', r'kys', r'nobody\s+likes\s+you', r'better\s+off\s+dead'],
        'weight': 0.9
    },
    'moderate': {
        'patterns': [r'stupid', r'idiot', r'retard', r'worthless', r'loser'],
        'weight': 0.6
    },
    'mild': {
        'patterns': [r'shut\s+up', r'nobody\s+asked', r'who\s+asked'],
        'weight': 0.3
    }
}

class MindfulBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.pending_messages = {}
        self.stats = {'total': 0, 'reflected': 0, 'cancelled': 0}
        
    async def on_ready(self):
        print(f'✅ {self.user} is online!')
        print(f'📊 Monitoring {len(self.guilds)} servers')
        
    async def on_message(self, message):
        if message.author.bot:
            return
            
        # Check for harmful content
        harm_score = self.check_harm(message.content)
        
        if harm_score > 0.5:
            await self.show_reflection(message, harm_score)
        else:
            await self.process_commands(message)
            
    def check_harm(self, content):
        """Simple pattern matching - no ML needed"""
        content_lower = content.lower()
        max_score = 0.0
        
        for category, config in HARMFUL_PATTERNS.items():
            for pattern in config['patterns']:
                if re.search(pattern, content_lower):
                    max_score = max(max_score, config['weight'])
                    break
                    
        return max_score
        
    async def show_reflection(self, message, harm_score):
        """Show reflection prompt"""
        # Delete original
        await message.delete()
        
        # Create embed
        embed = discord.Embed(
            title="🤔 Moment of Reflection",
            description=f"Your message: *{message.content[:50]}...*",
            color=discord.Color.blue()
        )
        
        prompts = [
            "Is this accurate and fair?",
            "Could this harm someone?",
            "Does this reflect who you want to be?"
        ]
        
        for i, prompt in enumerate(prompts, 1):
            embed.add_field(name=f"{i}.", value=prompt, inline=False)
            
        embed.set_footer(text="React to choose: ✅ Send | ❌ Cancel")
        
        # Send reflection
        ref_msg = await message.channel.send(
            f"{message.author.mention}, please reflect:",
            embed=embed
        )
        
        # Add reactions
        await ref_msg.add_reaction('✅')
        await ref_msg.add_reaction('❌')
        
        # Store for handling
        self.pending_messages[ref_msg.id] = {
            'author': message.author,
            'content': message.content,
            'channel': message.channel
        }
        
        # Update stats
        self.stats['total'] += 1
        self.stats['reflected'] += 1
        
        # Auto-timeout
        await asyncio.sleep(5)
        if ref_msg.id in self.pending_messages:
            await self.send_message(ref_msg.id)
            await ref_msg.delete()
            
    async def on_reaction_add(self, reaction, user):
        if user.bot or reaction.message.id not in self.pending_messages:
            return
            
        msg_data = self.pending_messages.get(reaction.message.id)
        if not msg_data or user != msg_data['author']:
            return
            
        if str(reaction.emoji) == '✅':
            await self.send_message(reaction.message.id)
        elif str(reaction.emoji) == '❌':
            self.stats['cancelled'] += 1
            await reaction.message.edit(
                content="Message cancelled. Taking a moment to breathe 🌱"
            )
            
        del self.pending_messages[reaction.message.id]
        await reaction.message.delete(delay=3)
        
    async def send_message(self, msg_id):
        """Send the reflected message"""
        msg_data = self.pending_messages.get(msg_id)
        if msg_data:
            await msg_data['channel'].send(
                f"{msg_data['author'].mention}: {msg_data['content']}"
            )

@bot.command(name='stats')
async def stats(ctx):
    """Show bot statistics"""
    stats = bot.stats
    embed = discord.Embed(
        title="📊 MindfulMessage Stats",
        color=discord.Color.green()
    )
    embed.add_field(name="Total Messages", value=stats['total'])
    embed.add_field(name="Reflected", value=stats['reflected'])
    embed.add_field(name="Cancelled", value=stats['cancelled'])
    
    if stats['reflected'] > 0:
        cancel_rate = (stats['cancelled'] / stats['reflected']) * 100
        embed.add_field(name="Cancel Rate", value=f"{cancel_rate:.1f}%")
        
    await ctx.send(embed=embed)

bot = MindfulBot()
bot.run(os.getenv('DISCORD_TOKEN'))
'''
    
    with open('bot.py', 'w') as f:
        f.write(bot_code)
    
    print(f"{Colors.GREEN}✅ Created bot.py{Colors.END}")

def create_deployment_files():
    """Create deployment configuration files"""
    
    # Railway.app
    railway_config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "numReplicas": 1,
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 3
        }
    }
    
    with open('railway.json', 'w') as f:
        json.dump(railway_config, f, indent=2)
    
    # Render.com
    render_yaml = """services:
  - type: worker
    name: mindful-bot
    env: python
    buildCommand: pip install -r requirements-light.txt
    startCommand: python bot.py
    envVars:
      - key: DISCORD_TOKEN
        sync: false
    autoDeploy: true
"""
    
    with open('render.yaml', 'w') as f:
        f.write(render_yaml)
    
    # Fly.io
    fly_toml = """app = "mindful-discord-bot"
primary_region = "sjc"

[build]
  builder = "paketobuildpacks/builder:base"
  buildpacks = ["gcr.io/paketo-buildpacks/python"]

[env]
  PORT = "8080"

[[services]]
  protocol = "tcp"
  internal_port = 8080
  
[services.concurrency]
  type = "connections"
  hard_limit = 25
  soft_limit = 20
"""
    
    with open('fly.toml', 'w') as f:
        f.write(fly_toml)
    
    # Procfile for Heroku-style deployment
    with open('Procfile', 'w') as f:
        f.write('worker: python bot.py\n')
    
    print(f"{Colors.GREEN}✅ Created deployment configurations{Colors.END}")

def create_readme():
    """Create README with deployment instructions"""
    readme = """# 🌱 MindfulMessage Discord Bot

Making Discord communities kinder, one message at a time.

## 🚀 Quick Deploy

### Option 1: Railway (Easiest - $5/month)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/mindful-bot)

1. Click the button above
2. Add your Discord token
3. Deploy!

### Option 2: Render (Free tier)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/yourusername/mindful-message)

1. Click deploy button
2. Add environment variable: `DISCORD_TOKEN`
3. Deploy!

### Option 3: Fly.io (More control)
```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly secrets set DISCORD_TOKEN=your_token_here
```

## 📖 Setup Instructions

1. **Get Discord Bot Token**
   - Go to https://discord.com/developers/applications
   - Create new application
   - Go to "Bot" section
   - Copy token

2. **Invite Bot to Server**
   ```
   https://discord.com/api/oauth2/authorize?client_id=YOUR_CLIENT_ID&permissions=8&scope=bot
   ```

3. **Configure (Optional)**
   Edit `.env` file for customization

## 💬 Commands

- `!stats` - View reflection statistics
- `!help` - Show all commands

## 🔧 Customization

Edit `HARMFUL_PATTERNS` in `bot.py` to customize detection.

## 📊 Features

- ✅ Reflection prompts before sending harmful messages
- ✅ Simple pattern-based detection (no ML needed)
- ✅ Lightweight - runs on free hosting
- ✅ Privacy-focused - no data storage
- ✅ Easy one-click deployment

## 🆘 Support

- Discord: [Join our server](https://discord.gg/mindful)
- Issues: [GitHub Issues](https://github.com/yourusername/mindful-message/issues)

## 📝 License

MIT License - Make Discord kinder!
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print(f"{Colors.GREEN}✅ Created README.md{Colors.END}")

def create_github_actions():
    """Create GitHub Actions for auto-deployment"""
    Path('.github/workflows').mkdir(parents=True, exist_ok=True)
    
    workflow = """name: Deploy Bot

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy-railway:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Railway
        uses: bervProject/railway-deploy@main
        with:
          railway_token: ${{ secrets.RAILWAY_TOKEN }}
          
  deploy-render:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to Render
        uses: johnbeynon/render-deploy-action@v0.0.8
        with:
          service-id: ${{ secrets.RENDER_SERVICE_ID }}
          api-key: ${{ secrets.RENDER_API_KEY }}
"""
    
    with open('.github/workflows/deploy.yml', 'w') as f:
        f.write(workflow)
    
    print(f"{Colors.GREEN}✅ Created GitHub Actions workflow{Colors.END}")

def install_dependencies():
    """Install Python dependencies"""
    print(f"\n{Colors.YELLOW}📦 Installing dependencies...{Colors.END}")
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-light.txt'], 
                      check=True, capture_output=True)
        print(f"{Colors.GREEN}✅ Dependencies installed{Colors.END}")
    except subprocess.CalledProcessError:
        print(f"{Colors.YELLOW}⚠️  Could not install dependencies. Run: pip install -r requirements-light.txt{Colors.END}")

def show_next_steps():
    """Show deployment instructions"""
    print(f"""
{Colors.BLUE}{Colors.BOLD}✨ Setup Complete!{Colors.END}

{Colors.GREEN}Next Steps:{Colors.END}

1. {Colors.YELLOW}Test locally:{Colors.END}
   python bot.py

2. {Colors.YELLOW}Deploy to Railway (Easiest):{Colors.END}
   - Install Railway CLI: npm install -g @railway/cli
   - Run: railway login
   - Run: railway up
   - Add DISCORD_TOKEN in Railway dashboard

3. {Colors.YELLOW}Deploy to Render:{Colors.END}
   - Push to GitHub
   - Connect GitHub repo to Render
   - Add DISCORD_TOKEN as environment variable

4. {Colors.YELLOW}Invite bot to Discord:{Colors.END}
   Use the OAuth2 URL from Discord Developer Portal

{Colors.BLUE}📚 Full guide: https://github.com/yourusername/mindful-message{Colors.END}

{Colors.GREEN}Happy moderating! 🌱{Colors.END}
""")

def main():
    """Main setup flow"""
    print_banner()
    
    # Check environment
    check_python_version()
    
    # Create structure
    create_directory_structure()
    
    # Create files
    create_requirements()
    create_bot_file()
    create_env_file()
    create_deployment_files()
    create_readme()
    create_github_actions()
    
    # Optional steps
    if input(f"\n{Colors.YELLOW}Download ML model? (y/n): {Colors.END}").lower() == 'y':
        download_model()
    
    if input(f"{Colors.YELLOW}Install dependencies now? (y/n): {Colors.END}").lower() == 'y':
        install_dependencies()
    
    # Show completion
    show_next_steps()

if __name__ == "__main__":
    main()
