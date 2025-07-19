import discord
from discord.ext import commands
import asyncio
import json
import datetime
from collections import defaultdict
import random

# Configuration
REFLECTION_TIME = 5  # seconds
ANALYTICS_FILE = "message_analytics.json"

class ReflectionPrompts:
    """Quản lý các câu hỏi reflection với CBT-inspired prompts"""
    
    BASE_QUESTIONS = [
        "Is this accurate and fair?",
        "Could this harm someone?", 
        "Does this reflect who I want to be?"
    ]
    
    CBT_PROMPTS = [
        "What emotion am I feeling right now?",
        "Is there another way to interpret this situation?",
        "What would I advise a friend in this situation?",
        "Will this matter in a week? A month?",
        "What's the most constructive response here?"
    ]
    
    MULTILINGUAL = {
        'vi': [
            "Điều này có chính xác và công bằng không?",
            "Điều này có thể làm tổn thương ai không?",
            "Điều này có phản ánh con người tôi muốn trở thành?"
        ],
        'es': [
            "¿Es esto preciso y justo?",
            "¿Podría esto dañar a alguien?",
            "¿Refleja esto quién quiero ser?"
        ]
    }
    
    @classmethod
    def get_prompts(cls, language='en', include_cbt=False):
        """Lấy prompts theo ngôn ngữ và tùy chọn CBT"""
        if language in cls.MULTILINGUAL:
            base = cls.MULTILINGUAL[language]
        else:
            base = cls.BASE_QUESTIONS
            
        if include_cbt and language == 'en':
            return base + [random.choice(cls.CBT_PROMPTS)]
        return base

class MessageAnalytics:
    """Theo dõi và phân tích message patterns"""
    
    def __init__(self, filename=ANALYTICS_FILE):
        self.filename = filename
        self.data = self.load_data()
        
    def load_data(self):
        try:
            with open(self.filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                'total_messages': 0,
                'messages_paused': 0,
                'messages_cancelled': 0,
                'toxic_patterns': defaultdict(int),
                'weekly_stats': defaultdict(lambda: {'total': 0, 'cancelled': 0})
            }
    
    def save_data(self):
        # Convert defaultdict to regular dict for JSON serialization
        data_to_save = {
            'total_messages': self.data['total_messages'],
            'messages_paused': self.data['messages_paused'],
            'messages_cancelled': self.data['messages_cancelled'],
            'toxic_patterns': dict(self.data['toxic_patterns']),
            'weekly_stats': dict(self.data['weekly_stats'])
        }
        with open(self.filename, 'w') as f:
            json.dump(data_to_save, f, indent=2)
    
    def record_message(self, paused=True, cancelled=False, toxic_keywords=None):
        """Ghi nhận thông tin về message"""
        self.data['total_messages'] += 1
        if paused:
            self.data['messages_paused'] += 1
        if cancelled:
            self.data['messages_cancelled'] += 1
            
        # Track weekly stats
        week = datetime.datetime.now().strftime("%Y-W%U")
        if week not in self.data['weekly_stats']:
            self.data['weekly_stats'][week] = {'total': 0, 'cancelled': 0}
        self.data['weekly_stats'][week]['total'] += 1
        if cancelled:
            self.data['weekly_stats'][week]['cancelled'] += 1
            
        # Track toxic patterns if detected
        if toxic_keywords:
            for keyword in toxic_keywords:
                self.data['toxic_patterns'][keyword] += 1
                
        self.save_data()
    
    def get_weekly_report(self):
        """Tạo báo cáo tuần"""
        current_week = datetime.datetime.now().strftime("%Y-W%U")
        stats = self.data['weekly_stats'].get(current_week, {'total': 0, 'cancelled': 0})
        
        report = f"📊 **Weekly Report**\n"
        report += f"Total messages: {stats['total']}\n"
        report += f"Messages cancelled after reflection: {stats['cancelled']}\n"
        
        if stats['total'] > 0:
            cancel_rate = (stats['cancelled'] / stats['total']) * 100
            report += f"Reflection impact: {cancel_rate:.1f}% messages reconsidered\n"
            
        return report

class MindfulBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(command_prefix='!', intents=intents)
        
        self.analytics = MessageAnalytics()
        self.pending_messages = {}  # Store messages pending reflection
        self.user_settings = {}  # Store user preferences
        
    async def on_ready(self):
        print(f'{self.user} has connected to Discord!')
        
    async def on_message(self, message):
        # Ignore bot messages
        if message.author.bot:
            return
            
        # Process commands first
        if message.content.startswith(self.command_prefix):
            await self.process_commands(message)
            return
            
        # Check for potential toxic patterns
        toxic_keywords = self.detect_toxic_patterns(message.content)
        
        # If toxic patterns detected or user has reflection enabled
        if toxic_keywords or self.user_settings.get(message.author.id, {}).get('reflection', True):
            await self.initiate_reflection(message, toxic_keywords)
        else:
            # Normal message flow
            await self.process_commands(message)
    
    def detect_toxic_patterns(self, content):
        """Simple toxic pattern detection - trong thực tế sẽ dùng ML model"""
        toxic_patterns = ['hate', 'stupid', 'idiot', 'dumb', 'loser']
        content_lower = content.lower()
        return [pattern for pattern in toxic_patterns if pattern in content_lower]
    
    async def initiate_reflection(self, message, toxic_keywords=None):
        """Bắt đầu quá trình reflection"""
        # Delete original message
        await message.delete()
        
        # Get user language preference
        user_lang = self.user_settings.get(message.author.id, {}).get('language', 'en')
        prompts = ReflectionPrompts.get_prompts(user_lang, include_cbt=bool(toxic_keywords))
        
        # Create reflection embed
        embed = discord.Embed(
            title="🤔 Moment of Reflection",
            description=f"Your message: *{message.content[:100]}{'...' if len(message.content) > 100 else ''}*",
            color=discord.Color.blue()
        )
        
        for i, prompt in enumerate(prompts, 1):
            embed.add_field(name=f"{i}.", value=prompt, inline=False)
        
        embed.set_footer(text=f"You have {REFLECTION_TIME} seconds to reflect...")
        
        # Send reflection message
        reflection_msg = await message.channel.send(
            f"{message.author.mention}, please take a moment to reflect:",
            embed=embed
        )
        
        # Add reactions for user choice
        await reflection_msg.add_reaction('✅')  # Send
        await reflection_msg.add_reaction('❌')  # Cancel
        await reflection_msg.add_reaction('✏️')  # Edit
        
        # Store pending message
        self.pending_messages[reflection_msg.id] = {
            'author': message.author,
            'content': message.content,
            'channel': message.channel,
            'toxic_keywords': toxic_keywords
        }
        
        # Record analytics
        self.analytics.record_message(paused=True, toxic_keywords=toxic_keywords)
        
        # Auto-timeout after REFLECTION_TIME seconds
        await asyncio.sleep(REFLECTION_TIME)
        
        if reflection_msg.id in self.pending_messages:
            # No reaction = auto-send
            await self.send_reflected_message(reflection_msg.id)
            await reflection_msg.delete()
    
    async def on_reaction_add(self, reaction, user):
        """Handle user reactions to reflection prompts"""
        if user.bot or reaction.message.id not in self.pending_messages:
            return
            
        msg_data = self.pending_messages.get(reaction.message.id)
        if not msg_data or user != msg_data['author']:
            return
            
        if str(reaction.emoji) == '✅':
            await self.send_reflected_message(reaction.message.id)
        elif str(reaction.emoji) == '❌':
            await reaction.message.edit(content="Message cancelled after reflection. 🌱")
            self.analytics.record_message(paused=False, cancelled=True, 
                                        toxic_keywords=msg_data['toxic_keywords'])
        elif str(reaction.emoji) == '✏️':
            await reaction.message.edit(
                content=f"{user.mention}, please send your edited message:"
            )
            # Wait for edited message
            # (Implementation cho edit flow)
            
        del self.pending_messages[reaction.message.id]
        await reaction.message.delete(delay=3)
    
    async def send_reflected_message(self, msg_id):
        """Send message after reflection period"""
        msg_data = self.pending_messages.get(msg_id)
        if msg_data:
            # Create webhook to preserve user identity
            webhooks = await msg_data['channel'].webhooks()
            webhook = webhooks[0] if webhooks else await msg_data['channel'].create_webhook(
                name="MindfulMessage"
            )
            
            await webhook.send(
                content=msg_data['content'],
                username=msg_data['author'].display_name,
                avatar_url=msg_data['author'].avatar.url if msg_data['author'].avatar else None
            )

# Bot commands
bot = MindfulBot()

@bot.command(name='settings')
async def settings(ctx, setting=None, value=None):
    """Configure personal settings"""
    user_id = ctx.author.id
    
    if not setting:
        # Show current settings
        user_settings = bot.user_settings.get(user_id, {})
        embed = discord.Embed(title="Your Settings", color=discord.Color.green())
        embed.add_field(name="Reflection", value=user_settings.get('reflection', True))
        embed.add_field(name="Language", value=user_settings.get('language', 'en'))
        await ctx.send(embed=embed)
        return
    
    if setting == 'reflection':
        bot.user_settings.setdefault(user_id, {})['reflection'] = value.lower() == 'on'
        await ctx.send(f"Reflection {'enabled' if value.lower() == 'on' else 'disabled'}")
    elif setting == 'language':
        bot.user_settings.setdefault(user_id, {})['language'] = value
        await ctx.send(f"Language set to: {value}")

@bot.command(name='stats')
async def stats(ctx):
    """Show weekly statistics"""
    report = bot.analytics.get_weekly_report()
    await ctx.send(report)

@bot.command(name='report')
async def report(ctx):
    """Generate detailed analytics report"""
    analytics = bot.analytics.data
    
    embed = discord.Embed(
        title="📈 MindfulMessage Analytics",
        color=discord.Color.blue()
    )
    
    embed.add_field(
        name="Overall Stats",
        value=f"Total messages: {analytics['total_messages']}\n"
              f"Messages paused: {analytics['messages_paused']}\n"
              f"Messages cancelled: {analytics['messages_cancelled']}",
        inline=False
    )
    
    if analytics['toxic_patterns']:
        top_patterns = sorted(analytics['toxic_patterns'].items(), 
                            key=lambda x: x[1], reverse=True)[:5]
        patterns_text = "\n".join([f"• {k}: {v}" for k, v in top_patterns])
        embed.add_field(
            name="Common Toxic Patterns",
            value=patterns_text or "None detected",
            inline=False
        )
    
    await ctx.send(embed=embed)

# Run bot
if __name__ == "__main__":
    # Bạn cần thay YOUR_BOT_TOKEN bằng token thật
    bot.run('YOUR_BOT_TOKEN')
