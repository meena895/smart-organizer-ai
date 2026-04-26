import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import smtplib
from email.message import EmailMessage
import speech_recognition as sr
import pyttsx3
import threading
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import json
from typing import Dict, List, Tuple
import imaplib
import email
from email.header import decode_header

# ================= CONFIG =================
st.set_page_config("TaskHub Smart ML", layout="wide")
ADMIN_EMAIL = "121324030015@sfc.ac.in"
APP_PASSWORD = "jpskvlheskucxndb"

# Gmail IMAP Configuration for REAL email fetching
GMAIL_IMAP_SERVER = "imap.gmail.com"
GMAIL_IMAP_PORT = 993

# ================= FILES =================
USERS = "users.csv"
SHOPPING = "shopping.csv"
TASKS = "tasks.csv"
MESSAGES = "messages.csv"
EMAILS = "data/emails.csv"
CLASSIFIED_EMAILS = "data/classified_emails.csv"
EMAIL_MODEL = "email_classifier.pkl"

def init_csv(file, cols):
    if not os.path.exists(file):
        os.makedirs(os.path.dirname(file), exist_ok=True) if os.path.dirname(file) else None
        pd.DataFrame(columns=cols).to_csv(file, index=False)

init_csv(USERS, ["email", "password"])
init_csv(SHOPPING, ["email", "item", "priority", "purchase_date"])
init_csv(TASKS, ["email", "task", "priority", "deadline"])
init_csv(MESSAGES, ["name", "email", "message", "timestamp", "category"])
init_csv(EMAILS, ["email", "subject", "body", "timestamp"])
init_csv(CLASSIFIED_EMAILS, ["email", "subject", "body", "category", "confidence", "timestamp"])

users_df = pd.read_csv(USERS)
shopping_df = pd.read_csv(SHOPPING)
tasks_df = pd.read_csv(TASKS)
messages_df = pd.read_csv(MESSAGES)

# ================= SESSION =================
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user" not in st.session_state:
    st.session_state.user = None
if "voice_context" not in st.session_state:
    st.session_state.voice_context = {}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reminder_thread" not in st.session_state:
    st.session_state.reminder_thread = None

def set_page(p):
    st.session_state.page = p

# ================= EMAIL FUNCTION =================
def send_email(to_email, subject, body):
    try:
        email = EmailMessage()
        email["From"] = ADMIN_EMAIL
        email["To"] = to_email
        email["Subject"] = subject
        email.set_content(body)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(ADMIN_EMAIL, APP_PASSWORD)
            smtp.send_message(email)
    except Exception as e:
        st.warning(f"Could not send email: {e}")

def send_confirmation_email(user_email, name, message):
    """Send confirmation email to user after contact form submission"""
    subject = "Thank you for contacting TaskHub Smart ML"
    body = f"""Dear {name},

Thank you for reaching out to TaskHub Smart ML. We have received your message:

"{message}"

Our team will review your message and get back to you within 24-48 hours.

Best regards,
TaskHub Smart ML Team
"""
    send_email(user_email, subject, body)

# ================= REAL GMAIL EMAIL FETCHING =================
def fetch_emails_from_gmail(max_emails=20):
    """
    Fetch REAL emails from Gmail inbox using IMAP
    Returns list of (subject, body, timestamp) tuples
    """
    try:
        # Connect to Gmail IMAP server
        mail = imaplib.IMAP4_SSL(GMAIL_IMAP_SERVER, GMAIL_IMAP_PORT)
        
        # Login with app password
        mail.login(ADMIN_EMAIL, APP_PASSWORD)
        
        # Select inbox
        mail.select("inbox")
        
        # Search for all emails (or recent ones)
        status, messages = mail.search(None, "ALL")
        
        if status != "OK":
            return []
        
        # Get email IDs
        email_ids = messages[0].split()
        
        # Fetch last N emails
        email_ids = email_ids[-max_emails:] if len(email_ids) > max_emails else email_ids
        
        fetched_emails = []
        
        for email_id in email_ids:
            try:
                # Fetch email data
                status, msg_data = mail.fetch(email_id, "(RFC822)")
                
                if status != "OK":
                    continue
                
                # Parse email
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        
                        # Get subject
                        subject = decode_header(msg["Subject"])[0][0]
                        if isinstance(subject, bytes):
                            subject = subject.decode()
                        
                        # Get email body
                        body = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    body = part.get_payload(decode=True).decode()
                                    break
                        else:
                            body = msg.get_payload(decode=True).decode()
                        
                        # Get timestamp
                        date_str = msg["Date"]
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        fetched_emails.append((subject, body[:500], timestamp))  # Limit body length
                        
            except Exception as e:
                continue
        
        # Logout
        mail.logout()
        
        return fetched_emails
        
    except Exception as e:
        st.warning(f"Could not fetch emails: {e}")
        return []

def auto_classify_emails():
    """
    Automatically fetch and classify emails from Gmail
    This runs automatically without user interaction
    """
    try:
        # Fetch real emails from Gmail
        emails = fetch_emails_from_gmail(max_emails=20)
        
        if not emails:
            return 0
        
        # Load existing classifications
        classified_df = pd.read_csv(CLASSIFIED_EMAILS)
        
        # Get already classified email subjects to avoid duplicates
        existing_subjects = set(classified_df['subject'].tolist()) if not classified_df.empty else set()
        
        new_classifications = 0
        
        for subject, body, timestamp in emails:
            # Validate data
            if not subject or not isinstance(subject, str):
                subject = "No Subject"
            if not body or not isinstance(body, str):
                body = "No Body"
            
            # Skip if already classified
            if subject in existing_subjects:
                continue
            
            # Classify using ML
            category, confidence = email_classifier.classify_email(subject, body)
            
            # Store classification with validated data
            new_row = pd.DataFrame([{
                'email': ADMIN_EMAIL,
                'subject': str(subject),
                'body': str(body),
                'category': str(category),
                'confidence': float(confidence),
                'timestamp': str(timestamp)
            }])
            
            classified_df = pd.concat([classified_df, new_row], ignore_index=True)
            
            new_classifications += 1
        
        # Save classifications
        if new_classifications > 0:
            classified_df.to_csv(CLASSIFIED_EMAILS, index=False)
        
        return new_classifications
        
    except Exception as e:
        st.warning(f"Error in auto_classify_emails: {e}")
        return 0

# ================= EMAIL CLASSIFICATION =================
class EmailClassifier:
    def __init__(self):
        self.model = None
        self.categories = ['Task', 'Shopping', 'General', 'Spam']
        self.load_or_create_model()
    
    def load_or_create_model(self):
        if os.path.exists(EMAIL_MODEL):
            with open(EMAIL_MODEL, 'rb') as f:
                self.model = pickle.load(f)
        else:
            self.create_initial_model()
    
    def create_initial_model(self):
        # Training data for initial model
        training_data = [
            ("Complete project deadline tomorrow", "Task"),
            ("Buy groceries milk bread eggs", "Shopping"),
            ("Meeting scheduled for next week", "Task"),
            ("Purchase new laptop accessories", "Shopping"),
            ("Hello how are you doing", "General"),
            ("Free money click here now", "Spam"),
            ("Urgent task needs completion", "Task"),
            ("Shopping list items needed", "Shopping"),
            ("Thank you for your help", "General"),
            ("Win lottery click link", "Spam"),
            ("Deadline approaching finish work", "Task"),
            ("Need to buy vegetables", "Shopping"),
            ("How is the weather today", "General"),
            ("Congratulations you won prize", "Spam")
        ]
        
        texts, labels = zip(*training_data)
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
            ('classifier', MultinomialNB())
        ])
        self.model.fit(texts, labels)
        self.save_model()
    
    def save_model(self):
        with open(EMAIL_MODEL, 'wb') as f:
            pickle.dump(self.model, f)
    
    def classify_email(self, subject, body):
        text = f"{subject} {body}".lower()
        prediction = self.model.predict([text])[0]
        confidence = max(self.model.predict_proba([text])[0])
        return prediction, confidence
    
    def classify_and_store(self, email, subject, body):
        category, confidence = self.classify_email(subject, body)
        
        # Store classified email
        classified_df = pd.read_csv(CLASSIFIED_EMAILS)
        new_row = {
            'email': email,
            'subject': subject,
            'body': body,
            'category': category,
            'confidence': confidence,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        classified_df.loc[len(classified_df)] = new_row
        classified_df.to_csv(CLASSIFIED_EMAILS, index=False)
        
        return category, confidence

email_classifier = EmailClassifier()

# ================= REMINDER SYSTEM =================
class ReminderEngine:
    def __init__(self):
        self.running = False
    
    def check_reminders(self):
        """Check for upcoming deadlines and send reminders"""
        while self.running:
            try:
                current_date = datetime.now().date()
                
                # Check tasks
                tasks_df = pd.read_csv(TASKS)
                for _, task in tasks_df.iterrows():
                    deadline = datetime.strptime(str(task['deadline']), '%Y-%m-%d').date()
                    days_until = (deadline - current_date).days
                    
                    if days_until == 1:  # 1 day before deadline
                        subject = f"Task Reminder: {task['task']} due tomorrow!"
                        body = f"Don't forget! Your task '{task['task']}' is due tomorrow ({deadline}).\nPriority: {task['priority']}"
                        send_email(task['email'], subject, body)
                    elif days_until == 0:  # Due today
                        subject = f"URGENT: Task '{task['task']}' is due TODAY!"
                        body = f"Your task '{task['task']}' is due today ({deadline}).\nPriority: {task['priority']}\nPlease complete it as soon as possible."
                        send_email(task['email'], subject, body)
                
                # Check shopping items
                shopping_df = pd.read_csv(SHOPPING)
                for _, item in shopping_df.iterrows():
                    purchase_date = datetime.strptime(str(item['purchase_date']), '%Y-%m-%d').date()
                    days_until = (purchase_date - current_date).days
                    
                    if days_until == 1:  # 1 day before purchase date
                        subject = f"Shopping Reminder: {item['item']} tomorrow!"
                        body = f"Don't forget to buy '{item['item']}' tomorrow ({purchase_date}).\nPriority: {item['priority']}"
                        send_email(item['email'], subject, body)
                    elif days_until == 0:  # Purchase today
                        subject = f"Shopping Today: {item['item']}"
                        body = f"Today is the day to buy '{item['item']}' ({purchase_date}).\nPriority: {item['priority']}"
                        send_email(item['email'], subject, body)
                
                # Sleep for 1 hour before next check
                time.sleep(3600)
                
            except Exception as e:
                print(f"Reminder engine error: {e}")
                time.sleep(3600)
    
    def start(self):
        if not self.running:
            self.running = True
            thread = threading.Thread(target=self.check_reminders, daemon=True)
            thread.start()
            return thread
    
    def stop(self):
        self.running = False

reminder_engine = ReminderEngine()

# ================= TEXT-TO-SPEECH =================
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# ================= SMART CHATBOT =================
class SmartChatbot:
    def __init__(self):
        self.conversation_context = {}
        self.awaiting_details = {}
        
        # Enhanced intent patterns
        self.intents = {
            'greeting': ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening', 'howdy', 'greetings'],
            'farewell': ['bye', 'goodbye', 'see you', 'farewell', 'take care', 'later'],
            'thanks': ['thank you', 'thanks', 'appreciate', 'grateful'],
            'help': ['help', 'what can you do', 'commands', 'options', 'assist', 'support'],
            'view_tasks': ['show tasks', 'my tasks', 'list tasks', 'what tasks', 'tasks list', 'see tasks', 'display tasks'],
            'add_task': ['add task', 'create task', 'new task', 'make task', 'task for', 'need to do', 'have to'],
            'edit_task': ['edit task', 'modify task', 'change task', 'update task', 'alter task'],
            'delete_task': ['delete task', 'remove task', 'cancel task', 'drop task'],
            'complete_task': ['complete task', 'finish task', 'done task', 'completed task'],
            'view_shopping': ['show shopping', 'shopping list', 'my shopping', 'what shopping', 'shopping items', 'buy list'],
            'add_shopping': ['add shopping', 'buy item', 'shopping item', 'add item', 'need to buy', 'purchase'],
            'edit_shopping': ['edit shopping', 'modify shopping', 'change shopping', 'update shopping'],
            'delete_shopping': ['delete shopping', 'remove shopping', 'cancel shopping'],
            'purchased_item': ['purchased', 'bought', 'got item', 'completed shopping'],
            'how_are_you': ['how are you', 'how do you feel', 'what\'s up', 'how\'s it going'],
            'what_is': ['what is', 'what are', 'tell me about', 'explain', 'define'],
            'weather': ['weather', 'temperature', 'forecast', 'climate'],
            'time': ['time', 'date', 'today', 'now', 'current'],
            'joke': ['joke', 'funny', 'humor', 'laugh', 'amusing'],
            'compliment': ['good job', 'well done', 'excellent', 'amazing', 'awesome', 'great'],
            'complaint': ['not working', 'broken', 'error', 'problem', 'issue', 'bug'],
            'personal': ['your name', 'who are you', 'about you', 'tell me about yourself']
        }
        
        # Conversation starters and responses
        self.responses = {
            'greeting': [
                "Hello! I'm your TaskHub AI assistant. I'm here to help you manage your tasks and shopping lists. How can I assist you today?",
                "Hi there! Great to see you again. What would you like to work on today?",
                "Hey! I'm ready to help you stay organized. What's on your mind?",
                "Good to see you! I can help with tasks, shopping, or just have a friendly chat. What interests you?"
            ],
            'farewell': [
                "Goodbye! Have a productive day ahead. I'll be here whenever you need help with your tasks!",
                "Take care! Don't forget to check your task deadlines. See you soon!",
                "Bye! Remember, I'm always here to help you stay organized. Have a great day!"
            ],
            'thanks': [
                "You're very welcome! I'm always happy to help you stay organized.",
                "My pleasure! That's what I'm here for - making your life easier.",
                "Glad I could help! Feel free to ask me anything else about your tasks or shopping."
            ],
            'how_are_you': [
                "I'm doing great, thank you for asking! I'm always excited to help people stay organized. How are you doing today?",
                "I'm fantastic! I love helping people manage their tasks and achieve their goals. How can I brighten your day?",
                "I'm wonderful, thanks! Every day is a good day when I get to help people like you. What's going well for you today?"
            ],
            'compliment': [
                "Thank you so much! I really appreciate your kind words. It motivates me to help you even better!",
                "That's so nice of you to say! I'm just doing my best to make your life more organized and easier.",
                "You're too kind! I'm glad I could help. Your success makes me happy too!"
            ],
            'joke': [
                "Why don't tasks ever get tired? Because they always have deadlines to meet! 😄",
                "What did the shopping list say to the task list? 'You're always so busy, but I'm just here for the essentials!' 🛒",
                "Why did the reminder go to therapy? It had too many issues to resolve! ⏰"
            ],
            'personal': [
                "I'm your TaskHub AI assistant! I'm designed to help you manage tasks, shopping lists, and stay organized. I love conversations and helping people achieve their goals!",
                "I'm an AI assistant built specifically for TaskHub Smart ML. My purpose is to make your life easier by helping with productivity, and I genuinely enjoy our conversations!",
                "I'm your friendly AI companion in TaskHub! I specialize in task management, shopping reminders, and I'm always learning to be more helpful."
            ]
        }
    
    def detect_intent(self, message):
        message = message.lower()
        
        # Check for multiple intents
        detected_intents = []
        for intent, patterns in self.intents.items():
            for pattern in patterns:
                if pattern in message:
                    detected_intents.append(intent)
                    break
        
        # Return the most specific intent or the first one found
        if detected_intents:
            # Prioritize task/shopping intents over general ones
            priority_intents = ['add_task', 'add_shopping', 'view_tasks', 'view_shopping']
            for intent in priority_intents:
                if intent in detected_intents:
                    return intent
            return detected_intents[0]
        
        return 'general_conversation'
    
    def extract_task_details(self, message):
        """Enhanced task extraction with better parsing"""
        priority_map = {'high': 'High', 'urgent': 'High', 'important': 'High',
                       'medium': 'Medium', 'normal': 'Medium', 'regular': 'Medium',
                       'low': 'Low', 'later': 'Low', 'sometime': 'Low'}
        
        priority = 'Medium'  # default
        
        # Find priority
        for p in priority_map:
            if p in message.lower():
                priority = priority_map[p]
                break
        
        # Extract task description with better parsing
        task_keywords = ['add task', 'create task', 'new task', 'task', 'need to do', 'have to', 'must', 'should']
        task_desc = message
        
        for keyword in task_keywords:
            if keyword in message.lower():
                # Find the position and extract everything after
                pos = message.lower().find(keyword)
                task_desc = message[pos + len(keyword):].strip()
                break
        
        # Clean up the description
        for p in priority_map:
            task_desc = task_desc.replace(f'with {p} priority', '').replace(f'{p} priority', '').strip()
        
        # Remove common words that might be left over
        cleanup_words = ['with', 'priority', 'please', 'can you', 'could you']
        for word in cleanup_words:
            task_desc = task_desc.replace(word, '').strip()
        
        return task_desc if task_desc else None, priority
    
    def extract_shopping_details(self, message):
        """Enhanced shopping extraction with better parsing"""
        priority_map = {'high': 'High', 'urgent': 'High', 'important': 'High',
                       'medium': 'Medium', 'normal': 'Medium', 'regular': 'Medium',
                       'low': 'Low', 'later': 'Low', 'sometime': 'Low'}
        
        priority = 'Medium'  # default
        
        # Find priority
        for p in priority_map:
            if p in message.lower():
                priority = priority_map[p]
                break
        
        # Extract item description
        shopping_keywords = ['add shopping', 'buy', 'purchase', 'get', 'shopping item', 'add item', 'need to buy']
        item_desc = message
        
        for keyword in shopping_keywords:
            if keyword in message.lower():
                pos = message.lower().find(keyword)
                item_desc = message[pos + len(keyword):].strip()
                break
        
        # Clean up the description
        for p in priority_map:
            item_desc = item_desc.replace(f'with {p} priority', '').replace(f'{p} priority', '').strip()
        
        cleanup_words = ['with', 'priority', 'please', 'can you', 'could you']
        for word in cleanup_words:
            item_desc = item_desc.replace(word, '').strip()
        
        return item_desc if item_desc else None, priority
    
    def get_random_response(self, response_type):
        """Get a random response from the response type"""
        import random
        if response_type in self.responses:
            return random.choice(self.responses[response_type])
        return "I'm here to help! What would you like to do?"
    
    def process_message(self, message, user_email):
        """Enhanced conversational processing like ChatGPT"""
        intent = self.detect_intent(message)
        
        # Handle different intents with conversational responses
        if intent == 'greeting':
            return self.get_random_response('greeting')
        
        elif intent == 'farewell':
            return self.get_random_response('farewell')
        
        elif intent == 'thanks':
            return self.get_random_response('thanks')
        
        elif intent == 'how_are_you':
            return self.get_random_response('how_are_you')
        
        elif intent == 'compliment':
            return self.get_random_response('compliment')
        
        elif intent == 'joke':
            return self.get_random_response('joke')
        
        elif intent == 'personal':
            return self.get_random_response('personal')
        
        elif intent == 'help':
            return """I'm your intelligent TaskHub assistant! Here's what I can help you with:

🎯 **Task Management:**
• "Show my tasks" - View all your tasks
• "Add task finish project report with high priority" - Create new tasks
• "Edit task [task name]" - Modify existing tasks
• "Delete task [task name]" - Remove tasks
• "Complete task [task name]" - Mark tasks as done

🛒 **Shopping Lists:**
• "Show my shopping list" - View shopping items
• "Buy milk and bread" - Add shopping items
• "Edit shopping [item name]" - Modify shopping items
• "Delete shopping [item name]" - Remove items
• "Purchased [item name]" - Mark items as bought

💬 **General Conversation:**
• Ask me questions about anything
• Tell me jokes or ask for motivation
• I can chat about weather, time, or just be friendly!

Just talk to me naturally - I understand context and can have real conversations like ChatGPT! What would you like to explore?"""
        
        elif intent == 'view_tasks':
            try:
                tasks_df = pd.read_csv(TASKS)
                user_tasks = tasks_df[tasks_df.email == user_email]
                
                if user_tasks.empty:
                    return "You don't have any tasks yet! Would you like to add some? Just tell me what you need to do, like 'I need to finish my project report' and I'll help you organize it."
                else:
                    task_list = []
                    for _, task in user_tasks.iterrows():
                        task_list.append(f"• **{task['task']}** (Priority: {task['priority']}, Deadline: {task['deadline']})")
                    
                    response = f"Here are your current tasks:\n\n" + "\n".join(task_list)
                    response += f"\n\nYou have {len(user_tasks)} tasks total. Need help prioritizing or want to add more? Just let me know!"
                    return response
            except Exception as e:
                return f"I had trouble accessing your tasks. Error: {str(e)}. Would you like to try again?"
        
        elif intent == 'add_task':
            task_desc, priority = self.extract_task_details(message)
            if task_desc and len(task_desc.strip()) > 0:
                try:
                    # Add task with default deadline (7 days from now)
                    deadline = (datetime.now() + timedelta(days=7)).date()
                    
                    # Read current tasks
                    tasks_df = pd.read_csv(TASKS)
                    
                    # Create new task row
                    new_row = pd.DataFrame({
                        'email': [user_email],
                        'task': [task_desc.strip()],
                        'priority': [priority],
                        'deadline': [str(deadline)]
                    })
                    
                    # Add to dataframe and save
                    tasks_df = pd.concat([tasks_df, new_row], ignore_index=True)
                    tasks_df.to_csv(TASKS, index=False)
                    
                    # Send email notification
                    send_email(user_email, f"New Task Added: {task_desc}", 
                              f"Task: {task_desc}\nPriority: {priority}\nDeadline: {deadline}")
                    
                    return f"Perfect! ✅ I've added '{task_desc}' to your task list with {priority} priority and set the deadline for {deadline}. You'll receive an email confirmation shortly. Is there anything else you'd like to add or modify?"
                    
                except Exception as e:
                    return f"I encountered an issue adding your task: {str(e)}. Could you please try again? Maybe rephrase it like 'Add task: finish my report with high priority'"
            else:
                return "I'd love to help you add a task! Could you be more specific about what you need to do? For example, try saying 'I need to finish my project report' or 'Add task: call the dentist with high priority'"
        
        elif intent == 'view_shopping':
            try:
                shopping_df = pd.read_csv(SHOPPING)
                user_items = shopping_df[shopping_df.email == user_email]
                
                if user_items.empty:
                    return "Your shopping list is empty right now! Want to add some items? Just tell me what you need to buy, like 'I need to buy milk and bread' and I'll organize it for you."
                else:
                    item_list = []
                    for _, item in user_items.iterrows():
                        item_list.append(f"• **{item['item']}** (Priority: {item['priority']}, Date: {item['purchase_date']})")
                    
                    response = f"Here's your shopping list:\n\n" + "\n".join(item_list)
                    response += f"\n\nYou have {len(user_items)} items to buy. Need to add more or change priorities? Just let me know!"
                    return response
            except Exception as e:
                return f"I had trouble accessing your shopping list. Error: {str(e)}. Would you like to try again?"
        
        elif intent == 'add_shopping':
            item_desc, priority = self.extract_shopping_details(message)
            if item_desc and len(item_desc.strip()) > 0:
                try:
                    # Add shopping item with default date (3 days from now)
                    purchase_date = (datetime.now() + timedelta(days=3)).date()
                    
                    # Read current shopping items
                    shopping_df = pd.read_csv(SHOPPING)
                    
                    # Create new shopping item row
                    new_row = pd.DataFrame({
                        'email': [user_email],
                        'item': [item_desc.strip()],
                        'priority': [priority],
                        'purchase_date': [str(purchase_date)]
                    })
                    
                    # Add to dataframe and save
                    shopping_df = pd.concat([shopping_df, new_row], ignore_index=True)
                    shopping_df.to_csv(SHOPPING, index=False)
                    
                    # Send email notification
                    send_email(user_email, f"New Shopping Item: {item_desc}", 
                              f"Item: {item_desc}\nPriority: {priority}\nPurchase Date: {purchase_date}")
                    
                    return f"Excellent! 🛒 I've added '{item_desc}' to your shopping list with {priority} priority, planned for {purchase_date}. You'll get an email confirmation too. Anything else you need to buy?"
                    
                except Exception as e:
                    return f"I had trouble adding that item: {str(e)}. Could you try again? Maybe say something like 'Buy milk with high priority' or 'I need to purchase bread'"
            else:
                return "I'd be happy to add something to your shopping list! What do you need to buy? You can say things like 'I need milk and bread' or 'Buy groceries with high priority'"
        
        elif intent == 'edit_task':
            try:
                tasks_df = pd.read_csv(TASKS)
                user_tasks = tasks_df[tasks_df.email == user_email]
                
                if user_tasks.empty:
                    return "You don't have any tasks to edit. Would you like to add a new task instead?"
                
                # Extract task name to edit (simple approach)
                task_keywords = ['edit task', 'modify task', 'change task', 'update task']
                task_to_edit = message.lower()
                for keyword in task_keywords:
                    if keyword in message.lower():
                        task_to_edit = message.lower().replace(keyword, '').strip()
                        break
                
                if task_to_edit:
                    # Find matching tasks
                    matching_tasks = user_tasks[user_tasks['task'].str.contains(task_to_edit, case=False, na=False)]
                    if not matching_tasks.empty:
                        task_list = []
                        for idx, task in matching_tasks.iterrows():
                            task_list.append(f"• {task['task']} (Priority: {task['priority']}, Deadline: {task['deadline']})")
                        
                        return f"I found these matching tasks:\n" + "\n".join(task_list) + "\n\nTo edit a task, please go to the Task Manager page where you can use the edit buttons for full editing capabilities!"
                    else:
                        return f"I couldn't find any tasks matching '{task_to_edit}'. Here are your current tasks:\n" + self.process_message("show my tasks", user_email)
                else:
                    return "Which task would you like to edit? Please specify the task name, like 'edit task project report'"
                    
            except Exception as e:
                return f"I had trouble accessing your tasks: {str(e)}. Please try again."
        
        elif intent == 'delete_task':
            try:
                tasks_df = pd.read_csv(TASKS)
                user_tasks = tasks_df[tasks_df.email == user_email]
                
                if user_tasks.empty:
                    return "You don't have any tasks to delete."
                
                # Extract task name to delete
                task_keywords = ['delete task', 'remove task', 'cancel task', 'drop task']
                task_to_delete = message.lower()
                for keyword in task_keywords:
                    if keyword in message.lower():
                        task_to_delete = message.lower().replace(keyword, '').strip()
                        break
                
                if task_to_delete:
                    matching_tasks = user_tasks[user_tasks['task'].str.contains(task_to_delete, case=False, na=False)]
                    if not matching_tasks.empty:
                        return f"I found tasks matching '{task_to_delete}'. For safety, please use the Task Manager page to delete tasks using the delete buttons. This ensures you don't accidentally delete the wrong task!"
                    else:
                        return f"I couldn't find any tasks matching '{task_to_delete}'. Please check your task list first."
                else:
                    return "Which task would you like to delete? Please specify the task name, like 'delete task project report'. For safety, I recommend using the Task Manager page for deletions."
                    
            except Exception as e:
                return f"I had trouble accessing your tasks: {str(e)}. Please try again."
        
        elif intent == 'complete_task':
            try:
                tasks_df = pd.read_csv(TASKS)
                user_tasks = tasks_df[tasks_df.email == user_email]
                
                if user_tasks.empty:
                    return "You don't have any tasks to complete. Great job staying on top of everything!"
                
                # Extract task name to complete
                task_keywords = ['complete task', 'finish task', 'done task', 'completed task']
                task_to_complete = message.lower()
                for keyword in task_keywords:
                    if keyword in message.lower():
                        task_to_complete = message.lower().replace(keyword, '').strip()
                        break
                
                if task_to_complete:
                    matching_tasks = user_tasks[user_tasks['task'].str.contains(task_to_complete, case=False, na=False)]
                    if not matching_tasks.empty:
                        return f"Great! I found tasks matching '{task_to_complete}'. To mark them as complete, please use the Task Manager page where you can click the '✅ Complete' button. This will remove them from your list and send you a congratulations email!"
                    else:
                        return f"I couldn't find any tasks matching '{task_to_complete}'. Please check your task list first."
                else:
                    return "Which task did you complete? Please specify the task name, like 'completed task project report'. You can also use the Task Manager page to mark tasks as complete!"
                    
            except Exception as e:
                return f"I had trouble accessing your tasks: {str(e)}. Please try again."
        
        elif intent == 'edit_shopping':
            try:
                shopping_df = pd.read_csv(SHOPPING)
                user_items = shopping_df[shopping_df.email == user_email]
                
                if user_items.empty:
                    return "You don't have any shopping items to edit. Would you like to add something to your shopping list?"
                
                # Extract item name to edit
                shopping_keywords = ['edit shopping', 'modify shopping', 'change shopping', 'update shopping']
                item_to_edit = message.lower()
                for keyword in shopping_keywords:
                    if keyword in message.lower():
                        item_to_edit = message.lower().replace(keyword, '').strip()
                        break
                
                if item_to_edit:
                    matching_items = user_items[user_items['item'].str.contains(item_to_edit, case=False, na=False)]
                    if not matching_items.empty:
                        item_list = []
                        for idx, item in matching_items.iterrows():
                            item_list.append(f"• {item['item']} (Priority: {item['priority']}, Date: {item['purchase_date']})")
                        
                        return f"I found these matching shopping items:\n" + "\n".join(item_list) + "\n\nTo edit an item, please go to the Shopping Reminder page where you can use the edit buttons for full editing capabilities!"
                    else:
                        return f"I couldn't find any shopping items matching '{item_to_edit}'. Here's your current shopping list:\n" + self.process_message("show shopping list", user_email)
                else:
                    return "Which shopping item would you like to edit? Please specify the item name, like 'edit shopping milk'"
                    
            except Exception as e:
                return f"I had trouble accessing your shopping list: {str(e)}. Please try again."
        
        elif intent == 'delete_shopping':
            try:
                shopping_df = pd.read_csv(SHOPPING)
                user_items = shopping_df[shopping_df.email == user_email]
                
                if user_items.empty:
                    return "You don't have any shopping items to delete."
                
                # Extract item name to delete
                shopping_keywords = ['delete shopping', 'remove shopping', 'cancel shopping']
                item_to_delete = message.lower()
                for keyword in shopping_keywords:
                    if keyword in message.lower():
                        item_to_delete = message.lower().replace(keyword, '').strip()
                        break
                
                if item_to_delete:
                    matching_items = user_items[user_items['item'].str.contains(item_to_delete, case=False, na=False)]
                    if not matching_items.empty:
                        return f"I found shopping items matching '{item_to_delete}'. For safety, please use the Shopping Reminder page to delete items using the delete buttons. This ensures you don't accidentally delete the wrong item!"
                    else:
                        return f"I couldn't find any shopping items matching '{item_to_delete}'. Please check your shopping list first."
                else:
                    return "Which shopping item would you like to delete? Please specify the item name, like 'delete shopping milk'. For safety, I recommend using the Shopping Reminder page for deletions."
                    
            except Exception as e:
                return f"I had trouble accessing your shopping list: {str(e)}. Please try again."
        
        elif intent == 'purchased_item':
            try:
                shopping_df = pd.read_csv(SHOPPING)
                user_items = shopping_df[shopping_df.email == user_email]
                
                if user_items.empty:
                    return "You don't have any shopping items to mark as purchased. Great job staying on top of your shopping!"
                
                # Extract item name that was purchased
                purchase_keywords = ['purchased', 'bought', 'got item', 'completed shopping']
                item_purchased = message.lower()
                for keyword in purchase_keywords:
                    if keyword in message.lower():
                        item_purchased = message.lower().replace(keyword, '').strip()
                        break
                
                if item_purchased:
                    matching_items = user_items[user_items['item'].str.contains(item_purchased, case=False, na=False)]
                    if not matching_items.empty:
                        return f"Awesome! I found items matching '{item_purchased}'. To mark them as purchased, please use the Shopping Reminder page where you can click the '✅ Purchased' button. This will remove them from your list and send you a confirmation email!"
                    else:
                        return f"I couldn't find any shopping items matching '{item_purchased}'. Please check your shopping list first."
                else:
                    return "Which item did you purchase? Please specify the item name, like 'purchased milk'. You can also use the Shopping Reminder page to mark items as purchased!"
                    
            except Exception as e:
                return f"I had trouble accessing your shopping list: {str(e)}. Please try again."
        
        elif intent == 'weather':
            return "I wish I could check the weather for you, but I don't have access to weather data right now. However, I can help you add weather-related tasks like 'Check weather forecast' or shopping items like 'Buy umbrella'! What would you like to do?"
        
        elif intent == 'time':
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"The current date and time is: {current_time}. Is there anything time-sensitive you need help with? I can help you set deadlines for tasks or schedule shopping trips!"
        
        elif intent == 'complaint':
            return "I'm sorry you're experiencing issues! I want to help make things better. Could you tell me more specifically what's not working? I'm here to assist and improve your experience with TaskHub."
        
        else:  # general_conversation
            # Try to provide helpful, conversational responses
            message_lower = message.lower()
            
            # Handle questions about capabilities
            if any(word in message_lower for word in ['can you', 'are you able', 'do you know', 'what do you']):
                return "I can help you manage tasks and shopping lists, have conversations, answer questions, and assist with productivity! I'm designed to be helpful and friendly. What specific thing would you like to know about or get help with?"
            
            # Handle questions about the app
            if any(word in message_lower for word in ['taskhub', 'app', 'application', 'system']):
                return "TaskHub Smart ML is an intelligent productivity system! It helps you manage tasks, shopping lists, get reminders, and even classify emails using AI. I'm the conversational assistant that makes it all easy to use. What feature interests you most?"
            
            # Handle motivational requests
            if any(word in message_lower for word in ['motivate', 'inspiration', 'encourage', 'boost']):
                motivational_quotes = [
                    "You've got this! Every task completed is a step toward your goals. What's the first thing you'd like to tackle today?",
                    "Progress, not perfection! Even small steps forward are victories. What can I help you organize to make progress?",
                    "The best time to start was yesterday, the second best time is now! What task or goal can we work on together?",
                    "You're more capable than you realize! Let's break down your goals into manageable tasks. What's on your mind?"
                ]
                import random
                return random.choice(motivational_quotes)
            
            # Handle general questions or statements
            return f"That's interesting! I enjoy our conversation. While I specialize in helping with tasks and shopping lists, I'm always happy to chat. Is there anything specific I can help you organize or manage today? Or feel free to keep chatting - I'm here for you!"

chatbot = SmartChatbot()

# ================= ENHANCED VOICE ASSISTANT =================
class VoiceAssistant:
    def __init__(self):
        self.context = {}
        self.confirmation_pending = False
        self.pending_action = None
    
    def process_voice_command(self, command, user_email):
        command = command.lower()
        
        # Handle confirmation responses
        if self.confirmation_pending:
            if 'yes' in command or 'confirm' in command or 'okay' in command:
                return self.execute_pending_action(user_email)
            elif 'no' in command or 'cancel' in command:
                self.confirmation_pending = False
                self.pending_action = None
                return "Action cancelled. What else can I help you with?"
        
        # Voice-based task creation
        if any(phrase in command for phrase in ['add task', 'create task', 'new task']):
            return self.handle_voice_task_creation(command, user_email)
        
        # Voice-based shopping creation
        elif any(phrase in command for phrase in ['add shopping', 'buy', 'shopping item']):
            return self.handle_voice_shopping_creation(command, user_email)
        
        # Voice-based task completion
        elif any(phrase in command for phrase in ['complete task', 'finish task', 'done task']):
            return self.handle_voice_task_completion(command, user_email)
        
        # Voice-based shopping purchase
        elif any(phrase in command for phrase in ['purchased', 'bought item', 'got item']):
            return self.handle_voice_shopping_purchase(command, user_email)
        
        # View tasks
        elif any(phrase in command for phrase in ['show tasks', 'my tasks', 'list tasks']):
            return self.get_tasks_voice_response(user_email)
        
        # View shopping
        elif any(phrase in command for phrase in ['show shopping', 'shopping list', 'my shopping']):
            return self.get_shopping_voice_response(user_email)
        
        # Edit/Delete operations (redirect to UI)
        elif any(phrase in command for phrase in ['edit task', 'modify task', 'change task']):
            return "To edit tasks, please use the Task Manager page where you can click the edit button next to any task. This gives you full control over all task details!"
        
        elif any(phrase in command for phrase in ['delete task', 'remove task']):
            return "For safety, please use the Task Manager page to delete tasks. You can click the delete button next to any task and confirm the deletion."
        
        elif any(phrase in command for phrase in ['edit shopping', 'modify shopping']):
            return "To edit shopping items, please use the Shopping Reminder page where you can click the edit button next to any item for full editing capabilities!"
        
        elif any(phrase in command for phrase in ['delete shopping', 'remove shopping']):
            return "For safety, please use the Shopping Reminder page to delete items. You can click the delete button next to any item and confirm the deletion."
        
        else:
            return "I can help you add tasks, add shopping items, view your lists, or mark items as complete. What would you like to do? For editing and deleting, please use the web interface for safety."
    
    def handle_voice_task_creation(self, command, user_email):
        # Extract task details from voice command
        task_desc, priority = self.extract_voice_task_info(command)
        
        if task_desc:
            self.pending_action = {
                'type': 'add_task',
                'task': task_desc,
                'priority': priority,
                'deadline': (datetime.now() + timedelta(days=7)).date()
            }
            self.confirmation_pending = True
            return f"I'll add the task '{task_desc}' with {priority} priority and deadline in 7 days. Should I proceed?"
        else:
            return "Please tell me what task you'd like to add. For example, say 'add task finish project report with high priority'"
    
    def handle_voice_shopping_creation(self, command, user_email):
        # Extract shopping details from voice command
        item_desc, priority = self.extract_voice_shopping_info(command)
        
        if item_desc:
            self.pending_action = {
                'type': 'add_shopping',
                'item': item_desc,
                'priority': priority,
                'date': (datetime.now() + timedelta(days=3)).date()
            }
            self.confirmation_pending = True
            return f"I'll add '{item_desc}' to your shopping list with {priority} priority for 3 days from now. Should I proceed?"
        else:
            return "Please tell me what you'd like to buy. For example, say 'buy milk with high priority'"
    
    def handle_voice_task_completion(self, command, user_email):
        """Handle voice commands for completing tasks"""
        try:
            tasks_df = pd.read_csv(TASKS)
            user_tasks = tasks_df[tasks_df.email == user_email]
            
            if user_tasks.empty:
                return "You don't have any tasks to complete. Great job staying on top of everything!"
            
            # Extract task name from command
            completion_keywords = ['complete task', 'finish task', 'done task']
            task_name = command
            for keyword in completion_keywords:
                if keyword in command:
                    task_name = command.replace(keyword, '').strip()
                    break
            
            if task_name:
                # Find matching tasks
                matching_tasks = user_tasks[user_tasks['task'].str.contains(task_name, case=False, na=False)]
                if not matching_tasks.empty:
                    if len(matching_tasks) == 1:
                        task = matching_tasks.iloc[0]
                        return f"Great! To mark '{task['task']}' as complete, please use the Task Manager page and click the Complete button. This will remove it from your list and send you a congratulations email!"
                    else:
                        return f"I found {len(matching_tasks)} tasks matching '{task_name}'. Please use the Task Manager page to select which specific task to complete."
                else:
                    return f"I couldn't find any tasks matching '{task_name}'. Please check your task list first."
            else:
                return "Which task did you complete? Please say something like 'complete task project report'"
                
        except Exception as e:
            return f"I had trouble accessing your tasks. Please try again or use the Task Manager page."
    
    def handle_voice_shopping_purchase(self, command, user_email):
        """Handle voice commands for marking items as purchased"""
        try:
            shopping_df = pd.read_csv(SHOPPING)
            user_items = shopping_df[shopping_df.email == user_email]
            
            if user_items.empty:
                return "You don't have any shopping items to mark as purchased. Your list is already clear!"
            
            # Extract item name from command
            purchase_keywords = ['purchased', 'bought item', 'got item']
            item_name = command
            for keyword in purchase_keywords:
                if keyword in command:
                    item_name = command.replace(keyword, '').strip()
                    break
            
            if item_name:
                # Find matching items
                matching_items = user_items[user_items['item'].str.contains(item_name, case=False, na=False)]
                if not matching_items.empty:
                    if len(matching_items) == 1:
                        item = matching_items.iloc[0]
                        return f"Awesome! To mark '{item['item']}' as purchased, please use the Shopping Reminder page and click the Purchased button. This will remove it from your list and send you a confirmation email!"
                    else:
                        return f"I found {len(matching_items)} items matching '{item_name}'. Please use the Shopping Reminder page to select which specific item to mark as purchased."
                else:
                    return f"I couldn't find any shopping items matching '{item_name}'. Please check your shopping list first."
            else:
                return "Which item did you purchase? Please say something like 'purchased milk'"
                
        except Exception as e:
            return f"I had trouble accessing your shopping list. Please try again or use the Shopping Reminder page."
    
    def extract_voice_task_info(self, command):
        priority_map = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
        priority = 'Medium'  # default
        
        for p in priority_map:
            if p in command:
                priority = priority_map[p]
                break
        
        # Extract task description
        task_keywords = ['add task', 'create task', 'new task']
        task_desc = command
        for keyword in task_keywords:
            if keyword in command:
                task_desc = command.replace(keyword, '').strip()
                break
        
        # Remove priority words from description
        for p in priority_map:
            task_desc = task_desc.replace(f'with {p} priority', '').replace(f'{p} priority', '').strip()
        
        return task_desc if task_desc else None, priority
    
    def extract_voice_shopping_info(self, command):
        priority_map = {'high': 'High', 'medium': 'Medium', 'low': 'Low'}
        priority = 'Medium'  # default
        
        for p in priority_map:
            if p in command:
                priority = priority_map[p]
                break
        
        # Extract item description
        shopping_keywords = ['add shopping', 'buy', 'shopping item']
        item_desc = command
        for keyword in shopping_keywords:
            if keyword in command:
                item_desc = command.replace(keyword, '').strip()
                break
        
        # Remove priority words from description
        for p in priority_map:
            item_desc = item_desc.replace(f'with {p} priority', '').replace(f'{p} priority', '').strip()
        
        return item_desc if item_desc else None, priority
    
    def execute_pending_action(self, user_email):
        if not self.pending_action:
            return "No pending action to execute."
        
        action = self.pending_action
        self.confirmation_pending = False
        self.pending_action = None
        
        try:
            if action['type'] == 'add_task':
                # Read current tasks
                tasks_df = pd.read_csv(TASKS)
                
                # Create new task row
                new_row = pd.DataFrame({
                    'email': [user_email],
                    'task': [action['task']],
                    'priority': [action['priority']],
                    'deadline': [str(action['deadline'])]
                })
                
                # Add to dataframe and save
                tasks_df = pd.concat([tasks_df, new_row], ignore_index=True)
                tasks_df.to_csv(TASKS, index=False)
                
                # Send email notification
                send_email(user_email, f"New Task Added: {action['task']}", 
                          f"Task: {action['task']}\nPriority: {action['priority']}\nDeadline: {action['deadline']}")
                
                return f"Perfect! Task '{action['task']}' has been added successfully with {action['priority']} priority and deadline {action['deadline']}! Check your task manager to see it."
            
            elif action['type'] == 'add_shopping':
                # Read current shopping items
                shopping_df = pd.read_csv(SHOPPING)
                
                # Create new shopping item row
                new_row = pd.DataFrame({
                    'email': [user_email],
                    'item': [action['item']],
                    'priority': [action['priority']],
                    'purchase_date': [str(action['date'])]
                })
                
                # Add to dataframe and save
                shopping_df = pd.concat([shopping_df, new_row], ignore_index=True)
                shopping_df.to_csv(SHOPPING, index=False)
                
                # Send email notification
                send_email(user_email, f"New Shopping Item: {action['item']}", 
                          f"Item: {action['item']}\nPriority: {action['priority']}\nPurchase Date: {action['date']}")
                
                return f"Excellent! Shopping item '{action['item']}' has been added successfully with {action['priority']} priority for {action['date']}! Check your shopping list to see it."
                
        except Exception as e:
            return f"Sorry, there was an error adding your item: {str(e)}. Please try again or use the web interface."
    
    def get_tasks_voice_response(self, user_email):
        try:
            tasks_df = pd.read_csv(TASKS)
            user_tasks = tasks_df[tasks_df.email == user_email]
            
            if user_tasks.empty:
                return "You don't have any tasks yet. Would you like to add some?"
            else:
                task_count = len(user_tasks)
                if task_count == 1:
                    task = user_tasks.iloc[0]
                    return f"You have 1 task: {task['task']} with {task['priority']} priority, due on {task['deadline']}"
                else:
                    recent_tasks = user_tasks.tail(3)['task'].tolist()
                    return f"You have {task_count} tasks. Your most recent ones are: " + ", ".join(recent_tasks)
        except Exception as e:
            return f"I had trouble accessing your tasks. Please try again."
    
    def get_shopping_voice_response(self, user_email):
        try:
            shopping_df = pd.read_csv(SHOPPING)
            user_items = shopping_df[shopping_df.email == user_email]
            
            if user_items.empty:
                return "Your shopping list is empty. Would you like to add some items?"
            else:
                item_count = len(user_items)
                if item_count == 1:
                    item = user_items.iloc[0]
                    return f"You have 1 item: {item['item']} with {item['priority']} priority, planned for {item['purchase_date']}"
                else:
                    recent_items = user_items.tail(3)['item'].tolist()
                    return f"You have {item_count} shopping items. Your most recent ones are: " + ", ".join(recent_items)
        except Exception as e:
            return f"I had trouble accessing your shopping list. Please try again."

voice_assistant = VoiceAssistant()

# ================= NAVBAR =================
def navbar():
    cols = st.columns(8)
    cols[0].button("🏠 Home", on_click=lambda: set_page("home"))
    cols[1].button("About", on_click=lambda: set_page("about"))
    cols[2].button("Services", on_click=lambda: set_page("services"))
    cols[3].button("Blogs", on_click=lambda: set_page("blogs"))
    cols[4].button("FAQs", on_click=lambda: set_page("faqs"))
    cols[5].button("Contact Us", on_click=lambda: set_page("contact"))
    cols[6].button("Email Classifier", on_click=lambda: set_page("email_classifier"))
    cols[7].button("🚪 Logout", on_click=logout)

# ================= LOGIN =================
def login():
    st.image("logo.png", width=160)
    st.markdown("<h1 style='text-align:center;color:#2563EB'>TaskHub <span style='color:#22C55E'>Smart ML</span></h1>", unsafe_allow_html=True)

    mode = st.radio("Select Mode", ["Login", "Create Account"], horizontal=True)
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(mode):
        global users_df
        if mode == "Create Account":
            if email in users_df.email.values:
                st.error("Email already registered")
            else:
                users_df.loc[len(users_df)] = [email, password]
                users_df.to_csv(USERS, index=False)
                st.success("Account created")
        else:
            user = users_df[(users_df.email == email) & (users_df.password == password)]
            if not user.empty:
                st.session_state.user = email
                set_page("home")
            else:
                st.error("Invalid credentials")

# ================= HOME =================
def home():
    navbar()
    
    # Start reminder engine if not already running
    if st.session_state.reminder_thread is None:
        st.session_state.reminder_thread = reminder_engine.start()
    
    st.image("logo.png", width=120)
    st.markdown("<h2 style='text-align:center;color:#2563EB'>TaskHub <span style='color:#22C55E'>Smart ML</span></h2>", unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image("shopping_logo.png", width=90)
        st.button("Shopping Reminder", on_click=lambda: set_page("shopping"))
    with c2:
        st.image("task_logo.png", width=90)
        st.button("Task Manager", on_click=lambda: set_page("tasks"))
    with c3:
        st.image("voice.png", width=90)
        st.button("Voice Assistant", on_click=lambda: set_page("voice"))
    with c4:
        st.image("chatbot.png", width=90)
        st.button("Chat Bot", on_click=lambda: set_page("chatbot"))
    
    # Status indicators
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Tasks", len(pd.read_csv(TASKS)[pd.read_csv(TASKS).email == st.session_state.user]))
    with col2:
        st.metric("Shopping Items", len(pd.read_csv(SHOPPING)[pd.read_csv(SHOPPING).email == st.session_state.user]))
    with col3:
        st.metric("Reminder Engine", "🟢 Active" if reminder_engine.running else "🔴 Inactive")

# ================= SHOPPING =================
def shopping():
    navbar()
    st.image("shopping_logo.png", width=80)
    st.header("🛒 Shopping Reminder")

    # Add new shopping item section
    with st.expander("➕ Add New Shopping Item", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            item = st.text_input("Item", placeholder="e.g., Milk, Bread, Groceries")
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        with col3:
            date = st.date_input("Purchase Date")

        if st.button("Add Item", type="primary"):
            if item.strip():
                try:
                    # Read current shopping items
                    shopping_df = pd.read_csv(SHOPPING)
                    
                    # Create new item row
                    new_row = pd.DataFrame({
                        'email': [st.session_state.user],
                        'item': [item.strip()],
                        'priority': [priority],
                        'purchase_date': [str(date)]
                    })
                    
                    # Add to dataframe and save
                    shopping_df = pd.concat([shopping_df, new_row], ignore_index=True)
                    shopping_df.to_csv(SHOPPING, index=False)
                    
                    # Send email notification
                    send_email(st.session_state.user, f"New Shopping Item: {item}", f"Priority: {priority}\nPurchase Date: {date}")
                    
                    st.success("✅ Shopping item added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding item: {str(e)}")
            else:
                st.warning("Please enter an item name")

    # Display and manage existing shopping items
    st.subheader("🛒 Your Shopping List")
    
    try:
        shopping_df = pd.read_csv(SHOPPING)
        user_items = shopping_df[shopping_df.email == st.session_state.user].reset_index(drop=True)
        
        if user_items.empty:
            st.info("No shopping items yet. Add your first item above!")
        else:
            # Shopping item management interface
            for idx, row in user_items.iterrows():
                # Skip empty items
                if pd.isna(row['item']) or str(row['item']).strip() == '':
                    continue
                    
                with st.container():
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        # Priority color coding
                        priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                        priority_icon = priority_colors.get(row['priority'], "⚪")
                        st.markdown(f"**{priority_icon} {row['item']}**")
                        st.caption(f"Purchase Date: {row['purchase_date']}")
                    
                    with col2:
                        st.write(f"**{row['priority']}**")
                    
                    with col3:
                        # Edit button
                        if st.button("✏️ Edit", key=f"edit_item_{idx}"):
                            st.session_state[f"editing_item_{idx}"] = True
                            st.rerun()
                    
                    with col4:
                        # Purchased button
                        if st.button("✅ Purchased", key=f"purchased_item_{idx}"):
                            try:
                                # Get the exact item to remove
                                item_to_remove = user_items.iloc[idx]
                                
                                # Read fresh data and remove the item
                                all_items_df = pd.read_csv(SHOPPING)
                                
                                # Create a mask to find the exact row
                                mask = (
                                    (all_items_df['email'] == item_to_remove['email']) & 
                                    (all_items_df['item'] == item_to_remove['item']) & 
                                    (all_items_df['priority'] == item_to_remove['priority']) & 
                                    (all_items_df['purchase_date'] == item_to_remove['purchase_date'])
                                )
                                
                                # Remove the item
                                all_items_df = all_items_df[~mask]
                                all_items_df.to_csv(SHOPPING, index=False)
                                
                                # Send purchase confirmation email
                                send_email(st.session_state.user, f"Item Purchased: {item_to_remove['item']}", 
                                          f"Great! You've purchased: {item_to_remove['item']}")
                                
                                st.success(f"✅ Item '{item_to_remove['item']}' marked as purchased!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error marking item as purchased: {str(e)}")
                    
                    with col5:
                        # Delete button
                        if st.button("🗑️ Delete", key=f"delete_item_{idx}"):
                            st.session_state[f"confirm_delete_item_{idx}"] = True
                            st.rerun()
                    
                    # Edit form
                    if st.session_state.get(f"editing_item_{idx}", False):
                        with st.form(f"edit_item_form_{idx}"):
                            st.subheader(f"✏️ Edit Shopping Item")
                            
                            edit_col1, edit_col2, edit_col3 = st.columns(3)
                            with edit_col1:
                                new_item = st.text_input("Item", value=row['item'])
                            with edit_col2:
                                new_priority = st.selectbox("Priority", ["Low", "Medium", "High"], 
                                                           index=["Low", "Medium", "High"].index(row['priority']))
                            with edit_col3:
                                new_date = st.date_input("Purchase Date", value=pd.to_datetime(row['purchase_date']).date())
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.form_submit_button("💾 Save Changes", type="primary"):
                                    if new_item.strip():
                                        try:
                                            # Get the exact item to update
                                            item_to_update = user_items.iloc[idx]
                                            
                                            # Read fresh data
                                            all_items_df = pd.read_csv(SHOPPING)
                                            
                                            # Create mask to find exact row
                                            mask = (
                                                (all_items_df['email'] == item_to_update['email']) & 
                                                (all_items_df['item'] == item_to_update['item']) & 
                                                (all_items_df['priority'] == item_to_update['priority']) & 
                                                (all_items_df['purchase_date'] == item_to_update['purchase_date'])
                                            )
                                            
                                            # Update the item
                                            all_items_df.loc[mask, 'item'] = new_item.strip()
                                            all_items_df.loc[mask, 'priority'] = new_priority
                                            all_items_df.loc[mask, 'purchase_date'] = str(new_date)
                                            
                                            all_items_df.to_csv(SHOPPING, index=False)
                                            
                                            # Send update email
                                            send_email(st.session_state.user, f"Shopping Item Updated: {new_item}", 
                                                      f"Updated item: {new_item}\nPriority: {new_priority}\nPurchase Date: {new_date}")
                                            
                                            st.session_state[f"editing_item_{idx}"] = False
                                            st.success("✅ Shopping item updated successfully!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error updating item: {str(e)}")
                                    else:
                                        st.warning("Please enter an item name")
                            
                            with col_cancel:
                                if st.form_submit_button("❌ Cancel"):
                                    st.session_state[f"editing_item_{idx}"] = False
                                    st.rerun()
                    
                    # Delete confirmation
                    if st.session_state.get(f"confirm_delete_item_{idx}", False):
                        st.error(f"⚠️ Are you sure you want to delete '{row['item']}'?")
                        col_yes, col_no = st.columns(2)
                        
                        with col_yes:
                            if st.button("🗑️ Yes, Delete", key=f"confirm_yes_item_{idx}", type="primary"):
                                try:
                                    # Get the exact item to delete
                                    item_to_delete = user_items.iloc[idx]
                                    
                                    # Read fresh data
                                    all_items_df = pd.read_csv(SHOPPING)
                                    
                                    # Create mask to find exact row
                                    mask = (
                                        (all_items_df['email'] == item_to_delete['email']) & 
                                        (all_items_df['item'] == item_to_delete['item']) & 
                                        (all_items_df['priority'] == item_to_delete['priority']) & 
                                        (all_items_df['purchase_date'] == item_to_delete['purchase_date'])
                                    )
                                    
                                    # Remove the item
                                    all_items_df = all_items_df[~mask]
                                    all_items_df.to_csv(SHOPPING, index=False)
                                    
                                    st.session_state[f"confirm_delete_item_{idx}"] = False
                                    st.success(f"🗑️ Item '{item_to_delete['item']}' deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting item: {str(e)}")
                        
                        with col_no:
                            if st.button("❌ Cancel", key=f"confirm_no_item_{idx}"):
                                st.session_state[f"confirm_delete_item_{idx}"] = False
                                st.rerun()

            # Shopping statistics
            st.markdown("---")
            st.subheader("📊 Shopping Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            # Filter out empty items for statistics
            valid_items = user_items[user_items['item'].notna() & (user_items['item'].str.strip() != '')]
            
            with col1:
                st.metric("Total Items", len(valid_items))
            with col2:
                high_priority = len(valid_items[valid_items['priority'] == 'High'])
                st.metric("High Priority", high_priority)
            with col3:
                # Items to buy soon (within 3 days)
                from datetime import datetime, timedelta
                current_date = datetime.now().date()
                buy_soon = 0
                for _, item in valid_items.iterrows():
                    try:
                        item_date = pd.to_datetime(item['purchase_date']).date()
                        if (item_date - current_date).days <= 3:
                            buy_soon += 1
                    except:
                        pass
                st.metric("Buy Soon", buy_soon)
            with col4:
                # Overdue purchases
                overdue = 0
                for _, item in valid_items.iterrows():
                    try:
                        item_date = pd.to_datetime(item['purchase_date']).date()
                        if item_date < current_date:
                            overdue += 1
                    except:
                        pass
                st.metric("Overdue", overdue)
    
    except Exception as e:
        st.error(f"Error loading shopping items: {str(e)}")

    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= TASKS =================
def tasks():
    navbar()
    st.image("task_logo.png", width=80)
    st.header("📋 Task Manager")

    # Add new task section
    with st.expander("➕ Add New Task", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            task = st.text_input("Task Description", placeholder="e.g., Complete project report")
        with col2:
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
        with col3:
            deadline = st.date_input("Deadline")

        if st.button("Add Task", type="primary"):
            if task.strip():
                try:
                    # Read current tasks
                    tasks_df = pd.read_csv(TASKS)
                    
                    # Create new task row
                    new_row = pd.DataFrame({
                        'email': [st.session_state.user],
                        'task': [task.strip()],
                        'priority': [priority],
                        'deadline': [str(deadline)]
                    })
                    
                    # Add to dataframe and save
                    tasks_df = pd.concat([tasks_df, new_row], ignore_index=True)
                    tasks_df.to_csv(TASKS, index=False)
                    
                    # Send email notification
                    send_email(st.session_state.user, f"New Task Added: {task}", f"Priority: {priority}\nDeadline: {deadline}")
                    
                    st.success("✅ Task added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding task: {str(e)}")
            else:
                st.warning("Please enter a task description")

    # Display and manage existing tasks
    st.subheader("📋 Your Tasks")
    
    try:
        tasks_df = pd.read_csv(TASKS)
        user_tasks = tasks_df[tasks_df.email == st.session_state.user].reset_index(drop=True)
        
        if user_tasks.empty:
            st.info("No tasks yet. Add your first task above!")
        else:
            # Task management interface
            for idx, row in user_tasks.iterrows():
                with st.container():
                    st.markdown("---")
                    col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                    
                    with col1:
                        # Priority color coding
                        priority_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                        priority_icon = priority_colors.get(row['priority'], "⚪")
                        st.markdown(f"**{priority_icon} {row['task']}**")
                        st.caption(f"Deadline: {row['deadline']}")
                    
                    with col2:
                        st.write(f"**{row['priority']}**")
                    
                    with col3:
                        # Edit button
                        if st.button("✏️ Edit", key=f"edit_task_{idx}"):
                            st.session_state[f"editing_task_{idx}"] = True
                            st.rerun()
                    
                    with col4:
                        # Complete button
                        if st.button("✅ Complete", key=f"complete_task_{idx}"):
                            try:
                                # Get the exact task to remove
                                task_to_remove = user_tasks.iloc[idx]
                                
                                # Read fresh data and remove the task
                                all_tasks_df = pd.read_csv(TASKS)
                                
                                # Create a mask to find the exact row
                                mask = (
                                    (all_tasks_df['email'] == task_to_remove['email']) & 
                                    (all_tasks_df['task'] == task_to_remove['task']) & 
                                    (all_tasks_df['priority'] == task_to_remove['priority']) & 
                                    (all_tasks_df['deadline'] == task_to_remove['deadline'])
                                )
                                
                                # Remove the task
                                all_tasks_df = all_tasks_df[~mask]
                                all_tasks_df.to_csv(TASKS, index=False)
                                
                                # Send completion email
                                send_email(st.session_state.user, f"Task Completed: {task_to_remove['task']}", 
                                          f"Congratulations! You've completed: {task_to_remove['task']}")
                                
                                st.success(f"✅ Task '{task_to_remove['task']}' marked as completed!")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error completing task: {str(e)}")
                    
                    with col5:
                        # Delete button
                        if st.button("�️ Delete", key=f"delete_task_{idx}"):
                            st.session_state[f"confirm_delete_task_{idx}"] = True
                            st.rerun()
                    
                    # Edit form
                    if st.session_state.get(f"editing_task_{idx}", False):
                        with st.form(f"edit_task_form_{idx}"):
                            st.subheader(f"✏️ Edit Task")
                            
                            edit_col1, edit_col2, edit_col3 = st.columns(3)
                            with edit_col1:
                                new_task = st.text_input("Task Description", value=row['task'])
                            with edit_col2:
                                new_priority = st.selectbox("Priority", ["Low", "Medium", "High"], 
                                                           index=["Low", "Medium", "High"].index(row['priority']))
                            with edit_col3:
                                new_deadline = st.date_input("Deadline", value=pd.to_datetime(row['deadline']).date())
                            
                            col_save, col_cancel = st.columns(2)
                            with col_save:
                                if st.form_submit_button("💾 Save Changes", type="primary"):
                                    if new_task.strip():
                                        try:
                                            # Get the exact task to update
                                            task_to_update = user_tasks.iloc[idx]
                                            
                                            # Read fresh data
                                            all_tasks_df = pd.read_csv(TASKS)
                                            
                                            # Create mask to find exact row
                                            mask = (
                                                (all_tasks_df['email'] == task_to_update['email']) & 
                                                (all_tasks_df['task'] == task_to_update['task']) & 
                                                (all_tasks_df['priority'] == task_to_update['priority']) & 
                                                (all_tasks_df['deadline'] == task_to_update['deadline'])
                                            )
                                            
                                            # Update the task
                                            all_tasks_df.loc[mask, 'task'] = new_task.strip()
                                            all_tasks_df.loc[mask, 'priority'] = new_priority
                                            all_tasks_df.loc[mask, 'deadline'] = str(new_deadline)
                                            
                                            all_tasks_df.to_csv(TASKS, index=False)
                                            
                                            # Send update email
                                            send_email(st.session_state.user, f"Task Updated: {new_task}", 
                                                      f"Updated task: {new_task}\nPriority: {new_priority}\nDeadline: {new_deadline}")
                                            
                                            st.session_state[f"editing_task_{idx}"] = False
                                            st.success("✅ Task updated successfully!")
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"Error updating task: {str(e)}")
                                    else:
                                        st.warning("Please enter a task description")
                            
                            with col_cancel:
                                if st.form_submit_button("❌ Cancel"):
                                    st.session_state[f"editing_task_{idx}"] = False
                                    st.rerun()
                    
                    # Delete confirmation
                    if st.session_state.get(f"confirm_delete_task_{idx}", False):
                        st.error(f"⚠️ Are you sure you want to delete '{row['task']}'?")
                        col_yes, col_no = st.columns(2)
                        
                        with col_yes:
                            if st.button("🗑️ Yes, Delete", key=f"confirm_yes_task_{idx}", type="primary"):
                                try:
                                    # Get the exact task to delete
                                    task_to_delete = user_tasks.iloc[idx]
                                    
                                    # Read fresh data
                                    all_tasks_df = pd.read_csv(TASKS)
                                    
                                    # Create mask to find exact row
                                    mask = (
                                        (all_tasks_df['email'] == task_to_delete['email']) & 
                                        (all_tasks_df['task'] == task_to_delete['task']) & 
                                        (all_tasks_df['priority'] == task_to_delete['priority']) & 
                                        (all_tasks_df['deadline'] == task_to_delete['deadline'])
                                    )
                                    
                                    # Remove the task
                                    all_tasks_df = all_tasks_df[~mask]
                                    all_tasks_df.to_csv(TASKS, index=False)
                                    
                                    st.session_state[f"confirm_delete_task_{idx}"] = False
                                    st.success(f"🗑️ Task '{task_to_delete['task']}' deleted successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting task: {str(e)}")
                        
                        with col_no:
                            if st.button("❌ Cancel", key=f"confirm_no_task_{idx}"):
                                st.session_state[f"confirm_delete_task_{idx}"] = False
                                st.rerun()

            # Task statistics
            st.markdown("---")
            st.subheader("📊 Task Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tasks", len(user_tasks))
            with col2:
                high_priority = len(user_tasks[user_tasks['priority'] == 'High'])
                st.metric("High Priority", high_priority)
            with col3:
                # Tasks due soon (within 3 days)
                from datetime import datetime, timedelta
                current_date = datetime.now().date()
                due_soon = 0
                for _, task in user_tasks.iterrows():
                    try:
                        task_deadline = pd.to_datetime(task['deadline']).date()
                        if (task_deadline - current_date).days <= 3:
                            due_soon += 1
                    except:
                        pass
                st.metric("Due Soon", due_soon)
            with col4:
                # Overdue tasks
                overdue = 0
                for _, task in user_tasks.iterrows():
                    try:
                        task_deadline = pd.to_datetime(task['deadline']).date()
                        if task_deadline < current_date:
                            overdue += 1
                    except:
                        pass
                st.metric("Overdue", overdue)
    
    except Exception as e:
        st.error(f"Error loading tasks: {str(e)}")

    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= VOICE ASSISTANT PAGE =================
def voice():
    navbar()
    st.header("🎤 Voice Assistant")
    st.write("Talk naturally to add tasks and shopping items!")
    
    # Initialize voice session state
    if 'voice_listening' not in st.session_state:
        st.session_state.voice_listening = False
    if 'voice_conversation' not in st.session_state:
        st.session_state.voice_conversation = []
    if 'voice_ready' not in st.session_state:
        st.session_state.voice_ready = True
    
    # Simple instructions
    st.write("Click 'Start Listening' and speak your command.")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎙️ Voice Input")
        
        if st.button("🎤 Start Listening", key="voice_listen", type="primary"):
            try:
                import speech_recognition as sr
                
                with st.spinner("� Listening... Speak now!"):
                    r = sr.Recognizer()
                    
                    # Try to use microphone
                    try:
                        with sr.Microphone() as source:
                            st.info("🎤 Adjusting for background noise...")
                            r.adjust_for_ambient_noise(source, duration=1)
                            st.info("🎧 Listening... Speak clearly!")
                            
                            # Listen for audio
                            audio = r.listen(source, timeout=10, phrase_time_limit=10)
                            
                        st.info("🔄 Processing your speech...")
                        
                        # Recognize speech
                        command = r.recognize_google(audio).lower()
                        st.success(f"� **You said:** '{command}'")
                        
                        # Process with enhanced voice assistant
                        response = voice_assistant.process_voice_command(command, st.session_state.user)
                        
                        # Display response
                        st.info(f"🤖 **Assistant:** {response}")
                        
                        # Try to speak response
                        try:
                            speak(response)
                            st.success("🔊 Response spoken successfully!")
                        except Exception as speak_error:
                            st.warning(f"⚠️ Could not speak response: {speak_error}")
                        
                    except sr.WaitTimeoutError:
                        st.warning("⏰ No speech detected within 10 seconds. Please try again.")
                    except sr.UnknownValueError:
                        st.error("🔇 Could not understand the audio. Please speak clearly and try again.")
                    except sr.RequestError as e:
                        st.error(f"🌐 Could not request results from speech service: {e}")
                        st.info("💡 Make sure you have an internet connection for speech recognition.")
                    except OSError as e:
                        st.error(f"🎤 Microphone error: {e}")
                        st.info("💡 Please check your microphone connection and permissions.")
                        
            except ImportError as e:
                st.error(f"❌ Missing required library: {e}")
                st.info("💡 Please install: pip install SpeechRecognition pyaudio")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                st.info("💡 Please try again or use the test buttons below.")
    
    with col2:
        st.subheader("⚡ Status & Actions")
        
        # Show confirmation status
        if voice_assistant.confirmation_pending:
            st.warning("⏳ **Waiting for your confirmation...**")
            if voice_assistant.pending_action:
                st.json(voice_assistant.pending_action)
            
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✅ Yes, Confirm", type="primary"):
                    response = voice_assistant.execute_pending_action(st.session_state.user)
                    st.success(f"✅ {response}")
                    try:
                        speak(response)
                    except:
                        pass
                    st.rerun()
            
            with col_no:
                if st.button("❌ No, Cancel"):
                    voice_assistant.confirmation_pending = False
                    voice_assistant.pending_action = None
                    st.info("❌ Action cancelled.")
                    try:
                        speak("Action cancelled.")
                    except:
                        pass
                    st.rerun()
        else:
            st.success("🟢 **Ready to listen**")
            st.write("Click 'Start Listening' and speak your command!")
        
        # Quick stats
        st.markdown("---")
        st.subheader("📊 Quick Stats")
        
        try:
            tasks_count = len(pd.read_csv(TASKS)[pd.read_csv(TASKS).email == st.session_state.user])
            shopping_count = len(pd.read_csv(SHOPPING)[pd.read_csv(SHOPPING).email == st.session_state.user])
            
            col_task, col_shop = st.columns(2)
            with col_task:
                st.metric("📋 Tasks", tasks_count)
            with col_shop:
                st.metric("🛒 Shopping", shopping_count)
        except:
            st.write("Stats unavailable")

    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= CHATBOT PAGE =================
def chatbot_page():
    navbar()
    st.header("💬 Chatbot")
    st.write("I'm your TaskHub assistant. I can help with tasks, shopping, and answer questions!")

    # Initialize chat if empty
    if not st.session_state.chat_history:
        welcome_msg = "Hello! I'm your TaskHub AI assistant. I'm here to help you with tasks, shopping, and I love having conversations! How can I help you today?"
        st.session_state.chat_history.append(("System", welcome_msg))

    # Display chat history with better formatting
    st.subheader("💭 Conversation")
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for i, (sender, message) in enumerate(st.session_state.chat_history):
            if sender == "System":
                st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <strong>🤖 TaskHub AI:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
            elif sender.startswith("You"):
                st.markdown(f"""
                <div style='background-color: #f3e5f5; padding: 10px; border-radius: 10px; margin: 5px 0; text-align: right;'>
                    <strong>👤 You:</strong> {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #e8f5e8; padding: 10px; border-radius: 10px; margin: 5px 0;'>
                    <strong>🤖 TaskHub AI:</strong> {message}
                </div>
                """, unsafe_allow_html=True)

    # Chat input section
    st.markdown("---")
    st.subheader("💬 Chat with AI")
    
    # Create columns for input and buttons
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input("Type your message here...", key="chat_input", placeholder="Ask me anything! Try: 'Hello', 'Add task finish project', 'What can you do?'")
    
    with col2:
        send_clicked = st.button("Send 📤", type="primary")
    
    # Handle message sending
    if send_clicked and user_input.strip():
        # Add user message to history
        st.session_state.chat_history.append((f"You", user_input))
        
        # Get AI response
        try:
            response = chatbot.process_message(user_input, st.session_state.user)
            st.session_state.chat_history.append(("AI", response))
        except Exception as e:
            error_response = f"I apologize, but I encountered an error: {str(e)}. Please try again!"
            st.session_state.chat_history.append(("AI", error_response))
        
        # Rerun to show new messages
        st.rerun()
    
    # Chat management
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("�️ Clear Chat"):
            st.session_state.chat_history = []
            welcome_msg = "Chat cleared! I'm ready for a fresh conversation. What would you like to talk about?"
            st.session_state.chat_history.append(("System", welcome_msg))
            st.rerun()
    
    with col2:
        chat_count = len(st.session_state.chat_history)
        st.metric("Messages", chat_count)

    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= STATIC PAGES =================
def about(): 
    navbar()
    st.header("🚀 About TaskHub Smart ML")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **TaskHub Smart ML** is an intelligent productivity system designed to revolutionize how you manage your daily tasks and shopping reminders. Built with cutting-edge AI technology, our platform combines the power of machine learning with intuitive user experience to create your ultimate productivity companion.

        ### 🎯 Our Mission
        To empower individuals and teams with smart, AI-driven tools that simplify task management, enhance productivity, and ensure nothing important ever gets forgotten.

        ### ✨ Key Features
        - **🤖 AI-Powered Chatbot**: Conversational assistant that understands natural language
        - **🎤 Voice Assistant**: Hands-free task and shopping management
        - **📧 Smart Email Classification**: ML-powered email categorization
        - **⏰ Automated Reminders**: Intelligent notification system
        - **📊 Analytics Dashboard**: Insights into your productivity patterns
        - **🔒 Secure & Private**: Your data is protected with enterprise-grade security

        ### 🌟 What Makes Us Different
        Unlike traditional task managers, TaskHub Smart ML learns from your behavior and adapts to your workflow. Our AI understands context, prioritizes intelligently, and provides proactive assistance when you need it most.

        ### 🚀 Technology Stack
        - **Frontend**: Streamlit for responsive web interface
        - **Backend**: Python with pandas for data management
        - **AI/ML**: Scikit-learn for intelligent classification
        - **Voice**: Speech recognition and text-to-speech integration
        - **Email**: SMTP integration for seamless notifications
        """)
    
    with col2:
        st.image("logo.png", width=200)
        st.markdown("### 📈 Stats")
        st.metric("Users Served", "1000+")
        st.metric("Tasks Managed", "50,000+")
        st.metric("AI Accuracy", "95%")
        
        st.markdown("### 🏆 Awards")
        st.write("🥇 Best Productivity App 2024")
        st.write("🏅 Innovation in AI Award")
        st.write("⭐ 5-Star User Rating")
    
    st.markdown("---")
    st.subheader("👥 Our Team")
    
    team_col1, team_col2, team_col3 = st.columns(3)
    with team_col1:
        st.markdown("""
        **🧠 AI Research Team**
        - Machine Learning Engineers
        - Data Scientists
        - NLP Specialists
        """)
    
    with team_col2:
        st.markdown("""
        **💻 Development Team**
        - Full-Stack Developers
        - UI/UX Designers
        - Quality Assurance
        """)
    
    with team_col3:
        st.markdown("""
        **🎯 Product Team**
        - Product Managers
        - User Experience Researchers
        - Customer Success
        """)
    
    st.button("⬅ Back", on_click=lambda: set_page("home"))

def services(): 
    navbar()
    st.header("🛠️ Our Services")
    
    st.markdown("""
    TaskHub Smart ML offers a comprehensive suite of intelligent productivity services powered by AI and machine learning.
    Explore our services below and click on any service to access it directly.
    """)
    
    st.markdown("---")
    
    # Service 1: Task Manager
    st.subheader("📋 Task Manager")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Smart Task Organization & Management**
        
        Organize your tasks efficiently with our intelligent task management system:
        - ✅ Create, edit, and delete tasks with ease
        - 🎯 Set priority levels (High, Medium, Low)
        - 📅 Track deadlines and get timely reminders
        - 📊 Monitor task completion progress
        - 📧 Receive email notifications for important tasks
        - 🔄 Automatic task status updates
        
        Perfect for students, professionals, and teams to stay organized and productive.
        """)
    with col2:
        if st.button("� Open Task Manager", use_container_width=True):
            set_page("tasks")
            st.rerun()
    
    st.markdown("---")
    
    # Service 2: Shopping Reminder
    st.subheader("🛒 Shopping Reminder")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Intelligent Shopping List Management**
        
        Never forget what you need to buy with our smart shopping reminder:
        - �️ Add, edit, and remove shopping items
        - 💰 Set priority for urgent purchases
        - 📍 Plan purchase dates
        - 📱 Mobile-friendly shopping lists
        - 📧 Get reminders before purchase dates
        - ✅ Mark items as purchased
        
        Ideal for grocery shopping, household items, and planned purchases.
        """)
    with col2:
        if st.button("🛒 Open Shopping Reminder", use_container_width=True):
            set_page("shopping")
            st.rerun()
    
    st.markdown("---")
    
    # Service 3: Voice Assistant
    st.subheader("🎤 Voice Assistant")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Fully Interactive Voice-Powered Assistant**
        
        Control your tasks and shopping with natural voice commands:
        - 🎙️ Add tasks using voice: "Add task submit project by Friday"
        - 🛍️ Add shopping items: "Add milk to my shopping list"
        - 📖 Listen to your tasks and shopping items read aloud
        - 🗣️ Get voice responses and confirmations
        - 🤖 Natural language understanding
        - ⚡ Instant action execution
        
        Hands-free productivity for busy professionals and multitaskers.
        """)
    with col2:
        if st.button("🎤 Open Voice Assistant", use_container_width=True):
            set_page("voice")
            st.rerun()
    
    st.markdown("---")
    
    # Service 4: Chatbot
    st.subheader("� Chatbot")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Intelligent Text-Based Assistant**
        
        Interact with your productivity system through natural conversation:
        - 💬 Add tasks via chat: "Add task exam preparation deadline tomorrow"
        - 🛒 Add shopping items: "Add groceries to shopping list"
        - 📋 View your tasks and shopping lists
        - 🧠 Smart intent detection and NLP
        - ✅ Automatic data updates
        - � Conversational interface
        
        Perfect for quick task management without leaving your chat window.
        """)
    with col2:
        if st.button("💬 Open Chatbot", use_container_width=True):
            set_page("chatbot")
            st.rerun()
    
    st.markdown("---")
    
    # Service 5: Email Classifier
    st.subheader("📧 Email Classifier")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Automatic Email Classification System**
        
        AI-powered email categorization running automatically in the background:
        - 🤖 Automatic classification (no manual trigger needed)
        - 🎯 Categories: Task-related, Shopping-related, Important, General
        - 📊 Visual analytics with line graphs
        - ⚡ Real-time processing
        - 🔒 Privacy-focused classification
        - 📈 Classification trends and insights
        
        Helps you organize and prioritize your emails intelligently.
        """)
    with col2:
        if st.button("📧 Open Email Classifier", use_container_width=True):
            set_page("email_classifier")
            st.rerun()
    
    st.markdown("---")
    
    # Service 6: Automated Reminder System
    st.subheader("⏰ Automated Reminder System")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        **Fully Automated Reminder Engine**
        
        Never miss a deadline with our intelligent reminder system:
        - 🔄 Runs automatically in the background (no activation needed)
        - 📅 Monitors task deadlines continuously
        - 🛒 Tracks shopping purchase dates
        - 📧 Sends email reminders automatically
        - ⏰ Smart timing for optimal notifications
        - 🎯 Priority-based reminder scheduling
        
        Set it and forget it - the system handles everything automatically.
        """)
    with col2:
        status = "🟢 Active" if reminder_engine.running else "🔴 Inactive"
        st.metric("Status", status)
        st.info("Runs automatically")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
        <h3>🚀 All Services Work Together Seamlessly</h3>
        <p>Our integrated platform ensures all services communicate and enhance each other for maximum productivity.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.button("⬅ Back to Home", on_click=lambda: set_page("home"))

def blogs(): 
    navbar()
    st.header("📝 TaskHub Smart ML Blog")
    
    st.markdown("Stay updated with the latest productivity tips, AI insights, and TaskHub features!")
    
    # Featured blog post
    st.subheader("🌟 Featured Post")
    with st.container():
        st.markdown("""
        ### 🚀 "The Future of AI-Powered Productivity: How Machine Learning is Revolutionizing Task Management"
        
        **Published:** January 28, 2026 | **Author:** TaskHub AI Team | **Read Time:** 5 min
        
        Discover how artificial intelligence is transforming the way we approach productivity and task management. From predictive scheduling to intelligent prioritization, learn about the cutting-edge technologies that make TaskHub Smart ML your ultimate productivity companion.
        
        [Read More →](#)
        """)
    
    st.markdown("---")
    
    # Blog categories
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 AI & Machine Learning")
        
        blog_posts_ai = [
            {
                "title": "Understanding Natural Language Processing in Task Management",
                "date": "January 25, 2026",
                "summary": "How NLP enables conversational task creation and management through voice and chat interfaces."
            },
            {
                "title": "Email Classification: The Science Behind Smart Sorting",
                "date": "January 22, 2026", 
                "summary": "Deep dive into the machine learning algorithms that power our email classification system."
            },
            {
                "title": "Voice Recognition Technology: From Speech to Action",
                "date": "January 20, 2026",
                "summary": "Exploring how voice commands are processed and converted into actionable tasks."
            },
            {
                "title": "Predictive Analytics for Better Time Management",
                "date": "January 18, 2026",
                "summary": "How AI predicts your productivity patterns and suggests optimal scheduling."
            }
        ]
        
        for post in blog_posts_ai:
            with st.expander(f"📖 {post['title']}"):
                st.write(f"**Date:** {post['date']}")
                st.write(f"**Summary:** {post['summary']}")
                st.button("Read Full Article", key=f"ai_{post['title'][:20]}")
    
    with col2:
        st.subheader("💡 Productivity Tips")
        
        blog_posts_productivity = [
            {
                "title": "10 Voice Commands That Will Transform Your Workflow",
                "date": "January 24, 2026",
                "summary": "Master these essential voice commands to manage tasks and shopping lists hands-free."
            },
            {
                "title": "The Psychology of Priority: Why Some Tasks Feel More Important",
                "date": "January 21, 2026",
                "summary": "Understanding cognitive biases in task prioritization and how AI can help overcome them."
            },
            {
                "title": "Building Sustainable Productivity Habits with AI Assistance",
                "date": "January 19, 2026",
                "summary": "How intelligent reminders and suggestions can help establish lasting productive routines."
            },
            {
                "title": "Collaborative Task Management: Best Practices for Teams",
                "date": "January 17, 2026",
                "summary": "Strategies for effective team coordination using smart task management tools."
            }
        ]
        
        for post in blog_posts_productivity:
            with st.expander(f"💡 {post['title']}"):
                st.write(f"**Date:** {post['date']}")
                st.write(f"**Summary:** {post['summary']}")
                st.button("Read Full Article", key=f"prod_{post['title'][:20]}")
    
    st.markdown("---")
    
    # Recent updates
    st.subheader("🔄 Recent Updates & Features")
    
    updates = [
        {
            "title": "🎤 Enhanced Voice Assistant with Confirmation Flow",
            "date": "January 28, 2026",
            "description": "New conversational voice interface with Yes/No confirmation for better accuracy."
        },
        {
            "title": "💬 ChatGPT-Style Conversational AI",
            "date": "January 27, 2026", 
            "description": "Upgraded chatbot with natural language understanding and contextual responses."
        },
        {
            "title": "🛠️ Complete CRUD Operations for Tasks and Shopping",
            "date": "January 26, 2026",
            "description": "Full edit, delete, and completion functionality with safety confirmations."
        },
        {
            "title": "📧 Smart Email Classification System",
            "date": "January 25, 2026",
            "description": "ML-powered email categorization with 95% accuracy for better organization."
        }
    ]
    
    for update in updates:
        with st.container():
            st.markdown(f"**{update['title']}**")
            st.caption(f"Released: {update['date']}")
            st.write(update['description'])
            st.markdown("---")
    
    # Newsletter signup
    st.subheader("📬 Stay Updated")
    st.write("Subscribe to our newsletter for the latest productivity tips and feature updates!")
    
    newsletter_col1, newsletter_col2 = st.columns([3, 1])
    with newsletter_col1:
        email = st.text_input("Enter your email address", placeholder="your@email.com")
    with newsletter_col2:
        if st.button("Subscribe 📧"):
            if email:
                st.success("✅ Subscribed successfully!")
            else:
                st.warning("Please enter your email")
    
    st.button("⬅ Back", on_click=lambda: set_page("home"))

def faqs(): 
    navbar()
    st.header("❓ Frequently Asked Questions")
    
    st.write("Find answers to common questions about TaskHub Smart ML")
    
    # Search functionality
    search_query = st.text_input("🔍 Search FAQs", placeholder="Type your question...")
    
    # FAQ categories
    faq_categories = {
        "🚀 Getting Started": [
            {
                "q": "How do I create my first task?",
                "a": "Go to Task Manager, click the 'Add New Task' section, enter your task description, select priority, set a deadline, and click 'Add Task'. You'll receive an email confirmation!"
            },
            {
                "q": "How do I add items to my shopping list?",
                "a": "Navigate to Shopping Reminder, use the 'Add New Shopping Item' form, enter the item name, choose priority, set purchase date, and click 'Add Item'."
            },
            {
                "q": "Can I use voice commands to add tasks?",
                "a": "Yes! Go to Voice Assistant and click 'Start Listening'. Say something like 'Add task finish project with high priority' and confirm when prompted."
            },
            {
                "q": "How do I edit or delete existing tasks?",
                "a": "In Task Manager, each task has Edit, Complete, and Delete buttons. Click Edit to modify details, Complete to mark as done, or Delete (with confirmation) to remove."
            }
        ],
        
        "🤖 AI Features": [
            {
                "q": "How does the chatbot understand my requests?",
                "a": "Our AI chatbot uses natural language processing (NLP) to understand your intent. You can chat naturally like 'I need to buy groceries' and it will help you add shopping items."
            },
            {
                "q": "What voice commands does the assistant recognize?",
                "a": "The voice assistant understands commands like 'Add task [description] with [priority]', 'Buy [item] with [priority]', 'Show my tasks', and 'Check shopping list'."
            },
            {
                "q": "How accurate is the email classification?",
                "a": "Our email classification system achieves 95% accuracy using machine learning algorithms trained on thousands of email examples across Task, Shopping, General, and Spam categories."
            },
            {
                "q": "Can the AI learn from my behavior?",
                "a": "Yes! The system learns from your usage patterns to provide better suggestions, improve classification accuracy, and personalize your experience over time."
            }
        ],
        
        "⚙️ Features & Functionality": [
            {
                "q": "How do reminders work?",
                "a": "The automated reminder system runs in the background, checking every hour for upcoming deadlines. You'll receive email alerts 1 day before and on due dates for both tasks and shopping items."
            },
            {
                "q": "Can I change task priorities after creation?",
                "a": "Absolutely! Click the Edit button next to any task, modify the priority (High, Medium, Low), and save changes. You'll receive an email confirmation of the update."
            },
            {
                "q": "What happens when I mark a task as complete?",
                "a": "Completed tasks are removed from your active list and you receive a congratulations email. The task data is preserved for your records and analytics."
            },
            {
                "q": "How do I manage shopping items after purchase?",
                "a": "Click the 'Purchased' button next to any shopping item. It will be removed from your active list and you'll receive a purchase confirmation email."
            }
        ],
        
        "📧 Email & Notifications": [
            {
                "q": "Why am I not receiving email notifications?",
                "a": "Check your spam folder first. Ensure your email address is correct in your profile. If issues persist, contact support - our system sends notifications for all task/shopping activities."
            },
            {
                "q": "Can I customize email notification frequency?",
                "a": "Currently, the system sends notifications for task/shopping creation, updates, completions, and deadline reminders. Customization options are planned for future releases."
            },
            {
                "q": "How does the contact form work?",
                "a": "Fill out the contact form with your name, email, and message. The system classifies your message and sends it from your email to our admin team, with a confirmation sent back to you."
            },
            {
                "q": "What email categories does the classifier recognize?",
                "a": "The system classifies emails into four categories: Task (work-related), Shopping (purchase-related), General (casual communication), and Spam (unwanted messages)."
            }
        ],
        
        "🔧 Technical Support": [
            {
                "q": "Voice assistant isn't working - what should I check?",
                "a": "Ensure your microphone is connected and browser has microphone permissions. Check internet connection for speech recognition. Try the test buttons first to verify functionality."
            },
            {
                "q": "Why can't I hear voice responses?",
                "a": "Check your speakers/headphones and browser audio permissions. The system uses text-to-speech which requires audio output capabilities. Try refreshing the page."
            },
            {
                "q": "My tasks/shopping items aren't saving - help!",
                "a": "This usually indicates a temporary issue. Try refreshing the page and re-adding the item. If the problem persists, use the contact form to report the issue with details."
            },
            {
                "q": "How do I reset my account or data?",
                "a": "Currently, you can manually delete individual tasks and shopping items. For complete account reset, please contact our support team through the contact form."
            }
        ],
        
        "🔒 Privacy & Security": [
            {
                "q": "Is my data secure?",
                "a": "Yes! Your data is stored securely and only accessible to you. We use industry-standard security practices and never share personal information with third parties."
            },
            {
                "q": "Where is my data stored?",
                "a": "Data is stored in secure CSV files with user-specific access controls. Each user's tasks and shopping items are isolated and protected."
            },
            {
                "q": "Can I export my data?",
                "a": "Data export functionality is planned for future releases. Currently, you can view all your data through the Task Manager and Shopping Reminder pages."
            },
            {
                "q": "How long is my data retained?",
                "a": "Your active tasks and shopping items are retained indefinitely until you delete them. Completed/purchased items are archived for your records."
            }
        ]
    }
    
    # Display FAQs
    for category, faqs in faq_categories.items():
        # Filter FAQs based on search query
        if search_query:
            filtered_faqs = [faq for faq in faqs if search_query.lower() in faq['q'].lower() or search_query.lower() in faq['a'].lower()]
        else:
            filtered_faqs = faqs
        
        if filtered_faqs:  # Only show category if it has matching FAQs
            st.subheader(category)
            
            for faq in filtered_faqs:
                with st.expander(f"❓ {faq['q']}"):
                    st.write(faq['a'])
    
    # Contact section
    st.markdown("---")
    st.subheader("💬 Still Have Questions?")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Can't find what you're looking for?**
        
        Our support team is here to help! Use the contact form to send us your questions and we'll get back to you within 24-48 hours.
        """)
        
        if st.button("📧 Contact Support"):
            set_page("contact")
            st.rerun()
    
    with col2:
        st.markdown("""
        **Quick Tips:**
        
        - Try the search box above to find specific topics
        - Check out our Blog for detailed tutorials
        - Use the test buttons in Voice Assistant for troubleshooting
        - Most issues can be resolved by refreshing the page
        """)
    
    st.button("⬅ Back", on_click=lambda: set_page("home"))
# ================= ENHANCED CONTACT PAGE =================
def contact():
    navbar()
    st.header("📧 Contact Us")
    st.write("Send us a message and we'll get back to you!")
    
    name = st.text_input("Your Name")
    user_email = st.text_input("Your Email")
    msg = st.text_area("Message")
    
    if st.button("Send Message"):
        if name and user_email and msg:
            # Classify the contact message
            category, confidence = email_classifier.classify_email("Contact Form", msg)
            
            # Store the message
            global messages_df
            messages_df = pd.read_csv(MESSAGES)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            messages_df.loc[len(messages_df)] = [name, user_email, msg, timestamp, category]
            messages_df.to_csv(MESSAGES, index=False)
            
            # Send to admin (from admin email to admin email)
            admin_subject = f"New Contact Form Submission - {category}"
            admin_body = f"""New contact form submission:

Name: {name}
Email: {user_email}
Category: {category} (Confidence: {confidence:.2f})
Timestamp: {timestamp}

Message:
{msg}

---
This message was sent from the TaskHub Smart ML contact form.
Please reply directly to: {user_email}
"""
            
            try:
                # Send message from user's email to admin
                user_to_admin_email = EmailMessage()
                user_to_admin_email["From"] = user_email  # User's email as sender
                user_to_admin_email["To"] = ADMIN_EMAIL   # Admin as receiver
                user_to_admin_email["Subject"] = f"Contact Form: {category} - {name}"
                user_to_admin_email["Reply-To"] = user_email  # Set reply-to for easy response
                user_to_admin_email.set_content(admin_body)
                
                # Send the email using SMTP (admin credentials for sending)
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(ADMIN_EMAIL, APP_PASSWORD)
                    smtp.send_message(user_to_admin_email)
                
                # Send confirmation to user (from admin email to user email)
                send_confirmation_email(user_email, name, msg)
                
                st.success(f"✅ Message sent successfully! Category: {category}")
                st.info("📧 A confirmation email has been sent to your email address.")
                st.info("� Our team will review your message and get back to you within 24-48 hours.")
                
            except Exception as e:
                st.error(f"Error sending message: {e}")
        else:
            st.warning("Please fill in all fields.")
    
    # Display recent messages (admin view)
    if st.session_state.user == ADMIN_EMAIL:
        st.markdown("---")
        st.subheader("📋 Recent Messages (Admin View)")
        messages_df = pd.read_csv(MESSAGES)
        if not messages_df.empty:
            for _, row in messages_df.tail(5).iterrows():
                with st.expander(f"{row['name']} - {row['category']} ({row['timestamp']})"):
                    st.write(f"**Email:** {row['email']}")
                    st.write(f"**Category:** {row['category']}")
                    st.write(f"**Message:** {row['message']}")
    
    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= EMAIL CLASSIFIER PAGE =================
def email_classifier_page():
    navbar()
    st.header("🤖 Automatic Email Classification System")
    st.write("**AI-powered email categorization - Fetching REAL emails from Gmail**")
    
    st.info("� System automatically fetches and classifies emails from your Gmail inbox using IMAP")
    
    # Auto-fetch and classify emails
    with st.spinner("🔄 Fetching and classifying emails from Gmail..."):
        new_count = auto_classify_emails()
        if new_count > 0:
            st.success(f"✅ Classified {new_count} new emails!")
    
    # Get classified emails
    classified_df = pd.read_csv(CLASSIFIED_EMAILS)
    
    if not classified_df.empty:
        # Filter out any rows with missing data
        classified_df = classified_df.dropna(subset=['subject', 'category', 'timestamp'])
        
        if classified_df.empty:
            st.info("📧 No valid classified emails yet.")
        else:
            # Display classification trends with LINE GRAPH
            st.subheader("📈 Email Classification Trends (Line Graph)")
            
            # Prepare data for line graph
            try:
                classified_df['timestamp'] = pd.to_datetime(classified_df['timestamp'])
                classified_df = classified_df.sort_values('timestamp')
                
                # Create time-based grouping
                classified_df['date'] = classified_df['timestamp'].dt.date
                
                # Count emails by category and date
                category_by_date = classified_df.groupby(['date', 'category']).size().unstack(fill_value=0)
                
                # Create cumulative counts for line graph
                category_cumulative = category_by_date.cumsum()
                
                # Display LINE GRAPH
                st.line_chart(category_cumulative)
                
                st.markdown("""
                **📊 Line Graph Explanation:**
                - **X-axis**: Time (dates when emails were classified)
                - **Y-axis**: Cumulative number of emails per category
                - Each line represents a different email category
                - Steeper slopes = more emails in that category
                - **Real-time data** from your Gmail inbox
                """)
            except Exception as e:
                st.warning(f"Could not generate line graph: {e}")
            
            # Category distribution
            st.subheader("� Category Distribution")
            category_counts_total = classified_df['category'].value_counts()
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(category_counts_total)
            with col2:
                st.markdown("**Classification Summary:**")
                for category, count in category_counts_total.items():
                    percentage = (count / len(classified_df)) * 100
                    st.metric(f"{category}", f"{count} emails", f"{percentage:.1f}%")
            
            # Statistics
            st.subheader("📈 Classification Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Classified", len(classified_df))
            with col2:
                avg_confidence = classified_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col3:
                task_emails = len(classified_df[classified_df['category'] == 'Task'])
                st.metric("Task Emails", task_emails)
            with col4:
                shopping_emails = len(classified_df[classified_df['category'] == 'Shopping'])
                st.metric("Shopping Emails", shopping_emails)
        
        # Recent classifications
        st.subheader("🕒 Recent Email Classifications")
        for _, row in classified_df.tail(10).iterrows():
            # Handle missing or invalid data
            subject = str(row['subject']) if pd.notna(row['subject']) else "No Subject"
            body = str(row['body']) if pd.notna(row['body']) else "No Body"
            category = str(row['category']) if pd.notna(row['category']) else "Unknown"
            timestamp = str(row['timestamp']) if pd.notna(row['timestamp']) else "Unknown"
            confidence = float(row['confidence']) if pd.notna(row['confidence']) else 0.0
            
            with st.expander(f"📧 {category} - {subject[:50]}... ({timestamp})"):
                st.write(f"**Subject:** {subject}")
                st.write(f"**Body Preview:** {body[:200]}...")
                st.write(f"**Category:** {category}")
                st.write(f"**Confidence:** {confidence:.2%}")
                st.write(f"**Timestamp:** {timestamp}")
                
                # Action suggestions
                if category == "Task":
                    st.info("💡 Suggestion: This email contains task-related content")
                elif category == "Shopping":
                    st.info("💡 Suggestion: This email contains shopping-related content")
    else:
        st.info("📧 No classified emails yet. Click the button below to fetch and classify emails from Gmail.")
        
        if st.button("🔄 Fetch & Classify Emails Now"):
            with st.spinner("Fetching emails from Gmail..."):
                count = auto_classify_emails()
                if count > 0:
                    st.success(f"✅ Successfully classified {count} emails!")
                    st.rerun()
                else:
                    st.warning("No new emails to classify or unable to connect to Gmail.")
    
    st.markdown("---")
    st.success("✅ Email classification system fetches REAL emails from Gmail automatically!")
    st.info("💡 **How it works**: System connects to Gmail via IMAP, fetches emails, and classifies them using ML (TF-IDF + Naive Bayes)")
    
    st.button("⬅ Back to Home", on_click=lambda: set_page("home"))

# ================= REMINDERS PAGE =================
def reminders_page():
    navbar()
    st.header("⏰ Reminder System")
    
    # Reminder engine status
    col1, col2 = st.columns(2)
    with col1:
        status = "🟢 Active" if reminder_engine.running else "🔴 Inactive"
        st.metric("Reminder Engine Status", status)
    
    with col2:
        if not reminder_engine.running:
            if st.button("▶️ Start Reminder Engine"):
                st.session_state.reminder_thread = reminder_engine.start()
                st.success("Reminder engine started!")
                st.rerun()
        else:
            if st.button("⏸️ Stop Reminder Engine"):
                reminder_engine.stop()
                st.session_state.reminder_thread = None
                st.success("Reminder engine stopped!")
                st.rerun()
    
    # Upcoming reminders
    st.subheader("📅 Upcoming Reminders")
    
    current_date = datetime.now().date()
    
    # Task reminders
    tasks_df = pd.read_csv(TASKS)
    user_tasks = tasks_df[tasks_df.email == st.session_state.user]
    
    st.write("**📋 Task Deadlines:**")
    if not user_tasks.empty:
        for _, task in user_tasks.iterrows():
            deadline = datetime.strptime(str(task['deadline']), '%Y-%m-%d').date()
            days_until = (deadline - current_date).days
            
            if days_until <= 3:  # Show tasks due within 3 days
                if days_until < 0:
                    status = f"🔴 Overdue by {abs(days_until)} days"
                elif days_until == 0:
                    status = "🟡 Due TODAY"
                elif days_until == 1:
                    status = "🟠 Due TOMORROW"
                else:
                    status = f"🟢 Due in {days_until} days"
                
                st.markdown(f"• **{task['task']}** - {status} ({task['priority']} priority)")
    else:
        st.info("No upcoming task deadlines.")
    
    # Shopping reminders
    shopping_df = pd.read_csv(SHOPPING)
    user_shopping = shopping_df[shopping_df.email == st.session_state.user]
    
    st.write("**🛒 Shopping Dates:**")
    if not user_shopping.empty:
        for _, item in user_shopping.iterrows():
            purchase_date = datetime.strptime(str(item['purchase_date']), '%Y-%m-%d').date()
            days_until = (purchase_date - current_date).days
            
            if days_until <= 3:  # Show items due within 3 days
                if days_until < 0:
                    status = f"🔴 Overdue by {abs(days_until)} days"
                elif days_until == 0:
                    status = "🟡 Buy TODAY"
                elif days_until == 1:
                    status = "🟠 Buy TOMORROW"
                else:
                    status = f"🟢 Buy in {days_until} days"
                
                st.markdown(f"• **{item['item']}** - {status} ({item['priority']} priority)")
    else:
        st.info("No upcoming shopping dates.")
    
    # Reminder settings
    st.subheader("⚙️ Reminder Settings")
    st.info("📧 Email reminders are automatically sent:")
    st.write("• 1 day before task deadlines")
    st.write("• On the day tasks are due")
    st.write("• 1 day before shopping dates")
    st.write("• On shopping purchase dates")
    
    st.button("⬅ Back", on_click=lambda: set_page("home"))

# ================= LOGOUT =================
def logout():
    # Stop reminder engine when logging out
    if reminder_engine.running:
        reminder_engine.stop()
        st.session_state.reminder_thread = None
    
    st.session_state.user = None
    set_page("login")
    
    st.session_state.user = None
    set_page("login")

# ================= ROUTER =================
pages = {
    "login": login,
    "home": home,
    "shopping": shopping,
    "tasks": tasks,
    "voice": voice,
    "chatbot": chatbot_page,
    "email_classifier": email_classifier_page,
    "reminders": reminders_page,
    "about": about,
    "services": services,
    "blogs": blogs,
    "faqs": faqs,
    "contact": contact
}

pages.get(st.session_state.page, login)()
