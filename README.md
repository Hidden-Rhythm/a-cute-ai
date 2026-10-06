<div align="center">

# ♡ Hidden AI

### a cute little AI with a personality.

**A lightweight, personality-driven AI assistant powered by Google Gemini.**

<br>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)](#)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge\&logo=flask\&logoColor=white)](#)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-AI-4285F4?style=for-the-badge\&logo=google\&logoColor=white)](#)
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?style=for-the-badge\&logo=vercel\&logoColor=white)](#)

<br>

<a href="https://a-cute-ai-by-hidden.vercel.app">
  <img src="https://img.shields.io/badge/✦_TRY_HIDDEN_AI-111111?style=for-the-badge" alt="Try Hidden AI">
</a>

 

<a href="https://github.com/Hidden-Rhythm/a-cute-ai">
  <img src="https://img.shields.io/badge/VIEW_SOURCE-111111?style=for-the-badge&logo=github&logoColor=white" alt="View Source">
</a>

</div>

---

## ✦ What is Hidden AI?

**Hidden AI** is a small experimental AI assistant built around a simple idea:

> **AI doesn't have to feel robotic.**

It combines Google Gemini with a playful personality, conversation memory, live information sources, file attachments, image generation, and a lightweight chat interface.

The goal is to make an AI assistant that feels a little more **human, casual, and fun** instead of another generic chatbot.

---

## ✦ Features

| Feature                 | Description                                                 |
| ----------------------- | ----------------------------------------------------------- |
| 💬 **Natural Chat**     | Personality-driven AI conversations                         |
| 🧠 **Memory**           | Stores useful facts and context from conversations          |
| 🌐 **Live Information** | Weather, news, time, markets, GitHub and other live sources |
| 📎 **File Attachments** | Upload files and images directly in chat                    |
| 🎨 **Image Generation** | Generate images using `/img <prompt>`                       |
| 💾 **Chat History**     | Conversations are stored locally in the browser             |
| ⚡ **Model Fallback**    | Automatically tries alternative Gemini models when needed   |
| 📝 **Markdown**         | Rich formatting for AI responses                            |
| 📱 **Responsive UI**    | Designed for desktop and mobile                             |
| 🐱 **Cute Personality** | Casual responses, reactions and a little chaos              |

---

## ✦ How It Works

```text
                 ┌─────────────────┐
                 │      User       │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Chat UI       │
                 │ HTML / JS / CSS │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   Flask API     │
                 │   /api/chat     │
                 └────────┬────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
          Memory      Live Data    Attachments
              │           │           │
              └───────────┼───────────┘
                          ▼
                 ┌─────────────────┐
                 │ Google Gemini   │
                 │ Model Fallback  │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │   AI Response   │
                 └─────────────────┘
```

---

## ✦ Live Information

Hidden AI can gather information from multiple external sources depending on what you're asking.

Examples include:

* 🌦️ Weather
* 🌫️ Air quality
* 🕐 Local time
* 🌅 Sunrise & sunset
* 📰 News feeds
* 📚 Wikipedia
* 🔎 DuckDuckGo
* 💰 Cryptocurrency
* 📈 Stocks
* 💱 Currency exchange
* 🌍 World Bank data
* 🐙 GitHub information
* 🦊 GitLab information
* and other live sources implemented in the backend

This allows the AI to answer questions using **fresh external information** instead of relying entirely on model knowledge.

---

## ✦ Memory

Hidden AI has a lightweight client-side memory system.

During conversations, the application can extract useful facts and reuse relevant ones later.

```text
User message
     │
     ▼
Fact extraction
     │
     ▼
Local memory
     │
     ▼
Relevant memories
     │
     ▼
Gemini context
```

Memory is selectively provided to the model rather than dumping the entire memory store into every request.

---

## ✦ Image Generation

You can generate images directly from the chat interface.

```text
/img a cute cat sitting under the moon
```

The frontend uses the image-generation interface and displays the generated result directly inside the conversation.

Generated images can also be downloaded from the chat.

---

## ✦ File Attachments

The chat supports attaching files and images.

The browser converts supported attachments into payloads which are sent to the Flask API and passed to Gemini when supported.

```text
File
 ↓
Browser
 ↓
Base64 payload
 ↓
Flask
 ↓
Gemini
 ↓
AI analysis
```

Large files are rejected to keep requests lightweight.

---

## ✦ Model Fallback

Hidden AI doesn't depend on a single model attempt.

The backend maintains a list of Gemini models and tries alternatives when a model fails, returns an empty response, or is unavailable.

```text
Gemini Model
     │
     ├── success ──────► response
     │
     └── failure
            │
            ▼
       next model
            │
            ▼
       next model
            │
            ▼
         response
```

This makes the assistant more resilient to temporary model availability issues.

---

## ✦ Chat Experience

The interface includes:

* conversation sidebar
* multiple chats
* local chat persistence
* message history
* attachment previews
* markdown rendering
* code blocks
* generated-image previews
* image downloads
* thinking indicators
* mobile-friendly layout
* custom confirmation dialogs
* cute error states

The UI is intentionally lightweight rather than being built around a large frontend framework.

---

## ✦ Project Structure

```text
a-cute-ai-by-hidden/
│
├── api/
│   ├── index.py
│   ├── cat.jpg
│   └── favicon.ico
│
├── requirements.txt
└── README.md
```

### `api/index.py`

The main application containing:

* Flask server
* Gemini integration
* AI personality
* conversation handling
* memory processing
* live-data providers
* attachment processing
* image-generation integration
* chat API
* health endpoint
* Vercel serverless handler

### `api/cat.jpg`

Local image asset used by the interface.

### `api/favicon.ico`

Application favicon.

### `requirements.txt`

Python dependencies required by the backend.

---

## ✦ API Routes

| Route          | Method | Purpose                        |
| -------------- | ------ | ------------------------------ |
| `/`            | `GET`  | Serves the Hidden AI interface |
| `/api/chat`    | `POST` | Processes chat messages        |
| `/api/health`  | `GET`  | Basic health check             |
| `/cat.jpg`     | `GET`  | Serves the cat image           |
| `/favicon.ico` | `GET`  | Serves the favicon             |

---

## ✦ Tech Stack

| Technology        | Role                  |
| ----------------- | --------------------- |
| **Python**        | Backend language      |
| **Flask**         | Web server / API      |
| **Google Gemini** | AI engine             |
| **JavaScript**    | Chat interface        |
| **HTML / CSS**    | UI                    |
| **Requests**      | External API requests |
| **Vercel**        | Deployment            |

---

## ✦ Run Locally

Clone the repository:

```bash
git clone https://github.com/Hidden-Rhythm/a-cute-ai.git
cd a-cute-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Gemini API key as an environment variable and run the Flask application.

> **Important:** Never commit your Gemini API key directly into the repository.

---

## ✦ Security Note

The backend requires a Google Gemini API key.

For production deployments, keep the key in an environment variable rather than hardcoding it inside `api/index.py`.

If a key has ever been exposed publicly, **rotate it immediately**.

---

## ✦ Live

### ♡ Try Hidden AI

<a href="https://a-cute-ai-by-hidden.vercel.app">

**a-cute-ai-by-hidden.vercel.app**

</a>

---

## ✦ Source

<a href="https://github.com/Hidden-Rhythm/a-cute-ai">

**github.com/Hidden-Rhythm/a-cute-ai-by-hidden**

</a>

---

<div align="center">

### made with curiosity ♡

**Hidden AI · 2026**

<sub>small project. big personality.</sub>

</div>
