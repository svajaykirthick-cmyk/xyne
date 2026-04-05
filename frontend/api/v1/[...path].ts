const chats = new Map<string, { title: string | null; isBookmarked: boolean; messages: any[] }>()

type ChatSSEventName = "rm" | "u" | "e" | "er" | "cu"

function json(res: any, status: number, body: unknown) {
  res.status(status).setHeader("Content-Type", "application/json")
  res.end(JSON.stringify(body))
}

function asArray(input: unknown): string[] {
  if (Array.isArray(input)) return input.map((v) => String(v))
  if (typeof input === "string") return [input]
  return []
}

function getPathParts(req: any): string[] {
  return asArray(req.query?.path)
}

function randomId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2, 11)}`
}

function parseMessage(input: unknown): string {
  const raw = typeof input === "string" ? input : ""
  if (!raw) return ""

  try {
    const parsed = JSON.parse(raw)
    if (typeof parsed === "string") return parsed
    if (Array.isArray(parsed)) {
      return parsed
        .map((part) => {
          if (typeof part?.value === "string") return part.value
          if (typeof part?.text === "string") return part.text
          return ""
        })
        .join("")
        .trim()
    }
  } catch {
    // Keep raw text when not JSON
  }

  return raw.trim()
}

function ensureChat(chatId: string) {
  if (!chats.has(chatId)) {
    chats.set(chatId, {
      title: null,
      isBookmarked: false,
      messages: [],
    })
  }
  return chats.get(chatId)!
}

function sseWrite(res: any, event: ChatSSEventName, data: string) {
  res.write(`event: ${event}\n`)
  res.write(`data: ${data}\n\n`)
}

async function streamOpenAI(message: string, res: any) {
  const apiKey = process.env.OPENAI_API_KEY
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini"
  const systemPrompt =
    process.env.DIRECT_SYSTEM_PROMPT ||
    "You are Xyne running in Vercel Direct Mode. Be concise and helpful."

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not set")
  }

  const upstream = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      stream: true,
      temperature: 0.2,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: message },
      ],
    }),
  })

  if (!upstream.ok || !upstream.body) {
    const errText = await upstream.text().catch(() => "")
    throw new Error(`OpenAI request failed (${upstream.status}): ${errText}`)
  }

  const reader = upstream.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let fullText = ""

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const lines = buffer.split("\n")
    buffer = lines.pop() || ""

    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed.startsWith("data:")) continue

      const payload = trimmed.slice(5).trim()
      if (payload === "[DONE]") {
        return fullText
      }

      let parsed: any
      try {
        parsed = JSON.parse(payload)
      } catch {
        continue
      }

      const delta = parsed?.choices?.[0]?.delta?.content
      if (typeof delta === "string" && delta.length > 0) {
        fullText += delta
        sseWrite(res, "u", delta)
      }
    }
  }

  return fullText
}

async function handleMessageCreate(req: any, res: any) {
  const query = req.query || {}
  const message = parseMessage(query.message)

  if (!message) {
    return json(res, 400, { error: "Missing message" })
  }

  const chatId = typeof query.chatId === "string" && query.chatId.length > 0 ? query.chatId : randomId()
  const userMessageId = randomId()
  const assistantMessageId = randomId()
  const chat = ensureChat(chatId)

  // Persist the user message for chat history.
  chat.messages.push({
    externalId: userMessageId,
    messageRole: "user",
    message,
    sources: [],
    citationMap: {},
    thinking: "",
    imageCitations: [],
    attachments: [],
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  })

  res.statusCode = 200
  res.setHeader("Content-Type", "text/event-stream")
  res.setHeader("Cache-Control", "no-cache, no-transform")
  res.setHeader("Connection", "keep-alive")
  res.setHeader("X-Accel-Buffering", "no")

  sseWrite(
    res,
    "rm",
    JSON.stringify({
      chatId,
      messageId: assistantMessageId,
      timeTakenMs: 0,
    }),
  )

  const startedAt = Date.now()

  try {
    const assistantText = await streamOpenAI(message, res)

    chat.messages.push({
      externalId: assistantMessageId,
      messageRole: "assistant",
      message: assistantText,
      sources: [],
      citationMap: {},
      thinking: "",
      imageCitations: [],
      attachments: [],
      timeTakenMs: Date.now() - startedAt,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    })

    sseWrite(res, "cu", JSON.stringify({ sources: [], response: assistantText, citationMap: {} }))
    sseWrite(res, "e", "")
    res.end()
  } catch (error: any) {
    sseWrite(res, "er", error?.message || "Model call failed")
    sseWrite(res, "e", "")
    res.end()
  }
}

async function handleChatFetch(req: any, res: any) {
  const body = typeof req.body === "object" && req.body ? req.body : {}
  const chatId = typeof body.chatId === "string" ? body.chatId : ""

  if (!chatId) {
    return json(res, 200, { messages: [] })
  }

  const chat = ensureChat(chatId)
  return json(res, 200, {
    messages: chat.messages,
    chat: {
      title: chat.title,
      isBookmarked: chat.isBookmarked,
    },
  })
}

function handleHistory(req: any, res: any, favoritesOnly: boolean) {
  const allChats = [...chats.entries()].map(([chatId, data]) => ({
    externalId: chatId,
    title: data.title || "New chat",
    isBookmarked: data.isBookmarked,
    createdAt: data.messages[0]?.createdAt || new Date().toISOString(),
    updatedAt: data.messages[data.messages.length - 1]?.updatedAt || new Date().toISOString(),
  }))

  const filtered = favoritesOnly ? allChats.filter((c) => c.isBookmarked) : allChats
  json(res, 200, filtered)
}

function handleRename(req: any, res: any) {
  const body = typeof req.body === "object" && req.body ? req.body : {}
  const chatId = typeof body.chatId === "string" ? body.chatId : ""
  const title = typeof body.title === "string" ? body.title : ""

  if (!chatId) return json(res, 400, { error: "chatId is required" })

  const chat = ensureChat(chatId)
  chat.title = title || "New chat"
  return json(res, 200, { ok: true })
}

function handleBookmark(req: any, res: any) {
  const body = typeof req.body === "object" && req.body ? req.body : {}
  const chatId = typeof body.chatId === "string" ? body.chatId : ""
  const bookmark = Boolean(body.bookmark)

  if (!chatId) return json(res, 400, { error: "chatId is required" })

  const chat = ensureChat(chatId)
  chat.isBookmarked = bookmark
  return json(res, 200, { ok: true })
}

function handleDelete(req: any, res: any) {
  const body = typeof req.body === "object" && req.body ? req.body : {}
  const chatId = typeof body.chatId === "string" ? body.chatId : ""
  if (!chatId) return json(res, 400, { error: "chatId is required" })
  chats.delete(chatId)
  return json(res, 200, { ok: true })
}

export default async function handler(req: any, res: any) {
  const method = String(req.method || "GET").toUpperCase()
  const parts = getPathParts(req)
  const path = parts.join("/")

  // Lightweight compatibility endpoints for Vercel Direct Mode.
  if (method === "GET" && path === "me") {
    return json(res, 200, {
      user: {
        id: "vercel-direct-user",
        email: process.env.DIRECT_USER_EMAIL || "direct@xyne.app",
        name: process.env.DIRECT_USER_NAME || "Vercel User",
        photoLink: "",
        role: "SuperAdmin",
      },
      workspace: {
        id: "vercel-direct-workspace",
        domain: "vercel",
        name: "Vercel Direct Workspace",
      },
      agentWhiteList: false,
    })
  }

  if (method === "GET" && path === "config") {
    return json(res, 200, {
      agenticByDefault: false,
      isDemo: false,
    })
  }

  if (method === "POST" && path === "refresh-token") {
    return json(res, 200, { ok: true })
  }

  if (method === "GET" && path === "chat/models") {
    return json(res, 200, {
      models: [
        {
          actualName: process.env.OPENAI_MODEL || "gpt-4o-mini",
          labelName: process.env.DIRECT_MODEL_LABEL || "GPT-4o Mini",
          provider: "openai",
          reasoning: true,
          websearch: false,
          deepResearch: false,
          description: "Direct model call via Vercel serverless",
        },
      ],
    })
  }

  if (method === "POST" && path === "message/create") {
    return handleMessageCreate(req, res)
  }

  if (method === "POST" && path === "message/retry") {
    // Frontend uses retry endpoint with SSE semantics; reuse create behavior.
    const query = req.query || {}
    req.query = {
      ...query,
      message: typeof query.message === "string" ? query.message : "Please retry the previous response.",
    }
    return handleMessageCreate(req, res)
  }

  if (method === "POST" && path === "chat") {
    return handleChatFetch(req, res)
  }

  if (method === "GET" && path === "chat/history") {
    return handleHistory(req, res, false)
  }

  if (method === "GET" && path === "chat/favorites") {
    return handleHistory(req, res, true)
  }

  if (method === "POST" && path === "chat/rename") {
    return handleRename(req, res)
  }

  if (method === "POST" && path === "chat/bookmark") {
    return handleBookmark(req, res)
  }

  if (method === "POST" && path === "chat/delete") {
    return handleDelete(req, res)
  }

  if (method === "GET" && path === "agents") {
    return json(res, 200, [])
  }

  if (method === "POST" && path === "autocomplete") {
    return json(res, 200, { results: [] })
  }

  if (method === "POST" && path === "chat/stop") {
    return json(res, 200, { ok: true })
  }

  if (method === "GET" && path === "health") {
    return json(res, 200, { ok: true, mode: "vercel-direct" })
  }

  // Return empty success for non-critical endpoints to keep the UI running.
  return json(res, 200, {})
}
