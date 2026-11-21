// 模块：Session OpenAPI 描述实现
//
// 职责：
//   返回覆盖 P0 端点的 OpenAPI 3.0.3 JSON 文档。
//
// 边界：
//   静态文本；不随配置变化。

#include "Service/session/openapi_spec.h"

namespace mobilenet {
namespace service {
namespace {

// OpenAPI 文档；仅使用基础 schema，避免运行时反射。
constexpr const C8 kOpenApiSpec[] = R"json({
  "openapi": "3.0.3",
  "info": {
    "title": "MobileNet Session API",
    "description": "OpenAI-compatible chat and stateful session API for the MobileNet inference engine.",
    "version": "1.0.0"
  },
  "servers": [
    {"url": "http://127.0.0.1:8080"}
  ],
  "paths": {
    "/health": {
      "get": {
        "summary": "Service health",
        "responses": {
          "200": {"description": "Service is running"}
        }
      }
    },
    "/v1/models": {
      "get": {
        "summary": "List models",
        "responses": {
          "200": {"description": "OpenAI model list"}
        }
      }
    },
    "/v1/chat/completions": {
      "post": {
        "summary": "OpenAI-compatible chat completions",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {"$ref": "#/components/schemas/ChatRequest"}
            }
          }
        },
        "responses": {
          "200": {"description": "Chat completion or SSE stream"},
          "400": {"description": "Invalid request"},
          "404": {"description": "Model not found"},
          "503": {"description": "Engine unavailable"}
        }
      }
    },
    "/v1/sessions": {
      "post": {
        "summary": "Create a session",
        "requestBody": {
          "required": false,
          "content": {
            "application/json": {
              "schema": {"$ref": "#/components/schemas/CreateSessionRequest"}
            }
          }
        },
        "responses": {
          "200": {"description": "Session created"},
          "400": {"description": "Invalid request"}
        }
      }
    },
    "/v1/sessions/{session_id}": {
      "get": {
        "summary": "Get session information",
        "parameters": [
          {"name": "session_id", "in": "path", "required": true,
           "schema": {"type": "string"}}
        ],
        "responses": {
          "200": {"description": "Session information"},
          "404": {"description": "Session not found"}
        }
      },
      "delete": {
        "summary": "Delete a session",
        "parameters": [
          {"name": "session_id", "in": "path", "required": true,
           "schema": {"type": "string"}}
        ],
        "responses": {
          "200": {"description": "Session deleted"},
          "404": {"description": "Session not found"},
          "409": {"description": "Session is busy"}
        }
      }
    },
    "/v1/sessions/{session_id}/chat": {
      "post": {
        "summary": "Stateful chat in a session",
        "parameters": [
          {"name": "session_id", "in": "path", "required": true,
           "schema": {"type": "string"}}
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {"$ref": "#/components/schemas/SessionChatRequest"}
            }
          }
        },
        "responses": {
          "200": {"description": "Chat completion or SSE stream"},
          "400": {"description": "Invalid request"},
          "404": {"description": "Session not found"},
          "409": {"description": "Session is busy"}
        }
      }
    },
    "/openapi.json": {
      "get": {
        "summary": "OpenAPI specification",
        "responses": {
          "200": {"description": "This document"}
        }
      }
    }
  },
  "components": {
    "schemas": {
      "ChatMessage": {
        "type": "object",
        "required": ["role", "content"],
        "properties": {
          "role": {"type": "string", "enum": ["system", "user", "assistant"]},
          "content": {"type": "string"}
        }
      },
      "ChatRequest": {
        "type": "object",
        "required": ["messages"],
        "properties": {
          "model": {"type": "string"},
          "messages": {
            "type": "array",
            "items": {"$ref": "#/components/schemas/ChatMessage"}
          },
          "stream": {"type": "boolean"},
          "max_tokens": {"type": "integer"},
          "temperature": {"type": "number"},
          "top_p": {"type": "number"},
          "stop": {
            "oneOf": [
              {"type": "string"},
              {"type": "array", "items": {"type": "string"}}
            ]
          }
        }
      },
      "CreateSessionRequest": {
        "type": "object",
        "properties": {
          "session_id": {"type": "string"},
          "system_prompt": {"type": "string"}
        }
      },
      "SessionChatRequest": {
        "type": "object",
        "required": ["message"],
        "properties": {
          "message": {"type": "string"},
          "stream": {"type": "boolean"},
          "max_tokens": {"type": "integer"},
          "stop": {
            "oneOf": [
              {"type": "string"},
              {"type": "array", "items": {"type": "string"}}
            ]
          }
        }
      }
    }
  }
})json";

}  // namespace

const C8* OpenApiSpecJson() { return kOpenApiSpec; }

}  // namespace service
}  // namespace mobilenet
