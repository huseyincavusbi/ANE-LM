#pragma once

#include "../models/llm/qwen3_5.h"
#include "../core/tokenizer.h"
#include "../core/sampling.h"
#include <string>
#include <atomic>

namespace ane_lm {

// Minimal single-threaded HTTP inference server.
// Handles one request at a time — appropriate since ANE serialises compute.
//
// Endpoints:
//   GET  /health           → 200 OK {"status":"ok"}
//   POST /generate         → SSE stream of tokens
//   POST /chat             → SSE stream, multi-turn message history
//
// POST body (JSON):
//   {
//     "prompt":      "...",          // for /generate
//     "messages":    [{"role":"user","content":"..."},...], // for /chat
//     "max_tokens":  100,
//     "temperature": 0.6,
//     "rep_penalty": 1.2,
//     "stream":      true
//   }
//
// SSE format:
//   data: {"text":"...", "token":42}\n\n
//   data: [DONE]\n\n

class HttpServer {
public:
    HttpServer(LLMModel& model, Tokenizer& tokenizer, int port = 8080);
    ~HttpServer();

    // Blocks until stop() is called (e.g. from signal handler).
    void run();
    void stop();

    int port() const { return port_; }

private:
    LLMModel& model_;
    Tokenizer& tokenizer_;
    int port_;
    int server_fd_ = -1;
    std::atomic<bool> running_{false};

    void handle_client(int fd);
    void handle_generate(int fd, const std::string& body, bool is_chat);

    // HTTP helpers
    static std::string read_headers(int fd);
    static std::string read_body(int fd, const std::string& headers);
    static int parse_content_length(const std::string& headers);
    static std::string parse_method_path(const std::string& headers,
                                          std::string& out_method, std::string& out_path);

    static void send_headers(int fd, int status, const std::string& content_type,
                              bool is_sse, int content_length = -1);
    static void send_sse(int fd, const std::string& data);
    static void send_json(int fd, int status, const std::string& json_body);
    static bool write_all(int fd, const char* buf, size_t len);
};

} // namespace ane_lm
